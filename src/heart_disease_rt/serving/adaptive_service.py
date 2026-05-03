from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from river import compose, drift, forest, linear_model, metrics, preprocessing

from heart_disease_rt.data.preprocess import preprocess_dataframe
from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN
from heart_disease_rt.serving.schemas import (
    AdaptiveFeatureTrackingResponse,
    AdaptiveLearnFlushResponse,
    AdaptiveLearnResponse,
    AdaptiveObserveResponse,
    AdaptiveStatusResponse,
    FeatureStreamStats,
    PatientRecord,
    PredictionResponse,
)

_CHECKPOINT_NAME = re.compile(r"^checkpoint_\d{5}\.pkl$")


def _build_detector(detector_name: str):
    normalized = detector_name.strip().lower()
    if normalized == "adwin":
        return drift.ADWIN(), "adwin"
    if normalized == "ddm":
        return drift.binary.DDM(), "ddm"
    if normalized in {"page_hinkley", "pagehinkley", "ph"}:
        return drift.PageHinkley(), "page_hinkley"
    raise ValueError("Unsupported detector. Use adwin, ddm, or page_hinkley.")


class AdaptiveHeartDiseaseService:
    """In-memory adaptive model for online prediction and learning."""

    @dataclass
    class _PatientState:
        seen: int = 0

    @dataclass
    class _Welford:
        """Online mean / variance for one feature stream."""

        n: int = 0
        mean: float = 0.0
        m2: float = 0.0
        last: float = 0.0

        def update(self, x: float) -> None:
            self.last = float(x)
            self.n += 1
            delta = float(x) - self.mean
            self.mean += delta / self.n
            delta2 = float(x) - self.mean
            self.m2 += delta * delta2

        def variance(self) -> float:
            if self.n < 2:
                return 0.0
            return max(0.0, self.m2 / (self.n - 1))

    def __init__(
        self,
        detector_name: str = "adwin",
        threshold: float = 0.5,
        model_version: str = "adaptive-online-v1",
        on_drift: str = "log_only",
        model_type: str = "logreg",
        commit_batch_size: int = 1,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        self._detector_config = detector_name
        detector, normalized_name = _build_detector(detector_name)
        self._detector = detector
        self._detector_name = normalized_name
        self._model_type = model_type.strip().lower()
        self._model = self._build_model(self._model_type)
        self._threshold = self._clamp_threshold(threshold)
        self._model_version = model_version
        self._on_drift = on_drift.strip().lower()
        self._seen_samples = 0
        self._drift_events = 0
        self._metric_acc = metrics.Accuracy()
        self._metric_f1 = metrics.F1()
        self._patient_state: dict[str, AdaptiveHeartDiseaseService._PatientState] = {}
        self._commit_batch_size = max(1, int(commit_batch_size))
        self._learn_buffer: list[tuple[PatientRecord, int, float | None]] = []
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._feature_welford: dict[str, AdaptiveHeartDiseaseService._Welford] = {
            name: AdaptiveHeartDiseaseService._Welford() for name in FEATURE_COLUMNS
        }

    @staticmethod
    def _build_model(model_type: str):
        normalized = model_type.strip().lower()
        if normalized in {"logreg", "logistic_regression"}:
            return compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression(),
            )
        if normalized in {"arf", "adaptive_random_forest"}:
            return forest.ARFClassifier(n_models=15, seed=42)
        raise ValueError("Unsupported adaptive model_type. Use 'logreg' or 'arf'.")

    def _reset_after_drift(self) -> None:
        """Reset adaptive components after drift (policy action)."""
        detector, normalized_name = _build_detector(self._detector_config)
        self._detector = detector
        self._detector_name = normalized_name
        self._model = self._build_model(self._model_type)
        self._metric_acc = metrics.Accuracy()
        self._metric_f1 = metrics.F1()

    @staticmethod
    def _clamp_threshold(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _to_features(record: PatientRecord) -> dict[str, float]:
        payload = record.model_dump()
        return {name: float(payload[name]) for name in FEATURE_COLUMNS}

    def _update_feature_streams(self, record: PatientRecord) -> None:
        for name, value in self._to_features(record).items():
            self._feature_welford[name].update(value)

    def _update_feature_streams_from_row(self, row: pd.Series) -> None:
        for name in FEATURE_COLUMNS:
            self._feature_welford[name].update(float(row[name]))

    def feature_streams_dict(self) -> dict[str, FeatureStreamStats]:
        return {
            name: FeatureStreamStats(
                n=w.n,
                mean=round(w.mean, 6),
                variance=round(w.variance(), 6),
                last=round(w.last, 6),
            )
            for name, w in self._feature_welford.items()
        }

    def feature_tracking_response(self) -> AdaptiveFeatureTrackingResponse:
        return AdaptiveFeatureTrackingResponse(streams=self.feature_streams_dict())

    def _serialize_feature_welford(self) -> dict[str, tuple[int, float, float, float]]:
        return {name: (w.n, w.mean, w.m2, w.last) for name, w in self._feature_welford.items()}

    def _deserialize_feature_welford(self, data: dict[str, object]) -> None:
        for name in FEATURE_COLUMNS:
            if name not in data:
                continue
            tup = data[name]
            if not hasattr(tup, "__getitem__") or len(tup) != 4:
                continue
            n, mean, m2, last = int(tup[0]), float(tup[1]), float(tup[2]), float(tup[3])
            w = AdaptiveHeartDiseaseService._Welford()
            w.n, w.mean, w.m2, w.last = n, mean, m2, last
            self._feature_welford[name] = w

    def _resolve_threshold(self, threshold_override: float | None) -> float:
        if threshold_override is None:
            return self._threshold
        if not 0.0 <= threshold_override <= 1.0:
            raise ValueError("threshold_override must be in [0, 1].")
        return float(threshold_override)

    def _score_label(self, record: PatientRecord, threshold_override: float | None = None) -> tuple[float, int, float]:
        features = self._to_features(record)
        proba = self._model.predict_proba_one(features)
        score = float(proba.get(1, 0.0)) if proba else 0.0
        score = min(1.0, max(0.0, score))
        decision_threshold = self._resolve_threshold(threshold_override)
        label = int(score >= decision_threshold)
        return score, label, decision_threshold

    def _apply_learn_sample(
        self,
        record: PatientRecord,
        true_label: int,
        threshold_override: float | None,
    ) -> bool:
        """Apply one labeled online step. Returns whether drift fired on this step."""
        features = self._to_features(record)
        _, predicted_label, _ = self._score_label(record, threshold_override=threshold_override)

        self._metric_acc.update(true_label, predicted_label)
        self._metric_f1.update(true_label, predicted_label)

        error_signal = int(predicted_label != true_label)
        self._detector.update(error_signal)
        drift_detected = bool(self._detector.drift_detected)
        if drift_detected:
            self._drift_events += 1
            if self._on_drift == "reset":
                self._reset_after_drift()

        self._model.learn_one(features, true_label)
        self._seen_samples += 1
        return drift_detected

    def _flush_full_batches(self) -> tuple[int, bool]:
        """Apply queued samples in chunks of commit_batch_size. Returns (applied_count, last_drift)."""
        applied = 0
        last_drift = False
        while len(self._learn_buffer) >= self._commit_batch_size:
            batch = self._learn_buffer[: self._commit_batch_size]
            del self._learn_buffer[: self._commit_batch_size]
            for rec, y, th in batch:
                last_drift = self._apply_learn_sample(rec, y, th)
            applied += len(batch)
            self._maybe_save_checkpoint()
        return applied, last_drift

    def _maybe_save_checkpoint(self) -> Path | None:
        if self._checkpoint_dir is None:
            return None
        return self.save_checkpoint()

    def predict(self, record: PatientRecord, threshold_override: float | None = None) -> PredictionResponse:
        self._update_feature_streams(record)
        score, label, decision_threshold = self._score_label(record, threshold_override=threshold_override)
        return PredictionResponse(
            risk_score=round(score, 4),
            predicted_label=label,
            model_version=self._model_version,
            threshold=decision_threshold,
            model_loaded=True,
        )

    def learn(
        self,
        record: PatientRecord,
        true_label: int,
        threshold_override: float | None = None,
    ) -> AdaptiveLearnResponse:
        if true_label not in (0, 1):
            raise ValueError("true_label must be 0 or 1.")
        self._update_feature_streams(record)
        score, predicted_label, decision_threshold = self._score_label(
            record,
            threshold_override=threshold_override,
        )
        self._learn_buffer.append((record, true_label, threshold_override))
        applied_now, last_drift = self._flush_full_batches()

        return AdaptiveLearnResponse(
            risk_score=round(score, 4),
            predicted_label=predicted_label,
            model_version=self._model_version,
            threshold=decision_threshold,
            model_loaded=True,
            true_label=true_label,
            detector=self._detector_name,
            drift_detected=last_drift,
            seen_samples=self._seen_samples,
            online_accuracy=float(self._metric_acc.get()),
            online_f1=float(self._metric_f1.get()),
            drift_events=self._drift_events,
            learn_pending=len(self._learn_buffer),
            learn_applied_now=applied_now,
        )

    def observe(
        self,
        record: PatientRecord,
        *,
        label: int | None = None,
        threshold_override: float | None = None,
        patient_id: str | None = None,
    ) -> AdaptiveObserveResponse:
        """Predict and optionally learn if a label is provided."""
        self._update_feature_streams(record)
        score, predicted_label, decision_threshold = self._score_label(record, threshold_override=threshold_override)

        drift_detected = False
        learned = False
        applied_now = 0

        if label is not None:
            if label not in (0, 1):
                raise ValueError("label must be 0 or 1.")

            self._learn_buffer.append((record, label, threshold_override))
            applied_now, drift_detected = self._flush_full_batches()
            learned = True

        if patient_id is not None:
            state = self._patient_state.get(patient_id) or AdaptiveHeartDiseaseService._PatientState()
            state.seen += 1
            self._patient_state[patient_id] = state

        return AdaptiveObserveResponse(
            risk_score=round(score, 4),
            predicted_label=predicted_label,
            model_version=self._model_version,
            threshold=decision_threshold,
            model_loaded=True,
            learned=learned,
            patient_id=patient_id,
            seen_samples=self._seen_samples,
            detector=self._detector_name,
            drift_detected=drift_detected,
            drift_events=self._drift_events,
            online_accuracy=float(self._metric_acc.get()),
            online_f1=float(self._metric_f1.get()),
            learn_pending=len(self._learn_buffer),
            learn_applied_now=applied_now,
        )

    def flush_learn_buffer(self) -> AdaptiveLearnFlushResponse:
        """Apply all queued labeled samples immediately (manual commit)."""
        flushed = 0
        while self._learn_buffer:
            rec, y, th = self._learn_buffer.pop(0)
            self._apply_learn_sample(rec, y, th)
            flushed += 1
        if flushed:
            self._maybe_save_checkpoint()
        return AdaptiveLearnFlushResponse(
            flushed=flushed,
            seen_samples=self._seen_samples,
            online_accuracy=float(self._metric_acc.get()),
            online_f1=float(self._metric_f1.get()),
            drift_events=self._drift_events,
        )

    def warm_start_from_csv(self, path: str | Path, *, limit: int | None = None) -> int:
        """Warm-start the online model from a labeled CSV (e.g., processed UCI data).

        The CSV can be raw or processed; we run the project preprocessing to normalize columns.
        Returns number of samples learned.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise ValueError(f"Warm-start CSV not found: {csv_path}")

        self._learn_buffer.clear()
        frame = pd.read_csv(csv_path)
        processed = preprocess_dataframe(frame)
        if limit is not None:
            processed = processed.head(int(limit))

        learned = 0
        for _, row in processed.iterrows():
            features = {name: float(row[name]) for name in FEATURE_COLUMNS}
            label = int(row[LABEL_COLUMN])
            self._update_feature_streams_from_row(row)
            self._model.learn_one(features, label)
            learned += 1

        return learned

    def status(self) -> AdaptiveStatusResponse:
        return AdaptiveStatusResponse(
            model_version=self._model_version,
            model_type=self._model_type,
            detector=self._detector_name,
            threshold=self._threshold,
            seen_samples=self._seen_samples,
            drift_events=self._drift_events,
            online_accuracy=float(self._metric_acc.get()),
            online_f1=float(self._metric_f1.get()),
            learn_pending=len(self._learn_buffer),
            commit_batch_size=self._commit_batch_size,
            checkpoint_dir=str(self._checkpoint_dir) if self._checkpoint_dir else None,
            feature_streams=self.feature_streams_dict(),
        )

    def _next_checkpoint_path(self) -> Path:
        assert self._checkpoint_dir is not None
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        highest = 0
        for p in self._checkpoint_dir.glob("checkpoint_*.pkl"):
            m = re.match(r"checkpoint_(\d{5})\.pkl$", p.name)
            if m:
                highest = max(highest, int(m.group(1)))
        return self._checkpoint_dir / f"checkpoint_{highest + 1:05d}.pkl"

    def save_checkpoint(self) -> Path:
        """Persist current model + metrics + counters. Requires checkpoint_dir."""
        if self._checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not configured.")
        path = self._next_checkpoint_path()
        patient_map = {pid: st.seen for pid, st in self._patient_state.items()}
        payload = {
            "model": self._model,
            "detector": self._detector,
            "detector_name": self._detector_name,
            "metric_acc": self._metric_acc,
            "metric_f1": self._metric_f1,
            "seen_samples": self._seen_samples,
            "drift_events": self._drift_events,
            "patient_state": patient_map,
            "threshold": self._threshold,
            "model_type": self._model_type,
            "detector_config": self._detector_config,
            "on_drift": self._on_drift,
            "model_version": self._model_version,
            "feature_welford": self._serialize_feature_welford(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def _restore_payload(self, payload: dict) -> None:
        self._model = payload["model"]
        self._detector = payload["detector"]
        self._detector_name = str(payload.get("detector_name", self._detector_name))
        self._metric_acc = payload["metric_acc"]
        self._metric_f1 = payload["metric_f1"]
        self._seen_samples = int(payload["seen_samples"])
        self._drift_events = int(payload["drift_events"])
        self._threshold = self._clamp_threshold(float(payload.get("threshold", self._threshold)))
        self._model_type = str(payload.get("model_type", self._model_type)).strip().lower()
        self._detector_config = str(payload.get("detector_config", self._detector_config))
        self._on_drift = str(payload.get("on_drift", self._on_drift)).strip().lower()
        self._model_version = str(payload.get("model_version", self._model_version))
        patient_map = payload.get("patient_state") or {}
        self._patient_state = {
            pid: AdaptiveHeartDiseaseService._PatientState(seen=int(cnt))
            for pid, cnt in patient_map.items()
        }
        self._learn_buffer.clear()
        raw_fw = payload.get("feature_welford")
        if isinstance(raw_fw, dict) and raw_fw:
            try:
                self._deserialize_feature_welford(raw_fw)
            except (TypeError, ValueError):
                self._feature_welford = {
                    name: AdaptiveHeartDiseaseService._Welford() for name in FEATURE_COLUMNS
                }
        else:
            self._feature_welford = {
                name: AdaptiveHeartDiseaseService._Welford() for name in FEATURE_COLUMNS
            }

    def restore_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Checkpoint not found: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Invalid checkpoint payload.")
        self._restore_payload(payload)

    def restore_checkpoint_by_filename(self, filename: str) -> None:
        if self._checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not configured.")
        name = Path(filename).name
        if not _CHECKPOINT_NAME.match(name):
            raise ValueError("filename must match checkpoint_XXXXX.pkl")
        base = self._checkpoint_dir.resolve()
        target = (base / name).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise ValueError("Invalid checkpoint path.") from exc
        if not target.is_file():
            raise ValueError(f"Checkpoint not found: {name}")
        self.restore_checkpoint(target)

    def try_load_latest_checkpoint(self) -> bool:
        """Load newest checkpoint_*.pkl from checkpoint_dir, if any."""
        if self._checkpoint_dir is None or not self._checkpoint_dir.is_dir():
            return False
        files = sorted(self._checkpoint_dir.glob("checkpoint_*.pkl"))
        if not files:
            return False
        self.restore_checkpoint(files[-1])
        return True

    def list_checkpoint_files(self) -> list[Path]:
        if self._checkpoint_dir is None or not self._checkpoint_dir.is_dir():
            return []
        return sorted(self._checkpoint_dir.glob("checkpoint_*.pkl"))
