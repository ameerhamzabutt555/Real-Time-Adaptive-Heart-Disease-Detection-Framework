from __future__ import annotations

from river import compose, drift, linear_model, metrics, preprocessing

from heart_disease_rt.data.schema import FEATURE_COLUMNS
from heart_disease_rt.serving.schemas import (
    AdaptiveLearnResponse,
    AdaptiveStatusResponse,
    PatientRecord,
    PredictionResponse,
)


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

    def __init__(
        self,
        detector_name: str = "adwin",
        threshold: float = 0.5,
        model_version: str = "adaptive-online-v1",
    ) -> None:
        detector, normalized_name = _build_detector(detector_name)
        self._model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(),
        )
        self._detector = detector
        self._detector_name = normalized_name
        self._threshold = self._clamp_threshold(threshold)
        self._model_version = model_version
        self._seen_samples = 0
        self._drift_events = 0
        self._metric_acc = metrics.Accuracy()
        self._metric_f1 = metrics.F1()

    @staticmethod
    def _clamp_threshold(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _to_features(record: PatientRecord) -> dict[str, float]:
        payload = record.model_dump()
        return {name: float(payload[name]) for name in FEATURE_COLUMNS}

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

    def predict(self, record: PatientRecord, threshold_override: float | None = None) -> PredictionResponse:
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
        features = self._to_features(record)
        score, predicted_label, decision_threshold = self._score_label(
            record,
            threshold_override=threshold_override,
        )

        self._metric_acc.update(true_label, predicted_label)
        self._metric_f1.update(true_label, predicted_label)

        error_signal = int(predicted_label != true_label)
        self._detector.update(error_signal)
        drift_detected = bool(self._detector.drift_detected)
        if drift_detected:
            self._drift_events += 1

        self._model.learn_one(features, true_label)
        self._seen_samples += 1

        return AdaptiveLearnResponse(
            risk_score=round(score, 4),
            predicted_label=predicted_label,
            model_version=self._model_version,
            threshold=decision_threshold,
            model_loaded=True,
            true_label=true_label,
            detector=self._detector_name,
            drift_detected=drift_detected,
            seen_samples=self._seen_samples,
            online_accuracy=float(self._metric_acc.get()),
            online_f1=float(self._metric_f1.get()),
            drift_events=self._drift_events,
        )

    def status(self) -> AdaptiveStatusResponse:
        return AdaptiveStatusResponse(
            model_version=self._model_version,
            detector=self._detector_name,
            threshold=self._threshold,
            seen_samples=self._seen_samples,
            drift_events=self._drift_events,
            online_accuracy=float(self._metric_acc.get()),
            online_f1=float(self._metric_f1.get()),
        )
