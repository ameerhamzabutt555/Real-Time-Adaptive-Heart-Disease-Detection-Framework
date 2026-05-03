from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from heart_disease_rt.data.schema import FEATURE_COLUMNS
from heart_disease_rt.serving.config import ServingConfig, load_serving_config
from heart_disease_rt.serving.schemas import PatientRecord, PredictionResponse


@dataclass
class LoadedModel:
    model: object
    artifact_path: str


class HeartDiseasePredictor:
    """Model-backed predictor with deterministic fallback for cold start."""

    def __init__(self, model_config: ServingConfig | None = None) -> None:
        self.model_config = model_config or load_serving_config()
        self._loaded: LoadedModel | None = None

    def _try_load_model(self) -> LoadedModel | None:
        artifact_path = self.model_config.artifact_path
        if self._loaded is not None and self._loaded.artifact_path == str(artifact_path):
            return self._loaded
        if not artifact_path.exists():
            return None
        model = joblib.load(artifact_path)
        self._loaded = LoadedModel(model=model, artifact_path=str(artifact_path))
        return self._loaded

    @staticmethod
    def _to_feature_frame(record: PatientRecord) -> pd.DataFrame:
        payload = record.model_dump()
        ordered = {name: payload[name] for name in FEATURE_COLUMNS}
        return pd.DataFrame([ordered], columns=FEATURE_COLUMNS)

    def _fallback_score(self, record: PatientRecord) -> float:
        return min(
            1.0,
            max(
                0.0,
                (0.002 * record.age)
                + (0.001 * record.resting_bp)
                + (0.0005 * record.cholesterol),
            ),
        )

    @property
    def threshold(self) -> float:
        return float(self.model_config.classification_threshold)

    @property
    def model_version(self) -> str:
        return str(self.model_config.model_version)

    @property
    def model_loaded(self) -> bool:
        return self._try_load_model() is not None

    def model_info(self) -> dict[str, object]:
        loaded = self._try_load_model()
        return {
            "model_version": self.model_version,
            "threshold": self.threshold,
            "artifact_path": str(self.model_config.artifact_path),
            "model_loaded": loaded is not None,
        }

    def predict(self, record: PatientRecord, threshold_override: float | None = None) -> PredictionResponse:
        loaded = self._try_load_model()
        score: float
        model_version: str

        if loaded is None:
            score = self._fallback_score(record)
            model_version = "fallback-heuristic-v0"
        else:
            features = self._to_feature_frame(record)
            if hasattr(loaded.model, "predict_proba"):
                raw = loaded.model.predict_proba(features)[0][1]
            else:
                pred = loaded.model.predict(features)[0]
                raw = float(pred)
            score = float(np.clip(raw, 0.0, 1.0))
            model_version = self.model_version

        decision_threshold = threshold_override if threshold_override is not None else self.threshold
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("threshold_override must be in [0, 1].")
        label = int(score >= decision_threshold)
        return PredictionResponse(
            risk_score=round(score, 4),
            predicted_label=label,
            model_version=model_version,
            threshold=decision_threshold,
            model_loaded=loaded is not None,
        )
