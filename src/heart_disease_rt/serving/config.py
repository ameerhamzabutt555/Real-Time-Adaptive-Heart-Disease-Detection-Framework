from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/model_config.json")


@dataclass(frozen=True)
class ServingConfig:
    artifact_path: Path
    threshold: float
    model_version: str

    # Backward-compatible aliases
    @property
    def model_artifact_path(self) -> Path:
        return self.artifact_path

    @property
    def classification_threshold(self) -> float:
        return self.threshold


def _clamp_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_serving_config_from_file(config_path: Path = DEFAULT_CONFIG_PATH) -> ServingConfig:
    """Load serving config from JSON file with robust defaults."""
    if not config_path.exists():
        return ServingConfig(
            artifact_path=Path("models/artifacts/baseline.joblib"),
            threshold=0.5,
            model_version="baseline-v1",
        )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_value = payload.get("model_artifact_path", "models/artifacts/baseline.joblib")
    threshold_value = payload.get("probability_threshold", payload.get("threshold", 0.5))
    version_value = payload.get("model_version", "baseline-v1")
    return ServingConfig(
        artifact_path=Path(str(artifact_value)),
        threshold=_clamp_threshold(float(threshold_value)),
        model_version=str(version_value),
    )


def load_serving_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ServingConfig:
    """Load config from file and apply environment overrides."""
    base = load_serving_config_from_file(config_path=config_path)
    return ServingConfig(
        artifact_path=Path(os.getenv("MODEL_ARTIFACT_PATH", str(base.artifact_path))),
        threshold=_clamp_threshold(float(os.getenv("MODEL_THRESHOLD", str(base.threshold))),
        ),
        model_version=os.getenv("MODEL_VERSION", base.model_version),
    )


def save_serving_config(
    output_path: Path,
    model_artifact_path: Path,
    threshold: float,
    model_version: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_artifact_path": str(model_artifact_path),
        "probability_threshold": _clamp_threshold(threshold),
        "model_version": model_version,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Compatibility aliases for older modules/tests.
ModelConfig = ServingConfig


def load_model_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ServingConfig:
    return load_serving_config_from_file(config_path=config_path)


def save_model_config(
    output_path: Path,
    artifact_path: Path,
    probability_threshold: float,
    model_version: str,
) -> None:
    save_serving_config(
        output_path=output_path,
        model_artifact_path=artifact_path,
        threshold=probability_threshold,
        model_version=model_version,
    )
