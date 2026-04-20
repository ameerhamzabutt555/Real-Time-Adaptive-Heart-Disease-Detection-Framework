from __future__ import annotations

from pathlib import Path

from heart_disease_rt.serving.config import load_model_config


def test_load_model_config_fallback_when_missing() -> None:
    config = load_model_config(Path("/tmp/does-not-exist-model-config.json"))
    assert config.threshold == 0.5
    assert "models/artifacts/baseline.joblib" in str(config.artifact_path)

