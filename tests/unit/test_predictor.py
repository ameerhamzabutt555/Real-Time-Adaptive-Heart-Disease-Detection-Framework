from pathlib import Path

from heart_disease_rt.serving.config import ServingConfig
from heart_disease_rt.serving.predictor import HeartDiseasePredictor
from heart_disease_rt.serving.schemas import PatientRecord


def test_predictor_returns_probability_in_range() -> None:
    predictor = HeartDiseasePredictor(
        ServingConfig(
            artifact_path=Path("/tmp/non-existent-baseline.joblib"),
            threshold=0.5,
            model_version="test-version",
        )
    )
    record = PatientRecord(
        age=54,
        sex=1,
        resting_bp=130.0,
        cholesterol=240.0,
        max_heart_rate=150.0,
        fasting_blood_sugar=0,
        exercise_angina=1,
        chest_pain_type=2,
        oldpeak=1.2,
    )

    result = predictor.predict(record)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.model_version in {"test-version", "fallback-heuristic-v0"}


def test_predictor_has_serving_metadata() -> None:
    predictor = HeartDiseasePredictor(
        ServingConfig(
            artifact_path=Path("/tmp/non-existent-baseline.joblib"),
            threshold=0.67,
            model_version="test-version",
        )
    )
    record = PatientRecord(
        age=50,
        sex=0,
        resting_bp=120.0,
        cholesterol=200.0,
        max_heart_rate=160.0,
        restecg=1,
        slope=1,
        ca=0,
        thal=2,
    )
    result = predictor.predict(record)
    assert result.threshold == 0.67
