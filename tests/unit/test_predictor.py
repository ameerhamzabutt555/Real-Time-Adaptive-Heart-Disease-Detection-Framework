from heart_disease_rt.serving.predictor import HeartDiseasePredictor
from heart_disease_rt.serving.schemas import PatientRecord


def test_predictor_returns_probability_in_range() -> None:
    predictor = HeartDiseasePredictor()
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
