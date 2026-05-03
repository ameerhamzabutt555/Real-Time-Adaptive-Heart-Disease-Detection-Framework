from heart_disease_rt.serving.adaptive_service import AdaptiveHeartDiseaseService
from heart_disease_rt.serving.schemas import PatientRecord


def _sample_patient() -> PatientRecord:
    return PatientRecord(
        age=54,
        sex=1,
        resting_bp=130.0,
        cholesterol=240.0,
        max_heart_rate=150.0,
        fasting_blood_sugar=0,
        exercise_angina=1,
        chest_pain_type=2,
        oldpeak=1.2,
        restecg=1,
        slope=1,
        ca=0,
        thal=2,
    )


def test_adaptive_service_learn_updates_counters() -> None:
    service = AdaptiveHeartDiseaseService(detector_name="adwin")
    patient = _sample_patient()

    before = service.status()
    assert before.seen_samples == 0
    assert before.drift_events == 0

    learn = service.learn(patient, true_label=1)
    assert learn.seen_samples == 1
    assert 0.0 <= learn.online_accuracy <= 1.0
    assert 0.0 <= learn.online_f1 <= 1.0
    assert learn.detector == "adwin"

    after = service.status()
    assert after.seen_samples == 1
    assert after.drift_events >= 0


def test_adaptive_service_rejects_invalid_threshold_override() -> None:
    service = AdaptiveHeartDiseaseService()
    patient = _sample_patient()
    try:
        service.predict(patient, threshold_override=1.5)
    except ValueError as exc:
        assert "threshold_override" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range threshold_override")
