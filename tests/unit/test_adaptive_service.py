from heart_disease_rt.data.schema import FEATURE_COLUMNS
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


def test_all_canonical_features_tracked_on_predict_and_learn() -> None:
    service = AdaptiveHeartDiseaseService(detector_name="adwin")
    patient = _sample_patient()
    service.predict(patient)
    service.learn(patient, true_label=1)
    streams = service.feature_streams_dict()
    assert set(streams.keys()) == set(FEATURE_COLUMNS)
    for name in FEATURE_COLUMNS:
        assert streams[name].n == 2, name


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


def test_buffered_learn_applies_on_nth_sample() -> None:
    service = AdaptiveHeartDiseaseService(detector_name="adwin", commit_batch_size=3)
    patient = _sample_patient()

    r1 = service.learn(patient, true_label=1)
    assert r1.seen_samples == 0
    assert r1.learn_pending == 1
    assert r1.learn_applied_now == 0

    r2 = service.learn(patient, true_label=0)
    assert r2.seen_samples == 0
    assert r2.learn_pending == 2
    assert r2.learn_applied_now == 0

    r3 = service.learn(patient, true_label=1)
    assert r3.seen_samples == 3
    assert r3.learn_pending == 0
    assert r3.learn_applied_now == 3


def test_flush_applies_partial_buffer() -> None:
    service = AdaptiveHeartDiseaseService(commit_batch_size=10)
    patient = _sample_patient()
    service.learn(patient, true_label=1)
    service.learn(patient, true_label=0)
    assert service.status().learn_pending == 2

    flushed = service.flush_learn_buffer()
    assert flushed.flushed == 2
    assert service.status().seen_samples == 2
    assert service.status().learn_pending == 0


def test_checkpoint_roundtrip(tmp_path) -> None:
    ck = tmp_path / "ck"
    service = AdaptiveHeartDiseaseService(detector_name="adwin", commit_batch_size=1, checkpoint_dir=ck)
    patient = _sample_patient()
    service.learn(patient, true_label=1)
    path = service.save_checkpoint()
    assert path.is_file()

    other = AdaptiveHeartDiseaseService(detector_name="adwin", commit_batch_size=1, checkpoint_dir=ck)
    other.learn(patient, true_label=0)
    other.learn(patient, true_label=1)
    assert other.status().seen_samples == 2

    assert other.feature_streams_dict()["age"].n == 2
    other.restore_checkpoint(path)
    assert other.status().seen_samples == 1
    assert other.feature_streams_dict()["age"].n == 1
