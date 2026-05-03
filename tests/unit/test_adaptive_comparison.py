import pandas as pd

from heart_disease_rt.models.adaptive import SUPPORTED_DETECTORS, compare_drift_detectors


def test_compare_drift_detectors_has_all_detectors() -> None:
    frame = pd.DataFrame(
        {
            "age": [44, 57, 60, 48, 50, 65],
            "sex": [1, 0, 1, 1, 0, 1],
            "chest_pain_type": [1, 2, 3, 0, 2, 3],
            "resting_bp": [120.0, 130.0, 145.0, 118.0, 125.0, 150.0],
            "cholesterol": [200.0, 250.0, 290.0, 180.0, 230.0, 300.0],
            "fasting_blood_sugar": [0, 1, 1, 0, 0, 1],
            "restecg": [0, 1, 1, 0, 1, 1],
            "max_heart_rate": [170.0, 150.0, 140.0, 175.0, 160.0, 135.0],
            "exercise_angina": [0, 1, 1, 0, 1, 1],
            "oldpeak": [0.0, 1.2, 2.4, 0.2, 0.8, 2.6],
            "slope": [1, 1, 2, 0, 1, 2],
            "ca": [0, 1, 2, 0, 1, 2],
            "thal": [2, 3, 3, 2, 2, 3],
            "label": [0, 1, 1, 0, 1, 1],
        }
    )

    summary_df, progress_df = compare_drift_detectors(frame, detectors=SUPPORTED_DETECTORS)
    assert set(summary_df["detector"]) == set(SUPPORTED_DETECTORS)
    assert len(progress_df) == len(frame) * len(SUPPORTED_DETECTORS)
