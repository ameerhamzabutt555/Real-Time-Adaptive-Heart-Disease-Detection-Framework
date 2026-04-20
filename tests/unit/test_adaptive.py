import pandas as pd

from heart_disease_rt.models.adaptive import run_adaptive_training


def test_run_adaptive_training_returns_summary() -> None:
    frame = pd.DataFrame(
        {
            "age": [44, 57, 60, 48],
            "sex": [1, 0, 1, 1],
            "chest_pain_type": [1, 2, 3, 0],
            "resting_bp": [120.0, 130.0, 145.0, 118.0],
            "cholesterol": [200.0, 250.0, 290.0, 180.0],
            "fasting_blood_sugar": [0, 1, 1, 0],
            "restecg": [0, 1, 1, 0],
            "max_heart_rate": [170.0, 150.0, 140.0, 175.0],
            "exercise_angina": [0, 1, 1, 0],
            "oldpeak": [0.0, 1.2, 2.4, 0.2],
            "slope": [1, 1, 2, 0],
            "ca": [0, 1, 2, 0],
            "thal": [2, 3, 3, 2],
            "label": [0, 1, 1, 0],
        }
    )

    summary, progress = run_adaptive_training(frame)
    assert summary.steps == 4
    assert 0.0 <= summary.accuracy <= 1.0
    assert 0.0 <= summary.f1 <= 1.0
    assert len(progress) == 4
