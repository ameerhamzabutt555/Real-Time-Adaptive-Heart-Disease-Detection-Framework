import pandas as pd

from heart_disease_rt.data.preprocess import preprocess_dataframe
from heart_disease_rt.models.baseline import run_baseline_training


def test_run_baseline_training_returns_metrics() -> None:
    frame = pd.DataFrame(
        {
            "age": [45, 54, 39, 60, 51, 48, 66, 58],
            "sex": [1, 1, 0, 1, 0, 1, 1, 0],
            "chest_pain_type": [0, 2, 1, 3, 2, 1, 0, 2],
            "resting_bp": [120, 130, 110, 140, 126, 128, 150, 135],
            "cholesterol": [210, 240, 190, 260, 230, 220, 280, 250],
            "fasting_blood_sugar": [0, 0, 0, 1, 0, 0, 1, 0],
            "restecg": [0, 1, 0, 1, 0, 1, 1, 0],
            "max_heart_rate": [165, 150, 172, 138, 155, 160, 130, 145],
            "exercise_angina": [0, 1, 0, 1, 1, 0, 1, 1],
            "oldpeak": [0.2, 1.2, 0.1, 2.0, 1.0, 0.5, 2.2, 1.8],
            "slope": [2, 1, 2, 1, 1, 2, 0, 1],
            "ca": [0, 1, 0, 2, 1, 0, 2, 1],
            "thal": [2, 2, 2, 3, 2, 2, 3, 3],
            "label": [0, 1, 0, 1, 1, 0, 1, 1],
        }
    )
    processed = preprocess_dataframe(frame)
    artifact = run_baseline_training(processed)
    assert 0.0 <= artifact.metrics["accuracy"] <= 1.0
    assert 0.0 <= artifact.metrics["f1"] <= 1.0
    assert 0.0 <= artifact.metrics["recall"] <= 1.0
