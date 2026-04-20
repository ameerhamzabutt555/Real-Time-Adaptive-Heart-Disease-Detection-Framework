import pandas as pd

from heart_disease_rt.data.preprocess import preprocess_dataframe
from heart_disease_rt.data.schema import LABEL_COLUMN


def test_preprocess_dataframe_normalizes_columns_and_label() -> None:
    frame = pd.DataFrame(
        [
            {
                "age": 52,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 160,
                "exang": 0,
                "oldpeak": 1.1,
                "slope": 1,
                "ca": 0,
                "thal": 2,
                "target": 1,
            }
        ]
    )

    processed = preprocess_dataframe(frame)
    assert "resting_bp" in processed.columns
    assert "cholesterol" in processed.columns
    assert LABEL_COLUMN in processed.columns
    assert int(processed[LABEL_COLUMN].iloc[0]) == 1
