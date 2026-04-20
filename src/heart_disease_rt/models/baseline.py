from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass
class BaselineArtifact:
    model: Pipeline
    metrics: dict[str, float]


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> BaselineArtifact:
    """Train and evaluate baseline logistic regression."""
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    return BaselineArtifact(model=pipeline, metrics=metrics)


def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def run_baseline_training(
    frame: pd.DataFrame,
    artifact_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineArtifact:
    """Split, train, evaluate, and optionally persist baseline model."""
    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[LABEL_COLUMN],
    )
    artifact = train_baseline(train_df, test_df)
    if artifact_path is not None:
        save_model(artifact.model, artifact_path)
    return artifact
