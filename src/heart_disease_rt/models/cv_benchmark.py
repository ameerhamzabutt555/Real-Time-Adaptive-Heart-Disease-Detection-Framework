"""Repeated cross-validation benchmarks for fair literature comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def _build_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def run_repeated_cv_benchmark(
    frame: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold-level and summary-level metrics for fair model comparison."""
    x = frame[FEATURE_COLUMNS].copy()
    y = frame[LABEL_COLUMN].astype(int).copy()

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    models = _build_models(random_state=random_state)
    fold_rows: list[dict[str, float | int | str]] = []

    fold_id = 0
    for train_idx, test_idx in cv.split(x, y):
        fold_id += 1
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(x_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            fold_rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, y_prob)),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby("model", as_index=False)
        .agg(
            folds=("fold", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .sort_values(by="accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )
    return fold_df, summary_df

