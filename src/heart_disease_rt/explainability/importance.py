"""Explainability helpers for baseline model interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heart_disease_rt.data.schema import FEATURE_COLUMNS


def compute_feature_importance(model: object, x_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute coefficient-based global importance for logistic model pipeline."""
    del x_frame
    estimator = getattr(model, "named_steps", {}).get("model")
    if estimator is None or not hasattr(estimator, "coef_"):
        raise ValueError("Expected sklearn pipeline with linear model coefficients.")

    coef = np.abs(estimator.coef_[0])
    total = float(coef.sum()) if float(coef.sum()) > 0 else 1.0
    frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "abs_coefficient": coef,
            "normalized_importance": coef / total,
        }
    )
    return frame.sort_values("normalized_importance", ascending=False).reset_index(drop=True)


def compute_local_explanations(
    model: object,
    x_frame: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Approximate local explanations for first few samples via feature contributions."""
    scaler = getattr(model, "named_steps", {}).get("scaler")
    estimator = getattr(model, "named_steps", {}).get("model")
    if scaler is None or estimator is None or not hasattr(estimator, "coef_"):
        raise ValueError("Expected pipeline with scaler and logistic model.")

    sample = x_frame[FEATURE_COLUMNS].head(max(1, top_k)).copy()
    transformed = scaler.transform(sample)
    coef = estimator.coef_[0]

    rows: list[dict[str, float | int | str]] = []
    for idx in range(len(sample)):
        contributions = transformed[idx] * coef
        ordered = np.argsort(np.abs(contributions))[::-1][:top_k]
        for rank, col_idx in enumerate(ordered, start=1):
            rows.append(
                {
                    "sample_index": int(sample.index[idx]),
                    "rank": rank,
                    "feature": FEATURE_COLUMNS[col_idx],
                    "value": float(sample.iloc[idx, col_idx]),
                    "contribution": float(contributions[col_idx]),
                    "abs_contribution": float(abs(contributions[col_idx])),
                }
            )
    return pd.DataFrame(rows)
