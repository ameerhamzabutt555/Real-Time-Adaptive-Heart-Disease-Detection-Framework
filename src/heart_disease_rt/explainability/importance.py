"""Explainability helpers for baseline model interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from heart_disease_rt.data.schema import FEATURE_COLUMNS


def _get_pipeline_feature_names(model: object) -> list[str] | None:
    named_steps = getattr(model, "named_steps", None)
    if not isinstance(named_steps, dict):
        return None
    prep = named_steps.get("prep")
    if prep is None or not hasattr(prep, "get_feature_names_out"):
        return None
    try:
        names = prep.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return None


def compute_feature_importance(
    model: object,
    x_frame: pd.DataFrame,
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute global feature importance for the baseline model.

    Strategy:
    - If the model is linear with coefficients, return absolute coefficients aligned with
      the transformed feature space (e.g., one-hot encoded features).
    - Otherwise, if y_true is provided, use permutation importance on the canonical features.
    """
    estimator = getattr(model, "named_steps", {}).get("model")
    feature_names = _get_pipeline_feature_names(model) or FEATURE_COLUMNS

    if estimator is not None and hasattr(estimator, "coef_"):
        coef = np.abs(np.asarray(estimator.coef_)[0])
        total = float(coef.sum()) if float(coef.sum()) > 0 else 1.0
        frame = pd.DataFrame(
            {
                "feature": feature_names[: len(coef)],
                "abs_coefficient": coef,
                "normalized_importance": coef / total,
            }
        )
        return frame.sort_values("normalized_importance", ascending=False).reset_index(drop=True)

    if y_true is None:
        raise ValueError("Permutation importance requires y_true for non-linear models.")

    # Permutation importance on original feature columns.
    x_eval = x_frame[FEATURE_COLUMNS].copy()
    result = permutation_importance(
        model,
        x_eval,
        y_true.to_numpy(),
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1,
    )
    imp = np.maximum(0.0, result.importances_mean.astype(float))
    total = float(imp.sum()) if float(imp.sum()) > 0 else 1.0
    frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "abs_coefficient": imp,
            "normalized_importance": imp / total,
        }
    )
    return frame.sort_values("normalized_importance", ascending=False).reset_index(drop=True)


def compute_local_explanations(
    model: object,
    x_frame: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Approximate local explanations via feature contributions (linear models only)."""
    named_steps = getattr(model, "named_steps", {})
    prep = named_steps.get("prep")
    estimator = named_steps.get("model")
    if prep is None or estimator is None or not hasattr(estimator, "coef_"):
        raise ValueError("Local explanations are only supported for linear models.")

    sample = x_frame[FEATURE_COLUMNS].head(max(1, top_k)).copy()
    transformed = prep.transform(sample)
    coef = np.asarray(estimator.coef_)[0]
    feature_names = _get_pipeline_feature_names(model) or FEATURE_COLUMNS

    rows: list[dict[str, float | int | str]] = []
    for idx in range(len(sample)):
        contrib_vec = transformed[idx]
        contributions = np.asarray(contrib_vec).reshape(-1) * coef
        ordered = np.argsort(np.abs(contributions))[::-1][:top_k]
        for rank, col_idx in enumerate(ordered, start=1):
            rows.append(
                {
                    "sample_index": int(sample.index[idx]),
                    "rank": rank,
                    "feature": str(feature_names[col_idx]) if col_idx < len(feature_names) else f"feature_{col_idx}",
                    "value": float(sample.iloc[idx, 0]),
                    "contribution": float(contributions[col_idx]),
                    "abs_contribution": float(abs(contributions[col_idx])),
                }
            )
    return pd.DataFrame(rows)
