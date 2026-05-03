from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN, NUMERIC_COLUMNS


@dataclass
class BaselineArtifact:
    model: Pipeline
    metrics: dict[str, float]
    y_true: list[int] | None = None
    y_pred: list[int] | None = None
    y_prob: list[float] | None = None


def _build_pipeline(model_type: str = "logreg") -> Pipeline:
    """Baseline pipeline.

    Notes:
    - Numeric features are scaled.
    - Categorical-coded integer features are one-hot encoded (better than scaling categories).
    """
    categorical_columns = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ],
        remainder="drop",
    )

    normalized = model_type.strip().lower()
    if normalized in {"logreg", "logistic_regression"}:
        model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
    elif normalized in {"hgb", "hist_gb", "hist_gradient_boosting"}:
        model = HistGradientBoostingClassifier(
            max_depth=None,
            learning_rate=0.05,
            max_iter=400,
            random_state=42,
        )
    else:
        raise ValueError("Unsupported model_type. Use 'logreg' or 'hgb'.")
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

def _tune_pipeline(train_df: pd.DataFrame, model_type: str) -> Pipeline:
    """Grid-search a few safe hyperparameters to improve accuracy without leakage."""
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]

    pipeline = _build_pipeline(model_type=model_type)
    normalized = model_type.strip().lower()
    if normalized in {"logreg", "logistic_regression"}:
        param_grid = {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0]}
    else:
        param_grid = {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [300, 600],
            "model__max_depth": [None, 3, 5],
        }
    # Small training sets (unit tests) may not have enough samples per class for 5-fold CV.
    class_counts = y_train.value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = min(5, min_class) if min_class > 1 else 0
    if n_splits < 2:
        pipeline.fit(x_train, y_train)
        return pipeline

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_

def _select_threshold_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick threshold that maximizes accuracy (validation-only)."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t

def train_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_type: str = "logreg",
    tune_threshold: bool = True,
) -> BaselineArtifact:
    """Train and evaluate a baseline model (optionally with threshold tuning)."""
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    inner_train, inner_val = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df[LABEL_COLUMN],
    )
    pipeline = _tune_pipeline(inner_train, model_type=model_type)

    threshold = 0.5
    if tune_threshold:
        val_prob = pipeline.predict_proba(inner_val[FEATURE_COLUMNS])[:, 1]
        threshold = _select_threshold_from_probs(inner_val[LABEL_COLUMN].to_numpy(), val_prob)

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "selected_threshold": float(threshold),
        "model_type": str(model_type),
    }
    return BaselineArtifact(
        model=pipeline,
        metrics=metrics,
        y_true=[int(value) for value in y_test.tolist()],
        y_pred=[int(value) for value in y_pred.tolist()],
        y_prob=[float(value) for value in y_prob.tolist()],
    )


def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def run_baseline_training(
    frame: pd.DataFrame,
    artifact_path: Path | None = None,
    threshold: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "logreg",
    tune_threshold: bool = True,
) -> BaselineArtifact:
    """Split, train, evaluate, and optionally persist baseline model."""
    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[LABEL_COLUMN],
    )
    artifact = train_baseline(train_df, test_df, model_type=model_type, tune_threshold=tune_threshold)
    if artifact_path is not None:
        save_model(artifact.model, artifact_path)
        save_model_metadata(
            artifact_path=artifact_path,
            metrics=artifact.metrics,
            threshold=float(artifact.metrics.get("selected_threshold", threshold)),
            model_version=f"baseline-{model_type}",
        )
    return artifact


def save_model_metadata(
    artifact_path: Path,
    metrics: dict[str, float],
    threshold: float = 0.5,
    model_version: str = "baseline-v1",
) -> Path:
    """Persist serving metadata next to the model artifact."""
    config = {
        "model_artifact_path": str(artifact_path),
        "probability_threshold": threshold,
        "model_version": model_version,
        "feature_order": FEATURE_COLUMNS,
        "metrics": metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config_path = Path("configs/model_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path
