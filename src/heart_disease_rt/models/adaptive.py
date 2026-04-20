"""Adaptive/online training utilities using River."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from river import compose, drift, linear_model, metrics, preprocessing

from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass
class AdaptiveRunSummary:
    steps: int
    accuracy: float
    f1: float
    drift_events: int


def _iter_records(df: pd.DataFrame) -> Iterable[tuple[dict[str, float], int]]:
    for _, row in df.iterrows():
        x = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
        y = int(row[LABEL_COLUMN])
        yield x, y


def run_adaptive_training(df: pd.DataFrame) -> AdaptiveRunSummary:
    """Run prequential online training with drift-aware tracking."""
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(),
    )
    metric_acc = metrics.Accuracy()
    metric_f1 = metrics.F1()
    drift_detector = drift.ADWIN()
    drift_events = 0
    steps = 0

    for x, y in _iter_records(df):
        y_pred = model.predict_one(x)
        if y_pred is None:
            y_pred = 0
        metric_acc.update(y, y_pred)
        metric_f1.update(y, y_pred)

        error_signal = int(y_pred != y)
        drift_detector.update(error_signal)
        if drift_detector.drift_detected:
            drift_events += 1

        model.learn_one(x, y)
        steps += 1

    return AdaptiveRunSummary(
        steps=steps,
        accuracy=metric_acc.get(),
        f1=metric_f1.get(),
        drift_events=drift_events,
    )


def run_adaptive_experiment(
    frame: pd.DataFrame,
    drift_window: int = 50,
    drift_threshold: float = 0.2,
) -> AdaptiveRunSummary:
    """Compatibility wrapper for CLI calls with future drift params."""
    del drift_window, drift_threshold
    return run_adaptive_training(frame)


def run_adaptive_from_csv(input_csv: Path, output_metrics: Path | None = None) -> AdaptiveRunSummary:
    """Load processed dataset and execute adaptive training loop."""
    frame = pd.read_csv(input_csv)
    summary = run_adaptive_training(frame)
    if output_metrics is not None:
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = (
            "steps,accuracy,f1,drift_events\n"
            f"{summary.steps},{summary.accuracy:.4f},{summary.f1:.4f},{summary.drift_events}\n"
        )
        output_metrics.write_text(metrics_payload, encoding="utf-8")
    return summary
