"""Adaptive/online training utilities using River."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from river import compose, drift, linear_model, metrics, preprocessing

from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN
from heart_disease_rt.monitoring.reporting import save_adaptive_report_plots, summarize_comparison

SUPPORTED_DETECTORS = ("adwin", "ddm", "page_hinkley")


def _iter_records(df: pd.DataFrame) -> Iterable[tuple[dict[str, float], int]]:
    for _, row in df.iterrows():
        x = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
        y = int(row[LABEL_COLUMN])
        yield x, y


def _build_detector(detector_name: str):
    normalized = detector_name.strip().lower()
    if normalized == "adwin":
        return drift.ADWIN()
    if normalized == "ddm":
        return drift.binary.DDM()
    if normalized in {"page_hinkley", "pagehinkley", "ph"}:
        return drift.PageHinkley()
    raise ValueError(f"Unsupported detector '{detector_name}'. Use adwin, ddm, or page_hinkley.")


def run_adaptive_training(
    df: pd.DataFrame,
    detector_name: str = "adwin",
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run prequential online training with selected drift detector."""
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(),
    )
    metric_acc = metrics.Accuracy()
    metric_f1 = metrics.F1()
    drift_detector = _build_detector(detector_name)
    drift_events = 0
    progress_rows: list[dict[str, float | int | str]] = []

    for step, (x, y) in enumerate(_iter_records(df), start=1):
        y_pred = model.predict_one(x)
        if y_pred is None:
            y_pred = 0

        metric_acc.update(y, y_pred)
        metric_f1.update(y, y_pred)

        error_signal = int(y_pred != y)
        drift_detector.update(error_signal)
        drift_flag = int(drift_detector.drift_detected)
        if drift_flag:
            drift_events += 1

        progress_rows.append(
            {
                "detector": detector_name,
                "step": step,
                "y_true": y,
                "y_pred": int(y_pred),
                "metric_accuracy": float(metric_acc.get()),
                "metric_f1": float(metric_f1.get()),
                "drift_flag": drift_flag,
            }
        )
        model.learn_one(x, y)

    summary: dict[str, float | int | str] = {
        "detector": detector_name,
        "steps": len(progress_rows),
        "accuracy": float(metric_acc.get()),
        "f1": float(metric_f1.get()),
        "drift_events": drift_events,
    }
    return summary, pd.DataFrame(progress_rows)


def run_adaptive_experiment(
    frame: pd.DataFrame,
    detector_name: str = "adwin",
) -> dict[str, float | int | str]:
    """Backwards-compatible helper returning only headline summary."""
    summary, _ = run_adaptive_training(frame, detector_name=detector_name)
    return summary


def compare_drift_detectors(
    frame: pd.DataFrame,
    detectors: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility helper returning summary and combined progress DataFrame."""
    selected = list(detectors) if detectors is not None else list(SUPPORTED_DETECTORS)
    summaries: list[dict[str, float | int | str]] = []
    all_progress: list[pd.DataFrame] = []

    for detector_name in selected:
        summary, progress = run_adaptive_training(frame, detector_name=detector_name)
        summaries.append(
            {
                "detector": summary["detector"],
                "steps": summary["steps"],
                "accuracy": summary["accuracy"],
                "f1": summary["f1"],
                "drift_events": summary["drift_events"],
            }
        )
        all_progress.append(progress)

    summary_df = pd.DataFrame(summaries)
    progress_df = pd.concat(all_progress, ignore_index=True) if all_progress else pd.DataFrame()
    return summary_df, progress_df


def run_adaptive_comparison(
    frame: pd.DataFrame,
    detector_names: list[str] | None = None,
    output_progress_csv: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, object]:
    """Run detector comparison and optionally persist progress/plot artifacts."""
    selected = detector_names or list(SUPPORTED_DETECTORS)
    summaries: list[dict[str, float | int | str]] = []
    all_progress: list[pd.DataFrame] = []

    for detector_name in selected:
        summary, progress = run_adaptive_training(frame, detector_name=detector_name)
        summaries.append(
            {
                "detector": summary["detector"],
                "steps": summary["steps"],
                "accuracy": summary["accuracy"],
                "f1": summary["f1"],
                "drift_events": summary["drift_events"],
            }
        )
        all_progress.append(progress)

    summary_df = pd.DataFrame(summaries)
    progress_df = pd.concat(all_progress, ignore_index=True) if all_progress else pd.DataFrame()
    summary_payload = summarize_comparison(progress_df)

    if output_progress_csv is not None:
        output_progress_csv.parent.mkdir(parents=True, exist_ok=True)
        progress_df.to_csv(output_progress_csv, index=False)
        summary_payload["progress_csv"] = str(output_progress_csv)

    if report_dir is not None:
        summary_payload["plots"] = save_adaptive_report_plots(progress_df, summary_df, report_dir)

    return {
        "summary": summary_payload,
        "summary_df": summary_df,
        "progress_df": progress_df,
    }


def run_adaptive_from_csv(
    input_csv: Path,
    output_metrics: Path | None = None,
    detector_name: str = "adwin",
) -> dict[str, float | int | str]:
    """Load processed dataset and execute adaptive training loop."""
    frame = pd.read_csv(input_csv)
    summary, _ = run_adaptive_training(frame, detector_name=detector_name)
    if output_metrics is not None:
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = (
            "detector,steps,accuracy,f1,drift_events\n"
            f"{summary['detector']},{summary['steps']},{summary['accuracy']:.4f},{summary['f1']:.4f},{summary['drift_events']}\n"
        )
        output_metrics.write_text(metrics_payload, encoding="utf-8")
    return summary
