"""Reporting utilities for adaptive experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_comparison(progress_df: pd.DataFrame) -> dict[str, object]:
    """Build dashboard-friendly summary grouped by detector."""
    detectors: dict[str, dict[str, float | int]] = {}
    if progress_df.empty:
        return {"best_detector": "", "detectors": detectors}

    for detector_name, group in progress_df.groupby("detector"):
        f1_value = float(group["metric_f1"].iloc[-1]) if "metric_f1" in group.columns else 0.0
        detectors[str(detector_name)] = {
            "accuracy": float(group["metric_accuracy"].iloc[-1]),
            "f1": f1_value,
            "drift_events": int(group["drift_flag"].sum()),
            "steps": int(group["step"].max()),
        }

    best_detector = max(detectors.items(), key=lambda item: item[1]["accuracy"])[0]
    return {"best_detector": best_detector, "detectors": detectors}


def _save_accuracy_plot(progress: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for detector_name, group in progress.groupby("detector"):
        ax.plot(group["step"], group["metric_accuracy"], label=f"{detector_name} accuracy")
        drift_points = group[group["drift_flag"] == 1]
        if not drift_points.empty:
            ax.scatter(drift_points["step"], drift_points["metric_accuracy"], s=18)

    ax.set_title("Adaptive Learning Progress")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_detector_comparison_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    sorted_df = summary_df.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sorted_df["detector"], sorted_df["accuracy"], color="#1f77b4")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Drift Detector Comparison")
    for bar, value in zip(bars, sorted_df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_adaptive_report_plots(
    progress_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Generate standard PNG artifacts for dashboard and thesis reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot_path = output_dir / "adaptive_accuracy_progress.png"
    detector_plot_path = output_dir / "adaptive_detector_comparison.png"

    _save_accuracy_plot(progress_df, accuracy_plot_path)
    _save_detector_comparison_plot(summary_df, detector_plot_path)

    return {
        "accuracy_progress_plot": str(accuracy_plot_path),
        "detector_comparison_plot": str(detector_plot_path),
    }


def build_confusion_summary(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, int | float]:
    """Return confusion matrix components with FN-focused metrics."""
    true = y_true.astype(int)
    pred = y_pred.astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    total = max(1, len(true))
    fn_rate = fn / max(1, int((true == 1).sum()))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_negative_rate": float(fn_rate),
        "error_rate": float((fp + fn) / total),
    }
