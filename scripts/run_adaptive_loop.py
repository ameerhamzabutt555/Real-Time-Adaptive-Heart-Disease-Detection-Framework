#!/usr/bin/env python3
"""Run online adaptive learning loop and save metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from heart_disease_rt.data import preprocess_dataframe
from heart_disease_rt.models import run_adaptive_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online adaptive learning experiment.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV dataset.")
    parser.add_argument(
        "--model",
        choices=["logreg", "arf"],
        default="logreg",
        help="Online model type (logreg or arf=Adaptive Random Forest).",
    )
    parser.add_argument(
        "--detectors",
        type=str,
        default="adwin,ddm,page_hinkley",
        help="Comma-separated detector names (adwin, ddm, page_hinkley).",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("experiments/tracking/adaptive_metrics.json"),
        help="Path to detector-comparison summary JSON output.",
    )
    parser.add_argument(
        "--output-progress",
        type=Path,
        default=Path("experiments/tracking/adaptive_progress.csv"),
        help="Path to per-step progress CSV output.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("experiments/runs/adaptive/plots"),
        help="Directory path for generated plot PNG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    processed = preprocess_dataframe(raw_df)

    detectors = [item.strip().lower() for item in args.detectors.split(",") if item.strip()]
    result = run_adaptive_comparison(
        processed,
        detector_names=detectors,
        model_type=args.model,
    )
    summary_df = result["summary_df"]
    progress_df = result["progress_df"]

    args.output_progress.parent.mkdir(parents=True, exist_ok=True)
    progress_df.to_csv(args.output_progress, index=False)

    args.output_plot.mkdir(parents=True, exist_ok=True)
    from heart_disease_rt.monitoring import save_adaptive_report_plots

    plot_paths = save_adaptive_report_plots(progress_df, summary_df, args.output_plot)

    summary_df = summary_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    detector_summary: dict[str, dict[str, float | int]] = {}
    progress_files: dict[str, str] = {}
    for _, row in summary_df.iterrows():
        detector_summary[str(row["detector"])] = {
            "accuracy": float(row["accuracy"]),
            "f1": float(row["f1"]),
            "drift_events": int(row["drift_events"]),
            "steps": int(row["steps"]),
        }
        detector = str(row["detector"])
        per_detector = progress_df[progress_df["detector"] == detector].copy()
        per_detector_path = args.output_progress.parent / f"adaptive_progress_{detector}.csv"
        per_detector.to_csv(per_detector_path, index=False)
        progress_files[detector] = str(per_detector_path)

    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    summary = {
        "summary": result["summary"],
        "best_detector": str(best_row.get("detector", "")),
        "accuracy": float(best_row.get("accuracy", 0.0)),
        "f1": float(best_row.get("f1", 0.0)),
        "drift_events": int(best_row.get("drift_events", 0)),
        "steps": int(best_row.get("steps", 0)),
        "detector_summary": detector_summary,
        "progress_csv": str(args.output_progress),
        "progress_files": progress_files,
        "plots": plot_paths,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved adaptive summary to {args.output_summary}")
    print(f"Saved adaptive progress to {args.output_progress}")
    print(f"Saved adaptive plots to {args.output_plot}")


if __name__ == "__main__":
    main()
