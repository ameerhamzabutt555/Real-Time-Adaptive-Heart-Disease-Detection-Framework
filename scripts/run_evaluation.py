#!/usr/bin/env python3
"""Run full baseline/adaptive evaluation and export thesis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from heart_disease_rt.data import LABEL_COLUMN, FEATURE_COLUMNS, preprocess_dataframe
from heart_disease_rt.explainability.importance import compute_feature_importance, compute_local_explanations
from heart_disease_rt.models import run_adaptive_comparison, run_baseline_training
from heart_disease_rt.monitoring import build_confusion_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete thesis evaluation workflow.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/tracking"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold used in interpretation text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    frame = preprocess_dataframe(raw_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_artifact = run_baseline_training(frame, artifact_path=Path("models/artifacts/baseline.joblib"))
    baseline_model = baseline_artifact.model
    y_true = pd.Series(baseline_artifact.y_true or [], name="y_true")
    y_pred = pd.Series(baseline_artifact.y_pred or [], name="y_pred")
    confusion = build_confusion_summary(y_true=y_true, y_pred=y_pred)

    adaptive_result = run_adaptive_comparison(
        frame,
        report_dir=Path("experiments/runs/adaptive/plots"),
    )
    adaptive_progress = adaptive_result["progress_df"]
    adaptive_summary = adaptive_result["summary"]
    adaptive_summary_df = adaptive_result["summary_df"]
    best_detector = str(adaptive_summary.get("best_detector", "adwin"))
    adaptive_best = adaptive_progress[adaptive_progress["detector"] == best_detector].copy()
    adaptive_final_acc = float(adaptive_best["metric_accuracy"].iloc[-1]) if not adaptive_best.empty else 0.0
    adaptive_final_f1 = float(adaptive_best["metric_f1"].iloc[-1]) if not adaptive_best.empty else 0.0

    comparison_table = pd.DataFrame(
        [
            {
                "model_type": "baseline_static",
                "accuracy": baseline_artifact.metrics["accuracy"],
                "precision": baseline_artifact.metrics["precision"],
                "recall": baseline_artifact.metrics["recall"],
                "f1": baseline_artifact.metrics["f1"],
                "roc_auc": baseline_artifact.metrics["roc_auc"],
                "drift_events": 0,
            },
            {
                "model_type": f"adaptive_{best_detector}",
                "accuracy": adaptive_final_acc,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": adaptive_final_f1,
                "roc_auc": float("nan"),
                "drift_events": int((adaptive_best["drift_flag"] == 1).sum()),
            },
        ]
    )
    comparison_table.to_csv(args.output_dir / "baseline_vs_adaptive.csv", index=False)

    fn_report = {
        "false_negatives": int(confusion["fn"]),
        "true_positives": int(confusion["tp"]),
        "false_negative_rate": float(confusion["false_negative_rate"]),
        "notes": "Lower false-negative rate is clinically preferred to reduce missed high-risk patients.",
    }
    (args.output_dir / "false_negative_analysis.json").write_text(
        json.dumps(fn_report, indent=2),
        encoding="utf-8",
    )

    baseline_acc = float(baseline_artifact.metrics["accuracy"])
    adaptive_acc = float(adaptive_final_acc)
    delta = adaptive_acc - baseline_acc
    stat_report = {
        "metric": "accuracy_difference(adaptive-baseline)",
        "mean_difference": float(delta),
        "ci_95_low": float(delta),
        "ci_95_high": float(delta),
        "alpha": args.alpha,
        "interpretation": "Point-estimate difference reported. Use repeated runs for robust significance claims.",
    }
    (args.output_dir / "statistical_comparison.json").write_text(
        json.dumps(stat_report, indent=2),
        encoding="utf-8",
    )

    x_eval = frame[FEATURE_COLUMNS].copy()
    y_eval = frame[LABEL_COLUMN].copy()
    try:
        importance_df = compute_feature_importance(baseline_model, x_eval, y_true=y_eval)
        importance_df.to_csv(args.output_dir / "feature_importance.csv", index=False)
    except Exception as exc:
        (args.output_dir / "feature_importance.csv").write_text(
            "error\n" + str(exc) + "\n",
            encoding="utf-8",
        )

    try:
        local_df = compute_local_explanations(baseline_model, x_eval, top_k=3)
        local_df.to_csv(args.output_dir / "local_explanations.csv", index=False)
    except Exception as exc:
        (args.output_dir / "local_explanations.csv").write_text(
            "error\n" + str(exc) + "\n",
            encoding="utf-8",
        )

    if not adaptive_summary_df.empty:
        adaptive_summary_df.to_csv(args.output_dir / "adaptive_summary.csv", index=False)
        adaptive_summary_df.to_csv(args.output_dir / "detector_comparison.csv", index=False)

    print(f"Saved evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
