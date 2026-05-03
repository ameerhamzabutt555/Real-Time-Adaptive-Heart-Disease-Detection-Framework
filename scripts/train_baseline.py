#!/usr/bin/env python3
"""Train baseline classifier on processed heart disease data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from heart_disease_rt.models import run_baseline_training
from heart_disease_rt.serving.config import save_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline static model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/heart_disease_processed.csv"),
        help="Path to processed CSV.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("models/artifacts/baseline.joblib"),
        help="Output path for serialized model artifact.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("experiments/tracking/baseline_metrics.json"),
        help="Output path for baseline metrics JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive-class decision.",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "hgb"],
        default="logreg",
        help="Baseline model family to train (logreg=logistic regression, hgb=hist gradient boosting).",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune decision threshold on a validation split (leakage-safe).",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("configs/model_config.json"),
        help="Output path for serving model config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    artifact = run_baseline_training(
        frame=frame,
        artifact_path=args.artifact,
        test_size=0.2,
        random_state=42,
        model_type=args.model,
        tune_threshold=bool(args.tune_threshold),
    )
    metrics = artifact.metrics
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model_config(
        output_path=args.config_output,
        artifact_path=args.artifact,
        probability_threshold=float(metrics.get("selected_threshold", args.threshold)),
        model_version=f"baseline-{args.model}",
    )
    print(f"Baseline metrics saved to {args.metrics_output}")
    print(f"Serving config saved to {args.config_output}")
    print(metrics)


if __name__ == "__main__":
    main()
