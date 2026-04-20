#!/usr/bin/env python3
"""Train baseline classifier on processed heart disease data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from heart_disease_rt.models import run_baseline_training


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    artifact = run_baseline_training(
        frame=frame,
        artifact_path=args.artifact,
        test_size=0.2,
        random_state=42,
    )
    metrics = artifact.metrics
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Baseline metrics saved to {args.metrics_output}")
    print(metrics)


if __name__ == "__main__":
    main()
