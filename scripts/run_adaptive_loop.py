#!/usr/bin/env python3
"""Run online adaptive learning loop and save metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from heart_disease_rt.data import preprocess_dataframe
from heart_disease_rt.models import run_adaptive_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online adaptive learning experiment.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tracking/adaptive_metrics.json"),
        help="Path to adaptive metrics JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    processed = preprocess_dataframe(raw_df)

    result = run_adaptive_training(processed)
    metrics = {
        "steps": result.steps,
        "accuracy": result.accuracy,
        "f1": result.f1,
        "drift_events": result.drift_events,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved adaptive experiment metrics to {args.output}")


if __name__ == "__main__":
    main()
