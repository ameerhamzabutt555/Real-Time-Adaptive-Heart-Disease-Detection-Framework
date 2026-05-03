#!/usr/bin/env python3
"""Run repeated stratified CV benchmark for literature comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from heart_disease_rt.data import preprocess_dataframe
from heart_disease_rt.models import run_repeated_cv_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated CV benchmark.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/tracking/repeated_cv_benchmark.csv"),
        help="Output CSV path for repeated CV summary.",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of CV splits.")
    parser.add_argument("--repeats", type=int, default=20, help="Number of CV repeats.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    frame = preprocess_dataframe(raw)
    fold_df, summary_df = run_repeated_cv_benchmark(
        frame,
        n_splits=args.splits,
        n_repeats=args.repeats,
        random_state=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    folds_output = args.output.with_name(args.output.stem + "_folds.csv")
    fold_df.to_csv(folds_output, index=False)
    print(f"Saved repeated CV benchmark summary to {args.output}")
    print(f"Saved repeated CV fold metrics to {folds_output}")


if __name__ == "__main__":
    main()
