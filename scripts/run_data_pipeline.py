#!/usr/bin/env python3
"""Run preprocessing and persist a cleaned dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from heart_disease_rt.data import preprocess_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heart disease preprocessing pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/heart_disease_processed.csv"),
        help="Path for processed CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    processed = preprocess_dataframe(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)
    print(f"Saved processed dataset to {args.output}")


if __name__ == "__main__":
    main()
