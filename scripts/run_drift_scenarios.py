#!/usr/bin/env python3
"""Simulate drift scenarios and evaluate detector response metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from heart_disease_rt.data import preprocess_dataframe
from heart_disease_rt.monitoring.drift_scenarios import evaluate_drift_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift scenario evaluation.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV.")
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("experiments/tracking/drift_scenarios_report.csv"),
        help="CSV output for scenario-level metrics.",
    )
    parser.add_argument(
        "--output-events",
        type=Path,
        default=Path("experiments/tracking/drift_event_log.csv"),
        help="CSV output for per-step detector event logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    processed = preprocess_dataframe(raw)
    summary_df, events_df = evaluate_drift_scenarios(processed)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary, index=False)
    events_df.to_csv(args.output_events, index=False)

    print(f"Saved scenario summary to {args.output_summary}")
    print(f"Saved detector event logs to {args.output_events}")


if __name__ == "__main__":
    main()
