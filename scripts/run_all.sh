#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-data/raw/heart.csv}"
PROCESSED_PATH="${2:-data/processed/heart_disease_processed.csv}"

echo "[1/6] Running data preprocessing pipeline..."
python3 scripts/run_data_pipeline.py --input "$INPUT_PATH" --output "$PROCESSED_PATH"

echo "[2/6] Training baseline model..."
python3 scripts/train_baseline.py \
  --input "$PROCESSED_PATH" \
  --artifact "models/artifacts/baseline.joblib" \
  --metrics-output "experiments/tracking/baseline_metrics.json" \
  --threshold 0.5

echo "[3/6] Running adaptive detector comparison..."
python3 scripts/run_adaptive_loop.py \
  --input "$PROCESSED_PATH" \
  --output-summary "experiments/tracking/adaptive_metrics.json"

echo "[4/6] Running drift scenario benchmark..."
python3 scripts/run_drift_scenarios.py \
  --input "$PROCESSED_PATH" \
  --output-summary "experiments/tracking/drift_scenarios_report.csv"

echo "[5/6] Running full evaluation exports..."
python3 scripts/run_evaluation.py \
  --input "$INPUT_PATH" \
  --output-dir "experiments/tracking"

echo "[6/6] Running tests..."
python3 -m pytest -q

echo "Completed full thesis workflow successfully."
