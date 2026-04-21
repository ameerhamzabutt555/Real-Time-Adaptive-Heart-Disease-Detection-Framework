#!/usr/bin/env bash
set -euo pipefail

# One-command local setup + full thesis workflow.
# Usage:
#   bash scripts/bootstrap_and_run_all.sh
# Optional env overrides:
#   PYTHON_BIN=python3.12 DATASET_OUT=data/raw/heart_uci_303.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATASET_OUT="${DATASET_OUT:-data/raw/heart_uci_303.csv}"
PROCESSED_OUT="${PROCESSED_OUT:-data/processed/heart_uci_303_processed.csv}"
EVAL_DIR="${EVAL_DIR:-experiments/tracking}"
SKIP_FULL_RUN="${SKIP_FULL_RUN:-0}"
ARTIFACT_OUT="${ARTIFACT_OUT:-models/artifacts/baseline.joblib}"
THRESHOLD="${THRESHOLD:-0.5}"

for arg in "$@"; do
  case "$arg" in
    --skip-full-run)
      SKIP_FULL_RUN=1
      ;;
  esac
done

ensure_venv() {
  if "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    return 0
  fi

  echo "Initial venv creation failed. Trying virtualenv fallback..."
  "$PYTHON_BIN" -m pip install --user virtualenv
  "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
}

echo "==> [1/8] Creating virtual environment ($VENV_DIR)"
ensure_venv
echo "==> Activating virtual environment"
if [ -f "$VENV_DIR/Scripts/activate" ]; then
  # Git Bash / Windows virtualenv path
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
  # Linux/macOS virtualenv path
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: Could not find activation script in $VENV_DIR"
  exit 1
fi

echo "==> [2/8] Upgrading pip"
python -m pip install --upgrade pip

echo "==> [3/8] Installing dependencies"
python -m pip install \
  numpy pandas scikit-learn river scipy joblib matplotlib \
  fastapi pydantic "uvicorn[standard]" streamlit pytest httpx ucimlrepo python-docx

echo "==> [4/8] Fetching official UCI Heart Disease (id=45)"
export DATASET_OUT
python - <<'PY'
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path
import os

out = Path(os.environ.get("DATASET_OUT", "data/raw/heart_uci_303.csv"))
ds = fetch_ucirepo(id=45)
X = ds.data.features.copy()
y = ds.data.targets.copy()
target_col = y.columns[0]

df = pd.concat([X, y[[target_col]].rename(columns={target_col: "target"})], axis=1)
df["target"] = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(int)

out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved dataset to {out} with shape={df.shape}")
PY

echo "==> [5/8] Running tests"
PYTHONPATH=src python -m pytest -q

run_full_workflow() {
  if command -v make >/dev/null 2>&1; then
    PYTHONPATH=src make run-all INPUT="$DATASET_OUT" DATA="$PROCESSED_OUT" EVAL_DIR="$EVAL_DIR"
    return
  fi

  echo "make not found; running direct Python fallback for full workflow"
  PYTHONPATH=src python scripts/run_data_pipeline.py --input "$DATASET_OUT" --output "$PROCESSED_OUT"
  PYTHONPATH=src python scripts/train_baseline.py --input "$PROCESSED_OUT" --artifact "$ARTIFACT_OUT" --metrics-output "$EVAL_DIR/baseline_metrics.json" --threshold "$THRESHOLD"
  PYTHONPATH=src python scripts/run_adaptive_loop.py --input "$PROCESSED_OUT" --output-summary "$EVAL_DIR/adaptive_metrics.json"
  PYTHONPATH=src python scripts/run_drift_scenarios.py --input "$PROCESSED_OUT" --output-summary "$EVAL_DIR/drift_scenarios_report.csv"
  PYTHONPATH=src python scripts/run_cv_benchmark.py --input "$PROCESSED_OUT" --output "$EVAL_DIR/cv_benchmark.csv"
  PYTHONPATH=src python scripts/run_evaluation.py --input "$DATASET_OUT" --output-dir "$EVAL_DIR"
}

run_cv_benchmark() {
  if command -v make >/dev/null 2>&1; then
    PYTHONPATH=src make run-cv-benchmark DATA="$PROCESSED_OUT" EVAL_DIR="$EVAL_DIR"
    return
  fi

  echo "make not found; running direct Python fallback for CV benchmark"
  PYTHONPATH=src python scripts/run_cv_benchmark.py --input "$PROCESSED_OUT" --output "$EVAL_DIR/cv_benchmark.csv"
}

if [[ "$SKIP_FULL_RUN" == "1" ]]; then
  echo "==> [6/8] Skipping full workflow (--skip-full-run enabled)"
else
  echo "==> [6/8] Running full workflow"
  run_full_workflow
fi

if [[ "$SKIP_FULL_RUN" == "1" ]]; then
  echo "==> [7/8] Skipping repeated CV benchmark (--skip-full-run enabled)"
else
  echo "==> [7/8] Running repeated CV benchmark"
  run_cv_benchmark
fi

if [[ "$SKIP_FULL_RUN" == "1" ]]; then
  echo "==> [8/8] Skipping latency check (--skip-full-run enabled)"
else
  echo "==> [8/8] Running latency check"
  PYTHONPATH=src:. python scripts/run_latency_check.py --requests 50 --output "$EVAL_DIR/latency_report.json"
fi

echo ""
echo "Done. Key outputs:"
echo "  - $EVAL_DIR/baseline_metrics.json"
echo "  - $EVAL_DIR/baseline_vs_adaptive.csv"
echo "  - $EVAL_DIR/adaptive_metrics.json"
echo "  - $EVAL_DIR/drift_scenarios_report.csv"
echo "  - $EVAL_DIR/cv_benchmark.csv"
echo "  - $EVAL_DIR/feature_importance.csv"
echo "  - $EVAL_DIR/local_explanations.csv"
echo "  - $EVAL_DIR/latency_report.json"
