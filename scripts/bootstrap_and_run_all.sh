#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"
DATASET_OUT="${DATASET_OUT:-data/raw/heart_uci_303.csv}"
PROCESSED_OUT="${PROCESSED_OUT:-data/processed/heart_uci_303_processed.csv}"
EVAL_DIR="${EVAL_DIR:-experiments/tracking}"
SKIP_FULL_RUN="${SKIP_FULL_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    --skip-full-run)
      SKIP_FULL_RUN=1
      ;;
  esac
done

# =========================
# Create venv
# =========================
ensure_venv() {
  if "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    return 0
  fi

  echo "venv failed, trying virtualenv..."
  "$PYTHON_BIN" -m pip install --user virtualenv
  "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
}

echo "==> [1/8] Creating virtual environment ($VENV_DIR)"
ensure_venv

# =========================
# Activate venv (FIXED)
# =========================
echo "==> Activating virtual environment"

if [ -f "$VENV_DIR/Scripts/activate" ]; then
  source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: Cannot find virtual environment activation script"
  exit 1
fi

# =========================
# Upgrade pip
# =========================
echo "==> [2/8] Upgrading pip"
python -m pip install --upgrade pip

# =========================
# Install dependencies
# =========================
echo "==> [3/8] Installing dependencies"
python -m pip install \
  numpy pandas scikit-learn river scipy joblib matplotlib \
  fastapi pydantic "uvicorn[standard]" streamlit pytest httpx ucimlrepo python-docx

# =========================
# Dataset fetch
# =========================
echo "==> [4/8] Fetching dataset"

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

print(f"Saved dataset -> {out} | shape={df.shape}")
PY

# =========================
# Tests (FIXED WINDOWS ISSUE)
# =========================
echo "==> [5/8] Running tests"

# FIX: normalize path issues for Windows
export PYTHONPATH="$(pwd)/src"

python -m pytest -q

# =========================
# Workflow
# =========================
if [[ "$SKIP_FULL_RUN" == "1" ]]; then
  echo "==> Skipping full workflow"
else
  echo "==> [6/8] Running full workflow"
  make run-all INPUT="$DATASET_OUT" DATA="$PROCESSED_OUT" EVAL_DIR="$EVAL_DIR"

  echo "==> [7/8] Running CV benchmark"
  make run-cv-benchmark DATA="$PROCESSED_OUT" EVAL_DIR="$EVAL_DIR"

  echo "==> [8/8] Running latency check"
  python scripts/run_latency_check.py --requests 50 --output "$EVAL_DIR/latency_report.json"
fi

# =========================
# Done
# =========================
echo ""
echo "✅ DONE - Outputs:"
echo "  - $EVAL_DIR/baseline_metrics.json"
echo "  - $EVAL_DIR/baseline_vs_adaptive.csv"
echo "  - $EVAL_DIR/adaptive_metrics.json"
echo "  - $EVAL_DIR/drift_scenarios_report.csv"
echo "  - $EVAL_DIR/cv_benchmark.csv"
echo "  - $EVAL_DIR/feature_importance.csv"
echo "  - $EVAL_DIR/local_explanations.csv"
echo "  - $EVAL_DIR/latency_report.json"