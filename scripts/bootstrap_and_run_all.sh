#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv}"
DATASET_OUT="${DATASET_OUT:-data/raw/heart_uci_303.csv}"
PROCESSED_OUT="${PROCESSED_OUT:-data/processed/heart_uci_303_processed.csv}"
EVAL_DIR="${EVAL_DIR:-experiments/tracking}"
SKIP_FULL_RUN="${SKIP_FULL_RUN:-0}"
ARTIFACT_OUT="${ARTIFACT_OUT:-models/artifacts/baseline.joblib}"
THRESHOLD="${THRESHOLD:-0.5}"

resolve_python() {
  # Project requires Python >= 3.10 (PEP604 unions, etc).
  # On macOS, `/usr/bin/python3` may be 3.9; prefer Homebrew/pyenv Python if present.

  python_is_ok() {
    local bin="$1"
    "$bin" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
  }

  choose_python() {
    local candidate=""
    for candidate in "$@"; do
      [[ -z "$candidate" ]] && continue
      if [[ "$candidate" == /* ]]; then
        [[ -x "$candidate" ]] || continue
      else
        command -v "$candidate" >/dev/null 2>&1 || continue
        candidate="$(command -v "$candidate")"
      fi

      if python_is_ok "$candidate"; then
        PYTHON_BIN="$candidate"
        return 0
      fi
    done
    return 1
  }

  if [[ -n "${PYTHON_BIN}" ]]; then
    if choose_python "${PYTHON_BIN}"; then
      return 0
    fi
    echo "ERROR: PYTHON_BIN='${PYTHON_BIN}' was not found or is < Python 3.10"
    exit 1
  fi

  if choose_python \
    /opt/homebrew/bin/python3 \
    /usr/local/bin/python3 \
    python3 \
    python; then
    return 0
  fi

  echo "ERROR: Python 3.10+ not found."
  echo "Install it (recommended on macOS):"
  echo "  brew install python"
  echo "Then re-run this script, or set PYTHON_BIN explicitly, e.g.:"
  echo "  PYTHON_BIN=/opt/homebrew/bin/python3 ./bootstrap_and_run_all.sh"
  exit 1
}

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
  resolve_python

  # If a venv already exists but is using an older Python, rebuild it.
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      echo "==> Existing venv uses Python < 3.10. Recreating $VENV_DIR ..."
      rm -rf "$VENV_DIR"
    fi
  fi

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
# Activate venv (FIXED ONLY ONCE)
# =========================
echo "==> Activating virtual environment"

VENV_PYTHON=""
if [ -f "$VENV_DIR/Scripts/activate" ]; then
  # Windows (Git Bash)
  source "$VENV_DIR/Scripts/activate"
  VENV_PYTHON="$VENV_DIR/Scripts/python"
elif [ -f "$VENV_DIR/bin/activate" ]; then
  # Linux / Mac / WSL
  source "$VENV_DIR/bin/activate"
  VENV_PYTHON="$VENV_DIR/bin/python"
else
  echo "ERROR: Could not find virtual environment activation script"
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  # Fallback to whatever is on PATH after activation
  VENV_PYTHON="$(command -v python3 || true)"
  if [[ -z "$VENV_PYTHON" ]]; then
    VENV_PYTHON="$(command -v python || true)"
  fi
fi

if [[ -z "$VENV_PYTHON" ]]; then
  echo "ERROR: Could not resolve Python interpreter inside virtualenv"
  exit 1
fi

# =========================
# Upgrade pip
# =========================
echo "==> [2/8] Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip

# =========================
# Install dependencies
# =========================
echo "==> [3/8] Installing dependencies"
"$VENV_PYTHON" -m pip install \
  numpy pandas scikit-learn river scipy joblib matplotlib \
  fastapi pydantic "uvicorn[standard]" streamlit pytest httpx ucimlrepo python-docx

# =========================
# Fetch dataset
# =========================
echo "==> [4/8] Fetching dataset"

export DATASET_OUT

"$VENV_PYTHON" - <<'PY'
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path
import os

out = Path(os.environ.get("DATASET_OUT"))

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
# Tests
# =========================
echo "==> [5/8] Running tests"

export PYTHONPATH="$(pwd)/src"
"$VENV_PYTHON" -m pytest -q

# =========================
# Full workflow
# =========================
run_full_workflow() {
  echo "==> Running full ML pipeline"

  PYTHONPATH=src "$VENV_PYTHON" scripts/run_data_pipeline.py \
    --input "$DATASET_OUT" \
    --output "$PROCESSED_OUT"

  PYTHONPATH=src "$VENV_PYTHON" scripts/train_baseline.py \
    --input "$PROCESSED_OUT" \
    --artifact "$ARTIFACT_OUT" \
    --metrics-output "$EVAL_DIR/baseline_metrics.json" \
    --threshold "$THRESHOLD"

  PYTHONPATH=src "$VENV_PYTHON" scripts/run_adaptive_loop.py \
    --input "$PROCESSED_OUT" \
    --output-summary "$EVAL_DIR/adaptive_metrics.json"

  PYTHONPATH=src "$VENV_PYTHON" scripts/run_drift_scenarios.py \
    --input "$PROCESSED_OUT" \
    --output-summary "$EVAL_DIR/drift_scenarios_report.csv"

  PYTHONPATH=src "$VENV_PYTHON" scripts/run_cv_benchmark.py \
    --input "$PROCESSED_OUT" \
    --output "$EVAL_DIR/cv_benchmark.csv"

  PYTHONPATH=src "$VENV_PYTHON" scripts/run_evaluation.py \
    --input "$DATASET_OUT" \
    --output-dir "$EVAL_DIR"
}

run_cv_benchmark() {
  echo "==> Running CV benchmark"

  PYTHONPATH=src "$VENV_PYTHON" scripts/run_cv_benchmark.py \
    --input "$PROCESSED_OUT" \
    --output "$EVAL_DIR/cv_benchmark.csv"
}

# =========================
# Execution
# =========================
if [[ "$SKIP_FULL_RUN" == "1" ]]; then
  echo "==> Skipping full workflow"
else
  echo "==> [6/8] Running full workflow"
  run_full_workflow
fi

if [[ "$SKIP_FULL_RUN" == "1" ]]; then
  echo "==> Skipping CV benchmark"
else
  echo "==> [7/8] Running CV benchmark"
  run_cv_benchmark
fi

echo "==> [8/8] Running latency check"
"$VENV_PYTHON" scripts/run_latency_check.py \
  --requests 50 \
  --output "$EVAL_DIR/latency_report.json"

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