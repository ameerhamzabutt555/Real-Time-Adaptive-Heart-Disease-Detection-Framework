#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
APP_MODULE="${APP_MODULE:-api.main:app}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$VENV_DIR" ]] || die "Virtualenv not found at '$VENV_DIR'. Run: bash scripts/bootstrap_and_run_all.sh --skip-full-run"

# Activate venv
VENV_PY=""
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # macOS/Linux/WSL
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  VENV_PY="$VENV_DIR/bin/python"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Windows (Git Bash)
  # shellcheck disable=SC1090
  source "$VENV_DIR/Scripts/activate"
  VENV_PY="$VENV_DIR/Scripts/python"
else
  die "Could not find venv activation script in '$VENV_DIR'"
fi

[[ -x "$VENV_PY" ]] || die "Could not resolve venv Python at '$VENV_PY'"

# Require Python 3.10+
"$VENV_PY" - <<'PY' >/dev/null 2>&1 || die "Python >= 3.10 required. Recreate venv with brew python and rerun bootstrap script."
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"

cleanup() {
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${DASH_PID:-}" ]] && kill "$DASH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting API: http://localhost:${API_PORT} (docs: /docs)"
uvicorn "$APP_MODULE" --reload --host "$HOST" --port "$API_PORT" &
API_PID=$!

echo "==> Starting Dashboard: http://localhost:${DASHBOARD_PORT}"
streamlit run dashboard/app.py --server.port "$DASHBOARD_PORT" &
DASH_PID=$!

echo ""
echo "Running..."
echo "  - API PID: $API_PID"
echo "  - Dashboard PID: $DASH_PID"
echo "Press Ctrl+C to stop both."

wait

