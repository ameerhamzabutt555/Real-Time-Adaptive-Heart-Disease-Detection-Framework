# Real-Time Adaptive Heart Disease Detection Framework

This repository follows a research-to-production layout so the same codebase can support:
- thesis experimentation,
- adaptive/streaming model development,
- real-time API serving,
- and dashboard/demo delivery.

## Recommended Folder Structure

```text
.
├── api/                      # FastAPI entrypoint and HTTP layer
├── configs/                  # YAML/TOML runtime configurations
├── dashboard/                # Streamlit app for real-time demo
├── data/
│   ├── external/             # Third-party datasets (read-only)
│   ├── interim/              # Temporary transformed data
│   ├── processed/            # Final training-ready datasets
│   └── raw/                  # Original unmodified data
├── docs/                     # Thesis notes, architecture, experiments docs
├── experiments/
│   ├── configs/              # Experiment-specific config files
│   ├── runs/                 # Local experiment outputs (gitignored)
│   └── tracking/             # Local tracking metadata (gitignored)
├── models/
│   └── artifacts/            # Saved model files (gitignored)
├── notebooks/                # Exploration notebooks (non-production)
├── scripts/                  # Run helpers for API/dashboard/pipeline
├── src/
│   └── heart_disease_rt/
│       ├── data/             # Ingestion, preprocessing, stream simulation
│       ├── explainability/   # SHAP / explanation utilities
│       ├── features/         # Feature engineering logic
│       ├── models/           # Baseline + online/adaptive modeling code
│       ├── monitoring/       # Drift/performance monitoring helpers
│       ├── serving/          # Inference service layer (schemas, predictor)
│       └── utils/            # Shared utilities (logging, helpers)
├── tests/
│   ├── integration/          # End-to-end/service behavior tests
│   └── unit/                 # Fast isolated tests
├── .env.example
├── Makefile
└── pyproject.toml
```

## Best-Practice Rules

1. Keep **all reusable code** in `src/heart_disease_rt`, not notebooks.
2. Keep `data/raw` immutable; write transformed outputs to `data/interim` or `data/processed`.
3. Track experiment settings in `experiments/configs` so runs are reproducible.
4. Keep large files (models/runs/datasets) out of git; commit only metadata/configs.
5. Keep API (`api/`) thin; business/model logic should stay in `src/`.
6. Add tests for every core module in `tests/unit` and pipeline/API checks in `tests/integration`.

## Quick Start (after environment setup)

```bash
make help
make run-api
make run-dashboard
make run-pipeline INPUT=data/raw/heart.csv DATA=data/processed/heart_processed.csv
make train-baseline DATA=data/processed/heart_processed.csv
make run-adaptive DATA=data/processed/heart_processed.csv
```

## Complete Local Setup and Testing (Fresh Clone)

```bash
# 1) Clone and enter project
git clone https://github.com/ameerhamzabutt555/Real-Time-Adaptive-Heart-Disease-Detection-Framework.git
cd Real-Time-Adaptive-Heart-Disease-Detection-Framework

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn river joblib matplotlib fastapi pydantic "uvicorn[standard]" streamlit pytest httpx

# 4) Run full tests
python -m pytest -q
```

Detailed documentation is available in:
- `docs/PROJECT_SETUP_AND_TECHNICAL_EXPLANATION.doc`
- `docs/PROJECT_SETUP_AND_TECHNICAL_EXPLANATION.docx` (Word file)

## Adaptive Detector Comparison + Reports

Run adaptive experiment across multiple drift detectors and auto-generate:
- summary JSON,
- per-detector progress CSVs,
- comparison plot PNGs.

```bash
make run-adaptive DATA=data/processed/heart_processed.csv
```

Outputs are generated under:
- `experiments/tracking/adaptive_metrics.json`
- `experiments/tracking/adaptive_progress.csv`
- `experiments/tracking/adaptive_progress_<detector>.csv`
- `experiments/runs/adaptive/plots/*.png`

## Adaptive API (Online Learning Mode)

In addition to static baseline endpoints (`/predict`), the API now exposes adaptive online endpoints:

- `GET /adaptive/status` - returns online learning state (samples seen, drift events, online metrics).
- `POST /adaptive/predict` - predicts with in-memory adaptive model.
- `POST /adaptive/learn` - performs one online learning step with labeled feedback.

Example learn payload:

```json
{
  "patient": {
    "age": 53,
    "sex": 1,
    "resting_bp": 132.0,
    "cholesterol": 246.0,
    "fasting_blood_sugar": 0,
    "restecg": 1,
    "max_heart_rate": 151.0,
    "exercise_angina": 1,
    "chest_pain_type": 2,
    "oldpeak": 1.3,
    "slope": 1,
    "ca": 0,
    "thal": 2
  },
  "label": 1
}
```

## Thesis Completion Commands

Use these targets to generate all thesis evidence artifacts:

```bash
make run-drift-scenarios DATA=data/processed/heart_processed.csv
make run-evaluation INPUT=data/raw/heart.csv EVAL_DIR=experiments/tracking
make run-cv-benchmark INPUT=data/raw/heart.csv
make run-all INPUT=data/raw/heart.csv
```

Main outputs:
- `experiments/tracking/baseline_vs_adaptive.csv`
- `experiments/tracking/drift_scenarios_report.csv`
- `experiments/tracking/false_negative_analysis.json`
- `experiments/tracking/statistical_comparison.json`
- `experiments/tracking/feature_importance.csv`
- `experiments/tracking/local_explanations.csv`

## One-Command Local Bootstrap (Setup + Test + Full Run)

Run this single script from repo root:

```bash
bash scripts/bootstrap_and_run_all.sh
```

Optional custom paths:

```bash
bash scripts/bootstrap_and_run_all.sh data/raw/heart_uci_303.csv data/processed/heart_uci_303_processed.csv experiments/tracking
```

This script performs:
- virtual environment creation (`.venv`) if missing,
- dependency installation,
- official UCI dataset fetch (303 rows),
- full test suite (`pytest`),
- full thesis workflow (`make run-all`),
- repeated CV benchmark (`make run-cv-benchmark`).
- `experiments/tracking/cv_benchmark_summary.csv`
