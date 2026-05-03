PYTHON ?= python3
APP_MODULE ?= api.main:app
INPUT ?= data/raw/heart_uci_303.csv
DATA ?= data/processed/heart_uci_303_processed.csv
ARTIFACT ?= models/artifacts/baseline.joblib
BASELINE_METRICS ?= experiments/tracking/baseline_metrics.json
ADAPTIVE_METRICS ?= experiments/tracking/adaptive_metrics.json
DRIFT_REPORT ?= experiments/tracking/drift_scenarios_report.csv
CV_REPORT ?= experiments/tracking/cv_benchmark.csv
EVAL_DIR ?= experiments/tracking
THRESHOLD ?= 0.5

.PHONY: help run-api run-dashboard run-pipeline train-baseline run-adaptive run-drift-scenarios run-cv-benchmark run-evaluation run-all test lint format

help:
	@echo "Available targets:"
	@echo "  make run-api        - Start FastAPI development server"
	@echo "  make run-dashboard  - Start Streamlit dashboard"
	@echo "  make run-pipeline   - Run preprocessing pipeline on CSV"
	@echo "  make train-baseline - Train and evaluate baseline model"
	@echo "  make run-adaptive   - Run online/adaptive training loop"
	@echo "  make run-drift-scenarios - Evaluate detectors on synthetic drifts"
	@echo "  make run-cv-benchmark - Repeated CV benchmark for fair comparison"
	@echo "  make run-evaluation - Generate thesis evaluation artifacts"
	@echo "  make run-all        - Execute full thesis workflow end-to-end"
	@echo "  make test           - Run unit and integration tests"
	@echo "  make lint           - Run Ruff linter"
	@echo "  make format         - Run Ruff formatter"

run-api:
	PYTHONPATH=src uvicorn $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

run-dashboard:
	PYTHONPATH=src streamlit run dashboard/app.py --server.port 8501

run-pipeline:
	PYTHONPATH=src $(PYTHON) scripts/run_data_pipeline.py --input $(INPUT) --output $(DATA)

train-baseline:
	PYTHONPATH=src $(PYTHON) scripts/train_baseline.py --input $(DATA) --artifact $(ARTIFACT) --metrics-output $(BASELINE_METRICS) --threshold $(THRESHOLD)

train-baseline-hgb:
	PYTHONPATH=src $(PYTHON) scripts/train_baseline.py --input $(DATA) --artifact $(ARTIFACT) --metrics-output $(BASELINE_METRICS) --model hgb --tune-threshold

run-adaptive:
	PYTHONPATH=src $(PYTHON) scripts/run_adaptive_loop.py --input $(DATA) --output-summary $(ADAPTIVE_METRICS)

run-drift-scenarios:
	PYTHONPATH=src $(PYTHON) scripts/run_drift_scenarios.py --input $(DATA) --output-summary $(DRIFT_REPORT)

run-cv-benchmark:
	PYTHONPATH=src $(PYTHON) scripts/run_cv_benchmark.py --input $(DATA) --output $(CV_REPORT)

run-evaluation:
	PYTHONPATH=src $(PYTHON) scripts/run_evaluation.py --input $(INPUT) --output-dir $(EVAL_DIR)

run-all:
	PYTHONPATH=src $(PYTHON) scripts/run_data_pipeline.py --input $(INPUT) --output $(DATA)
	PYTHONPATH=src $(PYTHON) scripts/train_baseline.py --input $(DATA) --artifact $(ARTIFACT) --metrics-output $(BASELINE_METRICS) --threshold $(THRESHOLD)
	PYTHONPATH=src $(PYTHON) scripts/run_adaptive_loop.py --input $(DATA) --output-summary $(ADAPTIVE_METRICS)
	PYTHONPATH=src $(PYTHON) scripts/run_drift_scenarios.py --input $(DATA) --output-summary $(DRIFT_REPORT)
	PYTHONPATH=src $(PYTHON) scripts/run_cv_benchmark.py --input $(DATA) --output $(CV_REPORT)
	PYTHONPATH=src $(PYTHON) scripts/run_evaluation.py --input $(INPUT) --output-dir $(EVAL_DIR)

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
