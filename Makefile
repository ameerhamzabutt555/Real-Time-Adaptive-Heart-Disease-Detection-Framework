PYTHON ?= python3
APP_MODULE ?= api.main:app
INPUT ?= data/raw/heart.csv
DATA ?= data/processed/heart_disease_processed.csv
ARTIFACT ?= models/artifacts/baseline.joblib
BASELINE_METRICS ?= experiments/tracking/baseline_metrics.json
ADAPTIVE_METRICS ?= experiments/tracking/adaptive_metrics.json

.PHONY: help run-api run-dashboard run-pipeline train-baseline run-adaptive test lint format

help:
	@echo "Available targets:"
	@echo "  make run-api        - Start FastAPI development server"
	@echo "  make run-dashboard  - Start Streamlit dashboard"
	@echo "  make run-pipeline   - Run preprocessing pipeline on CSV"
	@echo "  make train-baseline - Train and evaluate baseline model"
	@echo "  make run-adaptive   - Run online/adaptive training loop"
	@echo "  make test           - Run unit and integration tests"
	@echo "  make lint           - Run Ruff linter"
	@echo "  make format         - Run Ruff formatter"

run-api:
	uvicorn $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

run-pipeline:
	$(PYTHON) scripts/run_data_pipeline.py --input $(INPUT) --output $(DATA)

train-baseline:
	$(PYTHON) scripts/train_baseline.py --input $(DATA) --artifact $(ARTIFACT) --metrics-output $(BASELINE_METRICS)

run-adaptive:
	$(PYTHON) scripts/run_adaptive_loop.py --input $(DATA) --output $(ADAPTIVE_METRICS)

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
