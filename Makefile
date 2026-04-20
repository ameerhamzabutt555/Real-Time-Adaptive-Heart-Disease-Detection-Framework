PYTHON ?= python3
APP_MODULE ?= api.main:app

.PHONY: help run-api run-dashboard test lint format

help:
	@echo "Available targets:"
	@echo "  make run-api        - Start FastAPI development server"
	@echo "  make run-dashboard  - Start Streamlit dashboard"
	@echo "  make test           - Run unit and integration tests"
	@echo "  make lint           - Run Ruff linter"
	@echo "  make format         - Run Ruff formatter"

run-api:
	uvicorn $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
