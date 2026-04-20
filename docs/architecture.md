# Adaptive Real-Time System Architecture

## Components

1. **Data Layer**
   - `run_data_pipeline.py` -> `preprocess_dataframe`
   - normalizes schema, handles missing values, enforces label format

2. **Static Baseline Layer**
   - `run_baseline_training` (logistic regression pipeline)
   - persists model artifact and serving config (`configs/model_config.json`)

3. **Adaptive Streaming Layer**
   - `run_adaptive_comparison` runs online logistic model with detectors:
     - ADWIN
     - DDM
     - Page-Hinkley
   - exports summary/progress for detector comparison

4. **Drift Scenario Evaluation Layer**
   - synthetic scenarios:
     - sudden drift
     - gradual drift
     - recurring drift
   - reports:
     - detection delay
     - recovery delay
     - false alarms before drift

5. **Serving Layer**
   - FastAPI `/predict` for model inference
   - `/predict/threshold` for threshold override experiments
   - `/model-info` for serving metadata visibility

6. **Monitoring & Reporting Layer**
   - plots for adaptive progress + detector comparison
   - summary JSON/CSV artifacts consumed by dashboard

7. **Dashboard Layer**
   - Streamlit app reads tracked artifacts
   - shows baseline/adaptive KPIs, detector table, and drift visuals

## Data Flow

1. Raw CSV -> processed CSV
2. Processed CSV -> baseline model + config
3. Processed CSV -> adaptive comparison + drift scenario analysis
4. Artifacts -> dashboard + thesis reports
5. Baseline artifact + config -> API real-time predictions
