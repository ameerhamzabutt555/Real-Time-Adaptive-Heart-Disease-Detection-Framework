# API Documentation (Baseline + Adaptive)

This document explains how the API works, what data to send, and what each response field means.

## Base URL

Local development defaults:
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

If you run on a different port (example 8001), your base URL becomes `http://localhost:8001`.

---

## Shared Data Models

### `PatientRecord` (request body for predictions)

This is the canonical feature schema used for **serving**, **training**, and **adaptive learning**.

**Fields**
- **`age`** *(integer)*: Patient age.
  - Constraints: `1 <= age <= 120`
- **`sex`** *(integer)*: Encoded sex.
  - Constraints: `0 or 1`
  - Typical convention: `0=female`, `1=male`
- **`resting_bp`** *(number)*: Resting blood pressure (mmHg).
  - Constraints: `> 0`
- **`cholesterol`** *(number)*: Serum cholesterol (mg/dl).
  - Constraints: `> 0`
- **`max_heart_rate`** *(number)*: Maximum heart rate achieved.
  - Constraints: `> 0`
- **`fasting_blood_sugar`** *(integer)*: Fasting blood sugar > 120 mg/dl indicator.
  - Constraints: `0 or 1`
  - Default: `0`
- **`restecg`** *(integer)*: Resting ECG results.
  - Constraints: `0..2`
  - Default: `0`
- **`exercise_angina`** *(integer)*: Exercise-induced angina indicator.
  - Constraints: `0 or 1`
  - Default: `0`
- **`chest_pain_type`** *(integer)*: Chest pain type category.
  - Constraints: `0..3`
  - Default: `0`
- **`oldpeak`** *(number)*: ST depression induced by exercise relative to rest.
  - Constraints: `>= 0`
  - Default: `0.0`
- **`slope`** *(integer)*: Slope of the peak exercise ST segment.
  - Constraints: `0..2`
  - Default: `1`
- **`ca`** *(integer)*: Number of major vessels colored by fluoroscopy.
  - Constraints: `0..4`
  - Default: `0`
- **`thal`** *(integer)*: Thalassemia category encoding.
  - Constraints: `0..3`
  - Default: `2`

**Example**

```json
{
  "age": 53,
  "sex": 1,
  "resting_bp": 132.0,
  "cholesterol": 246.0,
  "max_heart_rate": 151.0,
  "fasting_blood_sugar": 0,
  "restecg": 1,
  "exercise_angina": 1,
  "chest_pain_type": 2,
  "oldpeak": 1.3,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

---

### `PredictionResponse` (response body for predictions)

Returned by:
- `POST /predict`
- `POST /adaptive/predict`

**Fields**
- **`risk_score`** *(number)*: The model’s estimated probability-like score of heart disease.
  - Range: `0.0 .. 1.0`
  - Interpretation: Higher means higher predicted risk.
- **`predicted_label`** *(integer)*: Binary decision derived from `risk_score` and `threshold`.
  - `1` if `risk_score >= threshold`, else `0`
- **`model_version`** *(string)*: A human-readable identifier for which model produced the prediction.
  - Baseline typically: `baseline-v1`
  - Adaptive typically: `adaptive-online-v1`
  - If baseline artifact missing: `fallback-heuristic-v0`
- **`threshold`** *(number)*: Decision threshold used for the label.
  - Range: `0.0 .. 1.0`
  - Default is configured; some endpoints allow overrides.
- **`model_loaded`** *(boolean)*:
  - Baseline: `true` if the joblib artifact was found and loaded, otherwise `false`.
  - Adaptive: always `true` (the adaptive model is in-memory).

**Example**

```json
{
  "risk_score": 0.6732,
  "predicted_label": 1,
  "model_version": "baseline-v1",
  "threshold": 0.5,
  "model_loaded": true
}
```

---

## Endpoints

### 1) `GET /health`

**What it does**
- Quick liveness check to confirm the API process is running.

**Request body**
- None

**Response (200)**
- **`status`** *(string)*: Always `"ok"` when healthy.

```json
{ "status": "ok" }
```

**Curl**

```bash
curl -s http://localhost:8000/health
```

---

### 2) `GET /model-info`

**What it does**
- Reports baseline model metadata (version, threshold, and whether the trained model artifact is loaded).

**Request body**
- None

**Response (200)**
- **`model_version`** *(string)*: Version label from config (default: `baseline-v1`).
- **`threshold`** *(number)*: Baseline decision threshold.
- **`model_loaded`** *(boolean)*: `true` if `models/artifacts/baseline.joblib` is present and loadable.

```json
{
  "model_version": "baseline-v1",
  "threshold": 0.5,
  "model_loaded": true
}
```

**Notes**
- If `model_loaded=false`, `POST /predict` still works but uses a fallback heuristic score.

**Curl**

```bash
curl -s http://localhost:8000/model-info
```

---

### 3) `GET /adaptive/status`

**What it does**
- Returns the current status of the **online/adaptive** model running in memory.

**Request body**
- None

**Response (200)**
- **`model_version`** *(string)*: Adaptive model version label (default: `adaptive-online-v1`).
- **`detector`** *(string)*: Drift detector name being used (`adwin`, `ddm`, `page_hinkley`).
- **`threshold`** *(number)*: Default decision threshold for adaptive predictions.
- **`seen_samples`** *(integer)*: How many labeled samples have been learned via `POST /adaptive/learn`.
- **`drift_events`** *(integer)*: Count of drift detections so far.
- **`online_accuracy`** *(number)*: Online accuracy tracked over the stream.
- **`online_f1`** *(number)*: Online F1 tracked over the stream.

```json
{
  "model_version": "adaptive-online-v1",
  "detector": "adwin",
  "threshold": 0.5,
  "seen_samples": 124,
  "drift_events": 4,
  "online_accuracy": 0.781,
  "online_f1": 0.742
}
```

**Curl**

```bash
curl -s http://localhost:8000/adaptive/status
```

---

### 4) `POST /predict` (baseline)

**What it does**
- Performs a baseline prediction using the trained scikit-learn model artifact (if available).
- If the artifact is missing, it returns a deterministic fallback score.

**Request body**
- `PatientRecord` (JSON)

**Response (200)**
- `PredictionResponse` (JSON)

**Validation errors**
- If input fields violate constraints (example negative cholesterol), FastAPI returns a `422` validation error.

**Curl**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 53,
    "sex": 1,
    "resting_bp": 132.0,
    "cholesterol": 246.0,
    "max_heart_rate": 151.0,
    "fasting_blood_sugar": 0,
    "restecg": 1,
    "exercise_angina": 1,
    "chest_pain_type": 2,
    "oldpeak": 1.3,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
```

---

### 5) `POST /adaptive/predict`

**What it does**
- Returns a prediction from the in-memory online model (River pipeline).
- This endpoint **does not train** by itself; it only predicts.

**Request body**
- `PatientRecord` (JSON)

**Response (200)**
- `PredictionResponse` (JSON) with `model_version` set to the adaptive version.

**Curl**

```bash
curl -s -X POST http://localhost:8000/adaptive/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 53,
    "sex": 1,
    "resting_bp": 132.0,
    "cholesterol": 246.0,
    "max_heart_rate": 151.0,
    "fasting_blood_sugar": 0,
    "restecg": 1,
    "exercise_angina": 1,
    "chest_pain_type": 2,
    "oldpeak": 1.3,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
```

---

### 6) `POST /adaptive/learn` (online learning)

**What it does**
- Predicts on the given patient record, then learns from the provided true label.
- Updates:
  - online metrics (accuracy, F1),
  - drift detector state,
  - and the adaptive model weights.

**Request body (`AdaptiveLearnRequest`)**
- **`patient`** *(object)*: `PatientRecord`
- **`label`** *(integer)*: True label used for learning.
  - Constraints: `0 or 1`
- **`threshold`** *(number, optional)*: Threshold override for **this request** only.
  - Range: `0.0 .. 1.0`

**Example**

```json
{
  "patient": {
    "age": 53,
    "sex": 1,
    "resting_bp": 132.0,
    "cholesterol": 246.0,
    "max_heart_rate": 151.0,
    "fasting_blood_sugar": 0,
    "restecg": 1,
    "exercise_angina": 1,
    "chest_pain_type": 2,
    "oldpeak": 1.3,
    "slope": 1,
    "ca": 0,
    "thal": 2
  },
  "label": 1,
  "threshold": 0.55
}
```

**Response (200) (`AdaptiveLearnResponse`)**
This extends `PredictionResponse` and includes learning/drift fields:
- **`true_label`** *(integer)*: The label you sent.
- **`detector`** *(string)*: Drift detector name in use.
- **`drift_detected`** *(boolean)*: Whether drift was detected on this step.
- **`seen_samples`** *(integer)*: Total online learning samples seen so far.
- **`online_accuracy`** *(number)*: Current online accuracy.
- **`online_f1`** *(number)*: Current online F1.
- **`drift_events`** *(integer)*: Total drift events detected so far.

**Curl**

```bash
curl -s -X POST http://localhost:8000/adaptive/learn \
  -H 'Content-Type: application/json' \
  -d '{
    "patient": {
      "age": 53,
      "sex": 1,
      "resting_bp": 132.0,
      "cholesterol": 246.0,
      "max_heart_rate": 151.0,
      "fasting_blood_sugar": 0,
      "restecg": 1,
      "exercise_angina": 1,
      "chest_pain_type": 2,
      "oldpeak": 1.3,
      "slope": 1,
      "ca": 0,
      "thal": 2
    },
    "label": 1
  }'
```

---

## How the Adaptive System Works (summary)

The adaptive system is an **in-memory online model**:
- Online scaling: `StandardScaler()`
- Online classifier: `LogisticRegression()`

For each `/adaptive/learn` call:
1. It predicts a probability and label.
2. It computes an **error signal**: `1` if wrong, `0` if correct.
3. It updates a drift detector (ADWIN/DDM/PageHinkley) using that error signal.
4. It updates online metrics (accuracy, F1).
5. It trains incrementally via `learn_one(features, true_label)`.

---

## Configuration

### Baseline serving config
Loaded from `configs/model_config.json` (if present), with environment overrides:
- `MODEL_ARTIFACT_PATH`
- `MODEL_THRESHOLD`
- `MODEL_VERSION`

### Adaptive configuration
Environment variables:
- `ADAPTIVE_DETECTOR` (default: `adwin`)
- `ADAPTIVE_THRESHOLD` (default: `0.5`)

