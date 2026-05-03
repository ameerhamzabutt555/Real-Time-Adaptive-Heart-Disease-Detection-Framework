import os

from fastapi import FastAPI, HTTPException

from heart_disease_rt.serving.adaptive_service import AdaptiveHeartDiseaseService
from heart_disease_rt.serving.predictor import HeartDiseasePredictor
from heart_disease_rt.serving.schemas import (
    AdaptiveLearnRequest,
    AdaptiveLearnResponse,
    AdaptiveStatusResponse,
    PatientRecord,
    PredictionResponse,
    ThresholdUpdateRequest,
)

app = FastAPI(title="Heart Disease RT API", version="0.1.0")
predictor = HeartDiseasePredictor()
adaptive_service = AdaptiveHeartDiseaseService(
    detector_name=os.getenv("ADAPTIVE_DETECTOR", "adwin"),
    threshold=float(os.getenv("ADAPTIVE_THRESHOLD", "0.5")),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "model_loaded": predictor.model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: PatientRecord) -> PredictionResponse:
    return predictor.predict(payload)


@app.post("/predict/threshold", response_model=PredictionResponse)
def predict_with_threshold(payload: ThresholdUpdateRequest) -> PredictionResponse:
    try:
        return predictor.predict(payload.patient, threshold_override=payload.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/adaptive/status", response_model=AdaptiveStatusResponse)
def adaptive_status() -> AdaptiveStatusResponse:
    return adaptive_service.status()


@app.post("/adaptive/predict", response_model=PredictionResponse)
def adaptive_predict(payload: PatientRecord) -> PredictionResponse:
    return adaptive_service.predict(payload)


@app.post("/adaptive/learn", response_model=AdaptiveLearnResponse)
def adaptive_learn(payload: AdaptiveLearnRequest) -> AdaptiveLearnResponse:
    try:
        return adaptive_service.learn(
            payload.patient,
            true_label=payload.label,
            threshold_override=payload.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
