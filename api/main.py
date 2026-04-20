from fastapi import FastAPI

from heart_disease_rt.serving.predictor import HeartDiseasePredictor
from heart_disease_rt.serving.schemas import PatientRecord, PredictionResponse

app = FastAPI(title="Heart Disease RT API", version="0.1.0")
predictor = HeartDiseasePredictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: PatientRecord) -> PredictionResponse:
    return predictor.predict(payload)
