import os

from fastapi import FastAPI, HTTPException

from heart_disease_rt.serving.adaptive_service import AdaptiveHeartDiseaseService
from heart_disease_rt.serving.predictor import HeartDiseasePredictor
from heart_disease_rt.serving.schemas import (
    AdaptiveCheckpointEntry,
    AdaptiveCheckpointRestoreRequest,
    AdaptiveCheckpointRestoreResponse,
    AdaptiveFeatureTrackingResponse,
    AdaptiveLearnFlushResponse,
    AdaptiveLearnRequest,
    AdaptiveLearnResponse,
    AdaptiveObserveRequest,
    AdaptiveObserveResponse,
    AdaptiveStatusResponse,
    PatientRecord,
    PredictionResponse,
    ThresholdUpdateRequest,
)

app = FastAPI(title="Heart Disease RT API", version="0.1.0")
predictor = HeartDiseasePredictor()
_ckpt_dir = os.getenv("ADAPTIVE_CHECKPOINT_DIR", "").strip()
adaptive_service = AdaptiveHeartDiseaseService(
    detector_name=os.getenv("ADAPTIVE_DETECTOR", "adwin"),
    threshold=float(os.getenv("ADAPTIVE_THRESHOLD", "0.5")),
    on_drift=os.getenv("ADAPTIVE_ON_DRIFT", "log_only"),
    model_type=os.getenv("ADAPTIVE_MODEL_TYPE", "logreg"),
    commit_batch_size=int(os.getenv("ADAPTIVE_COMMIT_BATCH_SIZE", "1")),
    checkpoint_dir=_ckpt_dir or None,
)


@app.on_event("startup")
def warm_start_adaptive() -> None:
    """Optionally warm-start the adaptive model from CSV.

    Set:
    - ADAPTIVE_WARM_START_PATH=data/processed/heart_uci_303_processed.csv
    - ADAPTIVE_WARM_START_LIMIT=303 (optional)
    - ADAPTIVE_LOAD_LATEST_ON_START=1 to load newest checkpoint after warm-start (overrides weights)
    """
    warm_path = os.getenv("ADAPTIVE_WARM_START_PATH", "").strip()
    if warm_path:
        limit_raw = os.getenv("ADAPTIVE_WARM_START_LIMIT", "").strip()
        limit = int(limit_raw) if limit_raw else None
        try:
            adaptive_service.warm_start_from_csv(warm_path, limit=limit)
        except Exception:
            pass

    if os.getenv("ADAPTIVE_LOAD_LATEST_ON_START", "").strip().lower() in ("1", "true", "yes"):
        try:
            adaptive_service.try_load_latest_checkpoint()
        except Exception:
            pass


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


@app.get("/adaptive/feature-tracking", response_model=AdaptiveFeatureTrackingResponse)
def adaptive_feature_tracking() -> AdaptiveFeatureTrackingResponse:
    """Running mean/variance/last for every canonical feature (full patient vector, not BP only)."""
    return adaptive_service.feature_tracking_response()


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


@app.post("/adaptive/observe", response_model=AdaptiveObserveResponse)
def adaptive_observe(payload: AdaptiveObserveRequest) -> AdaptiveObserveResponse:
    """Predict and optionally learn (if label is provided)."""
    try:
        return adaptive_service.observe(
            payload.patient,
            label=payload.label,
            threshold_override=payload.threshold,
            patient_id=payload.patient_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/adaptive/learn/flush", response_model=AdaptiveLearnFlushResponse)
def adaptive_learn_flush() -> AdaptiveLearnFlushResponse:
    """Apply all queued labeled samples (commit before batch is full)."""
    return adaptive_service.flush_learn_buffer()


@app.get("/adaptive/checkpoints", response_model=list[AdaptiveCheckpointEntry])
def adaptive_list_checkpoints() -> list[AdaptiveCheckpointEntry]:
    entries: list[AdaptiveCheckpointEntry] = []
    for path in adaptive_service.list_checkpoint_files():
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        entries.append(AdaptiveCheckpointEntry(filename=path.name, size_bytes=size))
    return entries


@app.post("/adaptive/checkpoint/restore", response_model=AdaptiveCheckpointRestoreResponse)
def adaptive_restore_checkpoint(payload: AdaptiveCheckpointRestoreRequest) -> AdaptiveCheckpointRestoreResponse:
    try:
        adaptive_service.restore_checkpoint_by_filename(payload.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    st = adaptive_service.status()
    return AdaptiveCheckpointRestoreResponse(
        restored=True,
        filename=payload.filename,
        seen_samples=st.seen_samples,
    )
