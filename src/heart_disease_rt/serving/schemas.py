from pydantic import BaseModel, Field


class PatientRecord(BaseModel):
    """Validated payload shape for inference endpoints."""

    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    resting_bp: float = Field(gt=0)
    cholesterol: float = Field(gt=0)
    max_heart_rate: float = Field(gt=0)
    fasting_blood_sugar: int = Field(default=0, ge=0, le=1)
    restecg: int = Field(default=0, ge=0, le=2)
    exercise_angina: int = Field(default=0, ge=0, le=1)
    chest_pain_type: int = Field(default=0, ge=0, le=3)
    oldpeak: float = Field(default=0.0, ge=0)
    slope: int = Field(default=1, ge=0, le=2)
    ca: int = Field(default=0, ge=0, le=4)
    thal: int = Field(default=2, ge=0, le=3)


class PredictionResponse(BaseModel):
    """Normalized API output for risk predictions."""

    risk_score: float = Field(ge=0.0, le=1.0)
    predicted_label: int = Field(ge=0, le=1)
    model_version: str = "baseline-v0"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_loaded: bool = False


class ThresholdUpdateRequest(BaseModel):
    patient: PatientRecord
    threshold: float = Field(ge=0.0, le=1.0)


class AdaptiveLearnRequest(BaseModel):
    patient: PatientRecord
    label: int = Field(ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class AdaptiveLearnResponse(PredictionResponse):
    true_label: int = Field(ge=0, le=1)
    detector: str
    drift_detected: bool
    seen_samples: int = Field(ge=0)
    online_accuracy: float = Field(ge=0.0, le=1.0)
    online_f1: float = Field(ge=0.0, le=1.0)
    drift_events: int = Field(ge=0)
    learn_pending: int = Field(
        default=0,
        ge=0,
        description="Labeled samples queued; weights update only after a full batch flush.",
    )
    learn_applied_now: int = Field(
        default=0,
        ge=0,
        description="How many samples were applied to the model in this request (0 if only queued).",
    )


class FeatureStreamStats(BaseModel):
    """Online summary for one feature over all patients seen by the adaptive service (predict/learn/observe)."""

    n: int = Field(ge=0, description="Number of times this feature was observed.")
    mean: float = Field(description="Running mean (Welford).")
    variance: float = Field(ge=0.0, description="Unbiased sample variance when n>=2, else 0.")
    last: float = Field(description="Most recent observed value.")


class AdaptiveFeatureTrackingResponse(BaseModel):
    """Per-feature distribution tracking (same snapshot as `feature_streams` on /adaptive/status)."""

    streams: dict[str, FeatureStreamStats]


class AdaptiveStatusResponse(BaseModel):
    model_version: str
    model_type: str
    detector: str
    threshold: float = Field(ge=0.0, le=1.0)
    seen_samples: int = Field(ge=0)
    drift_events: int = Field(ge=0)
    online_accuracy: float = Field(ge=0.0, le=1.0)
    online_f1: float = Field(ge=0.0, le=1.0)
    learn_pending: int = Field(default=0, ge=0, description="Queued labeled updates not yet applied.")
    commit_batch_size: int = Field(default=1, ge=1, description="Apply weight updates every N labeled samples.")
    checkpoint_dir: str | None = Field(default=None, description="Directory for saved checkpoints, if configured.")
    feature_streams: dict[str, FeatureStreamStats] = Field(
        default_factory=dict,
        description="Running stats for every canonical feature (age, bp, chol, …), not BP only.",
    )


class AdaptiveCheckpointEntry(BaseModel):
    filename: str
    size_bytes: int


class AdaptiveCheckpointRestoreRequest(BaseModel):
    filename: str = Field(..., min_length=1, description="Basename only, e.g. checkpoint_00003.pkl")


class AdaptiveCheckpointRestoreResponse(BaseModel):
    restored: bool
    filename: str
    seen_samples: int = Field(ge=0)


class AdaptiveLearnFlushResponse(BaseModel):
    flushed: int = Field(ge=0, description="Number of queued samples applied to the model.")
    seen_samples: int = Field(ge=0)
    online_accuracy: float = Field(ge=0.0, le=1.0)
    online_f1: float = Field(ge=0.0, le=1.0)
    drift_events: int = Field(ge=0)


class AdaptiveObserveRequest(BaseModel):
    """Predict for a patient and optionally learn if a label is provided.

    This supports production-style flows where labels arrive later. If you have a label, send it to
    update the online model; otherwise omit it to do prediction-only.
    """

    patient: PatientRecord
    label: int | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    patient_id: str | None = Field(default=None, min_length=1)


class AdaptiveObserveResponse(PredictionResponse):
    """Response for observe calls (predict-only or predict+learn)."""

    learned: bool
    patient_id: str | None = None
    seen_samples: int = Field(ge=0)
    detector: str
    drift_detected: bool
    drift_events: int = Field(ge=0)
    online_accuracy: float = Field(ge=0.0, le=1.0)
    online_f1: float = Field(ge=0.0, le=1.0)
    learn_pending: int = Field(default=0, ge=0)
    learn_applied_now: int = Field(default=0, ge=0)


PredictionRequest = PatientRecord
