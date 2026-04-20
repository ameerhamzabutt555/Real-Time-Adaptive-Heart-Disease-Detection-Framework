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


PredictionRequest = PatientRecord
