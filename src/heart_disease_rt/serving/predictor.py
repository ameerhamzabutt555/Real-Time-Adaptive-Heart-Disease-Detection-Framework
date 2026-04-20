from heart_disease_rt.serving.schemas import PatientRecord, PredictionResponse


class HeartDiseasePredictor:
    """Minimal predictor scaffold with deterministic placeholder scoring."""

    def predict(self, record: PatientRecord) -> PredictionResponse:
        score = min(
            1.0,
            max(
                0.0,
                (0.002 * record.age)
                + (0.001 * record.resting_bp)
                + (0.0005 * record.cholesterol),
            ),
        )
        label = int(score >= 0.5)
        return PredictionResponse(risk_score=round(score, 4), predicted_label=label)
