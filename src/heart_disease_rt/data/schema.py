"""Canonical schema and mappings for heart disease datasets."""

from __future__ import annotations

# Canonical feature set used across preprocessing, training and serving.
FEATURE_COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "restecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Backward-compatible alias used by older modules.
CANONICAL_FEATURES = FEATURE_COLUMNS

LABEL_COLUMN = "label"

NUMERIC_COLUMNS = ["age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak"]

TARGET_CANDIDATES = ["target", "num", "label", "heart_disease"]

ALIASES_TO_CANONICAL = {
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
}
