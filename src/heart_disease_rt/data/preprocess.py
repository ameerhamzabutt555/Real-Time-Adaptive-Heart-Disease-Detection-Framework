"""Preprocessing helpers for heart disease datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from heart_disease_rt.data.schema import (
    ALIASES_TO_CANONICAL,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_CANDIDATES,
)


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize incoming dataset columns to the canonical schema."""
    normalized = df.rename(columns=ALIASES_TO_CANONICAL).copy()
    normalized.columns = [column.strip().lower() for column in normalized.columns]

    if LABEL_COLUMN not in normalized.columns:
        for candidate in TARGET_CANDIDATES:
            if candidate in normalized.columns:
                normalized = normalized.rename(columns={candidate: LABEL_COLUMN})
                break
    return normalized


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS + [LABEL_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reproducible cleaning and type normalization."""
    processed = normalize_columns(df)
    validate_required_columns(processed)
    processed = processed[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()

    for column in NUMERIC_COLUMNS:
        if processed[column].isna().any():
            processed[column] = processed[column].fillna(processed[column].median())

    for column in FEATURE_COLUMNS:
        if column not in NUMERIC_COLUMNS:
            processed[column] = processed[column].fillna(processed[column].mode().iat[0])

    processed[LABEL_COLUMN] = processed[LABEL_COLUMN].astype(int).clip(0, 1)
    return processed


def train_test_split_time_order(df: pd.DataFrame, train_ratio: float = 0.8) -> SplitData:
    """Time-order preserving split for realistic simulation of future data."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    return SplitData(train=train_df, test=test_df)
