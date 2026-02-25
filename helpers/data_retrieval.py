"""Shared data loading and preprocessing helpers for model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_DATA_PATH = Path("data/preprocessed_data.csv")
LEGACY_DATA_PATH = Path("preprocessed_data.csv")

REQUIRED_COLUMNS = (
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "location_id",
    "temperature",
    "consumption",
)


def resolve_data_path(data_path: str | Path = DEFAULT_DATA_PATH) -> Path:
    """Resolve a dataset path while supporting legacy file locations."""
    candidate = Path(data_path)
    if candidate.exists():
        return candidate

    if candidate == DEFAULT_DATA_PATH and LEGACY_DATA_PATH.exists():
        return LEGACY_DATA_PATH

    raise FileNotFoundError(
        f"Could not find dataset at '{candidate}'. "
        f"Tried fallback '{LEGACY_DATA_PATH}' as well."
    )


def load_preprocessed_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the preprocessed modeling dataset."""
    path = resolve_data_path(data_path)
    df = pd.read_csv(path)
    validate_dataframe(df)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the expected columns exist and data is not empty."""
    missing_cols = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")


def validate_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "consumption",
) -> None:
    """Ensure selected feature/target columns are present in a dataframe."""
    requested = list(feature_cols) + [target_col]
    missing_cols = [column for column in requested if column not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Requested columns not found in dataframe: {missing}")


def split_features_target(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "consumption",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target arrays from a dataframe."""
    validate_feature_columns(df, feature_cols, target_col)
    features = df.loc[:, list(feature_cols)].copy()
    target = df.loc[:, target_col].copy()
    return features, target


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows in chronological order into train/val/test sets."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    n_rows = len(df)
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to make train/val/test splits.")

    train_end = max(1, int(n_rows * train_ratio))
    val_end = max(train_end + 1, int(n_rows * (train_ratio + val_ratio)))
    val_end = min(val_end, n_rows - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Split produced an empty subset. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    return train_df, val_df, test_df
