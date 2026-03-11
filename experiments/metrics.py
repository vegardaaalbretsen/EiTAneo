"""Metrics helpers for consistent evaluation across experiments."""

from __future__ import annotations

import math
from typing import Mapping
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE as percentage of mean demand."""
    
    rmse_val = rmse(y_true, y_pred)

    mean_true = np.mean(np.abs(y_true))

    if mean_true == 0:
        return float("nan")

    return float((rmse_val / mean_true) * 100)

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except ValueError:
        return float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 0.1)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute all regression metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }

def regression_metrics_df(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
) -> dict[str, float]:
    """Compute metrics from a dataframe."""
    return regression_metrics(df[target_col], df[pred_col])


def with_sample_count(metrics: dict[str, float], n_samples: int) -> dict[str, float]:
    """Attach sample count to a metric dictionary."""
    output = dict(metrics)
    output["n_samples"] = int(n_samples)
    return output


def weighted_overall(
    segment_metrics: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:

    if not segment_metrics:
        raise ValueError("segment_metrics is empty.")

    total_samples = sum(m["n_samples"] for m in segment_metrics.values())

    mae = sum(m["mae"] * m["n_samples"] for m in segment_metrics.values()) / total_samples
    mape = sum(m["mape"] * m["n_samples"] for m in segment_metrics.values()) / total_samples

    mse = sum((m["rmse"] ** 2) * m["n_samples"] for m in segment_metrics.values()) / total_samples

    nrmse = sum(m["nrmse"] * m["n_samples"] for m in segment_metrics.values()) / total_samples

    r2_values = [
        (m["r2"], m["n_samples"])
        for m in segment_metrics.values()
        if math.isfinite(m["r2"])
    ]

    if r2_values:
        r2_weight = sum(w for _, w in r2_values)
        r2 = sum(v * w for v, w in r2_values) / r2_weight
    else:
        r2 = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(math.sqrt(mse)),
        "nrmse": float(nrmse),
        "r2": float(r2),
        "mape": float(mape),
        "n_samples": int(total_samples),
    }

def metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    pred_col: str,
) -> dict[str, dict]:

    results = {}

    for key, group in df.groupby(group_col):
        metrics = regression_metrics(group[target_col], group[pred_col])
        results[str(key)] = with_sample_count(metrics, len(group))

    return results


def metrics_by_window(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
) -> dict[int, dict]:

    results = {}

    for window, group in df.groupby("window"):
        metrics = regression_metrics(group[target_col], group[pred_col])
        results[int(window)] = with_sample_count(metrics, len(group))

    return results
