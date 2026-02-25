"""Metrics helpers for consistent evaluation across experiments."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate a consistent set of regression metrics."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(math.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    try:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    except ValueError:
        r2 = float("nan")

    denominator = np.maximum(np.abs(y_true_arr), 0.1)
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def with_sample_count(metrics: dict[str, float], n_samples: int) -> dict[str, float]:
    """Attach sample count to a metric dictionary."""
    output = dict(metrics)
    output["n_samples"] = int(n_samples)
    return output


def weighted_overall(segment_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Create weighted overall metrics from segmented test metrics."""
    if not segment_metrics:
        raise ValueError("segment_metrics is empty.")

    total_samples = sum(int(metrics["n_samples"]) for metrics in segment_metrics.values())
    if total_samples <= 0:
        raise ValueError("Total sample count must be greater than zero.")

    weighted_mae = (
        sum(float(metrics["mae"]) * int(metrics["n_samples"]) for metrics in segment_metrics.values())
        / total_samples
    )
    weighted_mape = (
        sum(float(metrics["mape"]) * int(metrics["n_samples"]) for metrics in segment_metrics.values())
        / total_samples
    )
    weighted_mse = (
        sum((float(metrics["rmse"]) ** 2) * int(metrics["n_samples"]) for metrics in segment_metrics.values())
        / total_samples
    )

    finite_r2 = [
        (float(metrics["r2"]), int(metrics["n_samples"]))
        for metrics in segment_metrics.values()
        if math.isfinite(float(metrics["r2"]))
    ]
    if finite_r2:
        r2_weight_sum = sum(weight for _, weight in finite_r2)
        weighted_r2 = sum(score * weight for score, weight in finite_r2) / r2_weight_sum
    else:
        weighted_r2 = float("nan")

    return {
        "mae": weighted_mae,
        "rmse": float(math.sqrt(weighted_mse)),
        "r2": float(weighted_r2),
        "mape": weighted_mape,
        "n_samples": int(total_samples),
    }
