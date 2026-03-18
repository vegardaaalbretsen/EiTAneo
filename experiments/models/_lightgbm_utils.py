"""Shared LightGBM training/evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.config import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_LGB_PARAMS,
    DEFAULT_LOCATION_NAMES,
    DEFAULT_NUM_BOOST_ROUND,
    TARGET_COLUMN,
)
from experiments.metrics import regression_metrics
from helpers.data_retrieval import split_features_target
from helpers.plotter import Plotter


def train_lightgbm_regressor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COLUMN,
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    log_period: int = 0,
):
    """Train a LightGBM regressor with a standard setup."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required for this experiment. Install with `pip install lightgbm`."
        ) from exc

    resolved_params = dict(DEFAULT_LGB_PARAMS)
    if params:
        resolved_params.update(params)

    X_train, y_train = split_features_target(train_df, feature_cols, target_col)
    X_val, y_val = split_features_target(val_df, feature_cols, target_col)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(feature_cols))
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=list(feature_cols))

    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    if log_period > 0:
        callbacks.append(lgb.log_evaluation(period=log_period))

    model = lgb.train(
        resolved_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model


def evaluate_lightgbm(
    model,
    split_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COLUMN,
) -> dict[str, float]:
    """Evaluate a trained LightGBM model on one split."""
    prediction_df = lightgbm_prediction_frame(model, split_df, feature_cols, target_col)
    return regression_metrics(prediction_df["actual"], prediction_df["predicted"])


def lightgbm_prediction_frame(
    model,
    split_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Return actual and predicted values for a dataframe split."""
    X_split, y_split = split_features_target(split_df, feature_cols, target_col)
    best_iteration = getattr(model, "best_iteration", None)
    predictions = model.predict(X_split, num_iteration=best_iteration)

    prediction_df = pd.DataFrame(
        {
            "actual": y_split.to_numpy(),
            "predicted": predictions,
        },
        index=split_df.index.copy(),
    )

    if "location_id" in split_df.columns:
        prediction_df["location_id"] = split_df["location_id"].to_numpy()

    prediction_df.index.name = split_df.index.name or "row_index"
    return prediction_df


def save_lightgbm_prediction_artifacts(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    filename_stem: str,
    title: str,
) -> dict[str, str]:
    """Persist test predictions and a comparison plot for a LightGBM model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename_stem}.csv"
    plot_path = output_dir / f"{filename_stem}_actual_vs_predicted.png"

    predictions_df.to_csv(csv_path, index=True)
    Plotter.create_actual_vs_predicted_plot(
        predictions_df=predictions_df,
        output_path=plot_path,
        title=title,
    )
    daily_plot_dir = output_dir / f"{filename_stem}_daily_line_plots"
    daily_plot_paths = Plotter.create_daily_line_plots_by_location(
        predictions_df=predictions_df,
        output_dir=daily_plot_dir,
        filename_stem=filename_stem,
        title_prefix=title,
        location_names=DEFAULT_LOCATION_NAMES,
    )

    return {
        "predictions_path": str(csv_path),
        "prediction_plot_path": str(plot_path),
        "daily_prediction_plot_dir": str(daily_plot_dir),
        "daily_prediction_plot_paths": daily_plot_paths,
    }
