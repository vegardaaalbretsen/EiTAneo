"""Two-group LightGBM strategy: Helsingfors vs Norway."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import (
    BASE_FEATURES,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    GLOBAL_FEATURES,
    TARGET_COLUMN,
)
from experiments.metrics import weighted_overall, with_sample_count
from experiments.models._lightgbm_utils import evaluate_lightgbm, train_lightgbm_regressor
from helpers.data_retrieval import chronological_split


class LightGBMTwoGroupExperiment(BaseExperiment):
    """Train one model for Helsingfors and one for Norwegian cities."""

    name = "lightgbm_two_group"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        group_definitions = {
            "helsingfors": {
                "mask": df["location_id"] == 0,
                "feature_cols": BASE_FEATURES,
            },
            "norway": {
                "mask": df["location_id"].isin([1, 2, 3, 4, 5]),
                "feature_cols": GLOBAL_FEATURES,
            },
        }

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        segment_test_metrics: dict[str, dict[str, float]] = {}
        segment_metadata: dict[str, dict[str, object]] = {}

        for group_name, group_config in group_definitions.items():
            group_df = df.loc[group_config["mask"]].copy()
            if group_df.empty:
                raise ValueError(f"No rows found for group '{group_name}'.")

            train_df, val_df, test_df = chronological_split(
                group_df,
                train_ratio=DEFAULT_TRAIN_RATIO,
                val_ratio=DEFAULT_VAL_RATIO,
            )

            model = train_lightgbm_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=group_config["feature_cols"],
                target_col=TARGET_COLUMN,
            )

            train_metrics = evaluate_lightgbm(
                model,
                train_df,
                group_config["feature_cols"],
                TARGET_COLUMN,
            )
            val_metrics = evaluate_lightgbm(
                model,
                val_df,
                group_config["feature_cols"],
                TARGET_COLUMN,
            )
            test_metrics = evaluate_lightgbm(
                model,
                test_df,
                group_config["feature_cols"],
                TARGET_COLUMN,
            )
            test_with_n = with_sample_count(test_metrics, len(test_df))
            segment_test_metrics[group_name] = test_with_n

            model_path = experiment_dir / f"{group_name}.txt"
            model.save_model(str(model_path))

            segment_metadata[group_name] = {
                "feature_columns": list(group_config["feature_cols"]),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

        overall_test_metrics = weighted_overall(segment_test_metrics)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=overall_test_metrics,
            segment_test_metrics=segment_test_metrics,
            metadata={"groups": segment_metadata},
        )
