"""Location-specific LightGBM strategy: one model per location."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import (
    BASE_FEATURES,
    DEFAULT_LOCATION_NAMES,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    TARGET_COLUMN,
)
from experiments.metrics import weighted_overall, with_sample_count
from experiments.models._lightgbm_utils import evaluate_lightgbm, train_lightgbm_regressor
from helpers.data_retrieval import chronological_split


class LightGBMLocationSpecificExperiment(BaseExperiment):
    """Train one LightGBM model per location_id."""

    name = "lightgbm_location_specific"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        segment_test_metrics: dict[str, dict[str, float]] = {}
        segment_metadata: dict[str, dict[str, object]] = {}

        for location_id in sorted(df["location_id"].unique()):
            location_df = df.loc[df["location_id"] == location_id].copy()
            if location_df.empty:
                continue

            train_df, val_df, test_df = chronological_split(
                location_df,
                train_ratio=DEFAULT_TRAIN_RATIO,
                val_ratio=DEFAULT_VAL_RATIO,
            )

            model = train_lightgbm_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=BASE_FEATURES,
                target_col=TARGET_COLUMN,
            )

            train_metrics = evaluate_lightgbm(model, train_df, BASE_FEATURES, TARGET_COLUMN)
            val_metrics = evaluate_lightgbm(model, val_df, BASE_FEATURES, TARGET_COLUMN)
            test_metrics = evaluate_lightgbm(model, test_df, BASE_FEATURES, TARGET_COLUMN)
            test_with_n = with_sample_count(test_metrics, len(test_df))

            location_name = DEFAULT_LOCATION_NAMES.get(int(location_id), f"location_{location_id}")
            segment_key = f"{int(location_id)}_{location_name.lower()}"
            segment_test_metrics[segment_key] = test_with_n

            model_path = experiment_dir / f"{segment_key}.txt"
            model.save_model(str(model_path))

            segment_metadata[segment_key] = {
                "location_id": int(location_id),
                "location_name": location_name,
                "feature_columns": list(BASE_FEATURES),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

        if not segment_test_metrics:
            raise ValueError("No location-specific models were trained.")

        overall_test_metrics = weighted_overall(segment_test_metrics)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=overall_test_metrics,
            segment_test_metrics=segment_test_metrics,
            metadata={"locations": segment_metadata},
        )
