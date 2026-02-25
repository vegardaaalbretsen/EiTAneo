"""Single LightGBM model trained on all locations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, GLOBAL_FEATURES, TARGET_COLUMN
from experiments.metrics import with_sample_count
from experiments.models._lightgbm_utils import evaluate_lightgbm, train_lightgbm_regressor
from helpers.data_retrieval import chronological_split


class LightGBMGlobalExperiment(BaseExperiment):
    """Global LightGBM model with location_id as a feature."""

    name = "lightgbm_global"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        model = train_lightgbm_regressor(
            train_df=train_df,
            val_df=val_df,
            feature_cols=GLOBAL_FEATURES,
            target_col=TARGET_COLUMN,
        )

        train_metrics = evaluate_lightgbm(model, train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        val_metrics = evaluate_lightgbm(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_metrics = evaluate_lightgbm(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.txt"
        model.save_model(str(model_path))

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata={
                "feature_columns": list(GLOBAL_FEATURES),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )
