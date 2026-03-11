"""Single XGBoost model trained on all locations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import (
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    GLOBAL_FEATURES,
    TARGET_COLUMN,
    DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
    DEFAULT_VAL_SIZE_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE_ROLLING_WINDOW,
    DEFAULT_STEP_SIZE_ROLLING_WINDOW,
)
from experiments.metrics import with_sample_count
from helpers.data_retrieval import chronological_split, rolling_window, split_features_target
from experiments.models._xgboost_utils import evaluate_xgboost, train_xgboost_regressor


class XGBoostGlobalExperiment(BaseExperiment):
    """Global XGBoost model with location_id as a feature."""

    name = "xgboost_global"

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:
        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(df, output_dir)
        if mode == RunModes.SLIDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="sliding_window")
        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="expanding_window")
        raise ValueError(f"Unsupported mode '{mode}' for Experiment. Use 'chronological', 'sliding_window', or 'expanding_window'.")
    
    def _run_chronological(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        model = train_xgboost_regressor(
            train_df=train_df,
            val_df=val_df,
            feature_cols=GLOBAL_FEATURES,
            target_col=TARGET_COLUMN,
        )

        train_metrics = evaluate_xgboost(model, train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        val_metrics = evaluate_xgboost(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_metrics = evaluate_xgboost(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.xgb"
        # xgboost Booster has save_model
        model.save_model(str(model_path))

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata={
                "feature_columns": list(GLOBAL_FEATURES),
                "model_path": str(model_path),
                "best_iteration": int(getattr(model, "best_iteration", -1)),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )
    
    def _run_rolling_window(self, df: pd.DataFrame, output_dir: Path, mode: str) -> ExperimentResult:

        experiment_dir = output_dir / self.name / mode
        experiment_dir.mkdir(parents=True, exist_ok=True)

        window_results = []

        for i, (train_df, val_df, test_df) in enumerate(
            rolling_window(
                df,
                train_size=DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
                val_size=DEFAULT_VAL_SIZE_ROLLING_WINDOW,
                test_size=DEFAULT_TEST_SIZE_ROLLING_WINDOW,
                step_size=DEFAULT_STEP_SIZE_ROLLING_WINDOW,
                expanding=(mode == "expanding_window"),
            )
        ):
            model = train_xgboost_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=GLOBAL_FEATURES,
                target_col=TARGET_COLUMN,
            )

            test_metrics = evaluate_xgboost(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)

            window_results.append({
                "window": i,
                "train_start": train_df.index.min(),
                "train_end": train_df.index.max(),
                "test_start": test_df.index.min(),
                "test_end": test_df.index.max(),
                "n_samples": len(test_df),
                **test_metrics
            })

        if not window_results:
            raise ValueError("No rolling windows were generated.")

        results_df = pd.DataFrame(window_results)

        # Weighted aggregation across windows
        metric_cols = [
            c for c in results_df.columns
            if c not in ["window", "train_start", "train_end", "test_start", "test_end", "n_samples"]
        ]
        total_samples = results_df["n_samples"].sum()
        weighted_metrics = {}
        for col in metric_cols:
            weighted_metrics[col] = (results_df[col] * results_df["n_samples"]).sum() / total_samples
        weighted_metrics["n_samples"] = int(total_samples)

        # Save window metrics for inspection
        results_df.to_csv(experiment_dir / "window_metrics.csv", index=False)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=weighted_metrics,
            segment_test_metrics={"rolling": weighted_metrics},
            metadata={
                "num_windows": len(results_df),
                "mode": mode,
            },
        )

