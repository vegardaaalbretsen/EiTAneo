"""Two-group LightGBM strategy: Helsingfors vs Norway."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import (
    BASE_FEATURES,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    GLOBAL_FEATURES,
    TARGET_COLUMN,
    DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
    DEFAULT_VAL_SIZE_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE_ROLLING_WINDOW,
    DEFAULT_STEP_SIZE_ROLLING_WINDOW,
)
from experiments.metrics import weighted_overall, with_sample_count
from experiments.models._lightgbm_utils import evaluate_lightgbm, train_lightgbm_regressor
from helpers.data_retrieval import chronological_split, rolling_window


class LightGBMTwoGroupExperiment(BaseExperiment):
    """Train one model for Helsingfors and one for Norwegian cities."""

    name = "lightgbm_two_group"

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:
        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(df, output_dir)
        if mode == RunModes.SLIDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="sliding_window")
        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="expanding_window")
        
        raise ValueError(f"Unsupported mode '{mode}' for Experiment. Use 'chronological', 'sliding_window', or 'expanding_window'.")
    def _run_chronological(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
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

        experiment_dir = output_dir / self.name / "chronological"
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

    
    def _run_rolling_window(self, df: pd.DataFrame, output_dir: Path, mode: str) -> ExperimentResult:
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

        experiment_dir = output_dir / self.name / mode
        experiment_dir.mkdir(parents=True, exist_ok=True)

        segment_test_metrics: dict[str, dict[str, float]] = {}
        segment_metadata: dict[str, dict[str, object]] = {}

        for group_name, group_config in group_definitions.items():

            group_df = (
                df.loc[group_config["mask"]]
                .sort_index()
                .copy()
            )

            if group_df.empty:
                raise ValueError(f"No rows found for group '{group_name}'.")

            window_results = []

            for i, (train_df, val_df, test_df) in enumerate(
                rolling_window(
                    group_df,
                    train_size=DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
                    val_size=DEFAULT_VAL_SIZE_ROLLING_WINDOW,
                    test_size=DEFAULT_TEST_SIZE_ROLLING_WINDOW,
                    step_size=DEFAULT_STEP_SIZE_ROLLING_WINDOW,
                    expanding=(mode == "expanding_window"),
                )
            ):
                model = train_lightgbm_regressor(
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=group_config["feature_cols"],
                    target_col=TARGET_COLUMN,
                )

                test_metrics = evaluate_lightgbm(
                    model,
                    test_df,
                    group_config["feature_cols"],
                    TARGET_COLUMN,
                )

                window_results.append({
                    "window": i,
                    "train_start": train_df.index.min(),
                    "train_end": train_df.index.max(),
                    "test_start": test_df.index.min(),
                    "test_end": test_df.index.max(),
                    "n_samples": len(test_df),
                    **test_metrics,
                })

            if not window_results:
                continue

            results_df = pd.DataFrame(window_results)

            # Save window metrics
            results_df.to_csv(
                experiment_dir / f"{group_name}_window_metrics.csv",
                index=False,
            )

            # TODO: add plots

            metric_cols = [
                c for c in results_df.columns
                if c not in [
                    "window",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                    "n_samples",
                ]
            ]

            total_samples = results_df["n_samples"].sum()

            weighted_metrics = {}
            for col in metric_cols:
                weighted_metrics[col] = (
                    (results_df[col] * results_df["n_samples"]).sum()
                    / total_samples
                )

            weighted_metrics["n_samples"] = int(total_samples)

            segment_test_metrics[group_name] = weighted_metrics

            segment_metadata[group_name] = {
                "mode": mode,
                "num_windows": len(results_df),
                "feature_columns": list(group_config["feature_cols"]),
            }

        overall_test_metrics = weighted_overall(segment_test_metrics)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=overall_test_metrics,
            segment_test_metrics=segment_test_metrics,
            metadata={
                "groups": segment_metadata,
                "mode": mode,
                },
        )