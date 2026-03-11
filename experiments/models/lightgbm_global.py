"""Single LightGBM model trained on all locations."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import (
    DEFAULT_STEP_SIZE_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE_ROLLING_WINDOW,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SIZE_ROLLING_WINDOW,
    GLOBAL_FEATURES,
    LIGHTGBM_GLOBAL_PARAM_GRID,
    TARGET_COLUMN,
)
from experiments.metrics import with_sample_count
from experiments.models._lightgbm_utils import evaluate_lightgbm, train_lightgbm_regressor
from helpers.data_retrieval import chronological_split, rolling_window


class LightGBMGlobalExperiment(BaseExperiment):
    """Global LightGBM model with location_id as a feature."""

    name = "lightgbm_global"

    def _use_grid_search(self) -> bool:
        return bool(self.options.get("lightgbm_grid_search", False))

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:
        tune_hyperparameters = self._use_grid_search()
        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(
                df,
                output_dir,
                tune_hyperparameters=tune_hyperparameters,
            )

        if mode == RunModes.SLIDING_WINDOW:
            return self._run_sliding_window(
                df,
                output_dir,
                mode="sliding_window",
                tune_hyperparameters=tune_hyperparameters,
            )

        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_sliding_window(
                df,
                output_dir,
                mode="expanding_window",
                tune_hyperparameters=tune_hyperparameters,
            )

        raise ValueError(
            f"Unsupported mode '{mode}' for Experiment. Use "
            "'chronological', 'sliding_window', or 'expanding_window'."
        )

    def _get_param_grid(self) -> list[dict[str, Any]]:
        keys = list(LIGHTGBM_GLOBAL_PARAM_GRID.keys())
        values = list(LIGHTGBM_GLOBAL_PARAM_GRID.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _select_best_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ):
        best_model = None
        best_params = None
        best_val_metrics = None
        best_score = float("inf")
        search_rows: list[dict[str, Any]] = []

        for trial_idx, params in enumerate(self._get_param_grid(), start=1):
            model = train_lightgbm_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=GLOBAL_FEATURES,
                target_col=TARGET_COLUMN,
                params=params,
            )
            val_metrics = evaluate_lightgbm(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)
            score = val_metrics["mae"]

            search_row = {
                "trial": trial_idx,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_mape": val_metrics["mape"],
                "best_iteration": int(getattr(model, "best_iteration", -1)),
            }
            search_row.update({f"param_{key}": value for key, value in params.items()})
            search_rows.append(search_row)

            if score < best_score:
                best_score = score
                best_model = model
                best_params = params
                best_val_metrics = val_metrics

        if best_model is None or best_params is None or best_val_metrics is None:
            raise RuntimeError("LightGBM parameter search failed to produce a valid model.")

        search_df = (
            pd.DataFrame(search_rows)
            .sort_values("val_mae", ascending=True)
            .reset_index(drop=True)
        )
        return best_model, best_params, best_val_metrics, search_df

    def _run_chronological(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        tune_hyperparameters: bool,
    ) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        search_df: pd.DataFrame | None = None
        best_params: dict[str, Any] | None = None
        if tune_hyperparameters:
            model, best_params, val_metrics, search_df = self._select_best_model(
                train_df=train_df,
                val_df=val_df,
            )
        else:
            model = train_lightgbm_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=GLOBAL_FEATURES,
                target_col=TARGET_COLUMN,
            )
            val_metrics = evaluate_lightgbm(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)

        train_metrics = evaluate_lightgbm(model, train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_metrics = evaluate_lightgbm(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name / "chronological"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.txt"
        search_results_path: Path | None = None
        model.save_model(str(model_path))

        metadata: dict[str, Any] = {
            "feature_columns": list(GLOBAL_FEATURES),
            "model_path": str(model_path),
            "best_iteration": int(model.best_iteration),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "tune_hyperparameters": tune_hyperparameters,
        }

        if search_df is not None:
            search_results_path = experiment_dir / "grid_search_results.csv"
            search_df.to_csv(search_results_path, index=False)
            metadata["best_params"] = best_params
            metadata["num_grid_candidates"] = int(len(search_df))
            metadata["grid_search_results_path"] = str(search_results_path)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata=metadata,
        )

    def _run_sliding_window(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        mode: str,
        tune_hyperparameters: bool,
    ) -> ExperimentResult:
        window_results = []
        all_grid_trials: list[pd.DataFrame] = []
        experiment_dir = output_dir / self.name / mode
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for i, (train_df, val_df, test_df) in enumerate(
            rolling_window(
                df,
                train_size=DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,  # 5 cities, 24 hours, 20 days
                val_size=DEFAULT_VAL_SIZE_ROLLING_WINDOW,  # 5 cities, 24 hours, 3 days
                test_size=DEFAULT_TEST_SIZE_ROLLING_WINDOW,  # 5 cities, 24 hours, 1 day
                step_size=DEFAULT_STEP_SIZE_ROLLING_WINDOW,  # 5 cities, 24 hours, 1 day step
                expanding=(mode == "expanding_window"),
            )
        ):
            if tune_hyperparameters:
                model, best_params, val_metrics, search_df = self._select_best_model(
                    train_df=train_df,
                    val_df=val_df,
                )
                search_df.insert(0, "window", i)
                all_grid_trials.append(search_df)
            else:
                model = train_lightgbm_regressor(
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=GLOBAL_FEATURES,
                    target_col=TARGET_COLUMN,
                )
                best_params = None
                val_metrics = evaluate_lightgbm(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)

            test_metrics = evaluate_lightgbm(
                model,
                test_df,
                GLOBAL_FEATURES,
                TARGET_COLUMN,
            )

            window_results.append(
                {
                    "window": i,
                    "train_start": train_df.index.min(),
                    "train_end": train_df.index.max(),
                    "val_start": val_df.index.min(),
                    "val_end": val_df.index.max(),
                    "test_start": test_df.index.min(),
                    "test_end": test_df.index.max(),
                    "best_params": str(best_params),
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    **test_metrics,
                }
            )

        if not window_results:
            raise ValueError("No rolling windows were generated.")

        results_df = pd.DataFrame(window_results)
        results_df.to_csv(experiment_dir / "window_metrics.csv", index=False)

        metric_cols = [
            c
            for c in results_df.columns
            if c
            not in [
                "window",
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "test_start",
                "test_end",
                "best_params",
            ]
            and not c.startswith("val_")
        ]
        avg_metrics = {col: float(results_df[col].mean()) for col in metric_cols}

        metadata: dict[str, Any] = {
            "num_windows": len(window_results),
            "mode": mode,
            "tune_hyperparameters": tune_hyperparameters,
        }
        if all_grid_trials:
            all_trials_df = pd.concat(all_grid_trials, ignore_index=True)
            all_trials_df = all_trials_df.sort_values(["window", "val_mae"], ascending=True).reset_index(drop=True)
            grid_results_path = experiment_dir / "grid_search_results.csv"
            all_trials_df.to_csv(grid_results_path, index=False)
            metadata["grid_search_results_path"] = str(grid_results_path)
            metadata["num_grid_candidates"] = int(len(all_trials_df))

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=avg_metrics,
            segment_test_metrics={"rolling": avg_metrics},
            metadata=metadata,
        )
