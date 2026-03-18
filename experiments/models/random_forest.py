from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import (
    DEFAULT_RANDOM_FOREST_PARAMS,
    DEFAULT_STEP_SIZE_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE_ROLLING_WINDOW,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SIZE_ROLLING_WINDOW,
    GLOBAL_FEATURES,
    RANDOM_FOREST_PARAM_GRID,
    TARGET_COLUMN,
)
from experiments.metrics import regression_metrics, with_sample_count
from helpers.data_retrieval import chronological_split, rolling_window, split_features_target


class RandomForestExperiment(BaseExperiment):
    name = "random_forest"

    def _use_grid_search(self) -> bool:
        return bool(self.options.get("random_forest_grid_search", False))

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:
        tune_hyperparameters = self._use_grid_search()
        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(
                df,
                output_dir,
                tune_hyperparameters=tune_hyperparameters,
            )
        if mode == RunModes.SLIDING_WINDOW:
            return self._run_rolling_window(
                df,
                output_dir,
                mode="sliding_window",
                tune_hyperparameters=tune_hyperparameters,
            )
        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_rolling_window(
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
        keys = list(RANDOM_FOREST_PARAM_GRID.keys())
        values = list(RANDOM_FOREST_PARAM_GRID.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _fit_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]):
        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    def _select_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        best_model = None
        best_params = None
        best_val_metrics = None
        best_score = float("inf")
        search_rows: list[dict[str, Any]] = []

        for trial_idx, params in enumerate(self._get_param_grid(), start=1):
            model = self._fit_model(X_train, y_train, params)
            val_metrics = regression_metrics(y_val, model.predict(X_val))
            score = val_metrics["mae"]

            search_row = {
                "trial": trial_idx,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_mape": val_metrics["mape"],
            }
            search_row.update({f"param_{key}": value for key, value in params.items()})
            search_rows.append(search_row)

            if score < best_score:
                best_score = score
                best_model = model
                best_params = params
                best_val_metrics = val_metrics

        if best_model is None or best_params is None or best_val_metrics is None:
            raise RuntimeError("Random forest grid search failed to produce a valid model.")

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

        X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

        search_results_path: Path | None = None
        num_grid_candidates: int | None = None

        if tune_hyperparameters:
            model, best_params, val_metrics, search_df = self._select_best_model(
                X_train,
                y_train,
                X_val,
                y_val,
            )
        else:
            best_params = dict(DEFAULT_RANDOM_FOREST_PARAMS)
            model = self._fit_model(X_train, y_train, best_params)
            val_metrics = regression_metrics(y_val, model.predict(X_val))
            search_df = None

        train_metrics = regression_metrics(y_train, model.predict(X_train))
        test_metrics = regression_metrics(y_test, model.predict(X_test))
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        model_path = experiment_dir / "model.pkl"
        with open(model_path, "wb") as file_obj:
            pickle.dump(model, file_obj)

        if search_df is not None:
            search_results_path = experiment_dir / "grid_search_results.csv"
            search_df.to_csv(search_results_path, index=False)
            num_grid_candidates = int(len(search_df))

        metadata: dict[str, Any] = {
            "feature_columns": list(GLOBAL_FEATURES),
            "model_path": str(model_path),
            "best_params": best_params,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "tune_hyperparameters": tune_hyperparameters,
        }
        if search_results_path is not None and num_grid_candidates is not None:
            metadata["grid_search_results_path"] = str(search_results_path)
            metadata["num_grid_candidates"] = num_grid_candidates

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata=metadata,
        )

    def _run_rolling_window(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        mode: str,
        tune_hyperparameters: bool = True,
    ) -> ExperimentResult:
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
            X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
            X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
            X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

            if tune_hyperparameters:
                model, best_params, val_metrics, _ = self._select_best_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
            else:
                best_params = dict(DEFAULT_RANDOM_FOREST_PARAMS)
                model = self._fit_model(X_train, y_train, best_params)
                val_metrics = regression_metrics(y_val, model.predict(X_val))

            test_metrics = regression_metrics(y_test, model.predict(X_test))

            window_results.append(
                {
                    "window": i,
                    "train_start": train_df.index.min(),
                    "train_end": train_df.index.max(),
                    "val_start": val_df.index.min(),
                    "val_end": val_df.index.max(),
                    "test_start": test_df.index.min(),
                    "test_end": test_df.index.max(),
                    "n_samples": len(test_df),
                    "best_params": str(best_params),
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    **test_metrics,
                }
            )

        if not window_results:
            raise ValueError("No rolling windows were generated.")

        results_df = pd.DataFrame(window_results)

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
                "n_samples",
                "best_params",
            ]
            and not c.startswith("val_")
        ]

        total_samples = results_df["n_samples"].sum()
        weighted_metrics = {}
        for col in metric_cols:
            weighted_metrics[col] = (results_df[col] * results_df["n_samples"]).sum() / total_samples
        weighted_metrics["n_samples"] = int(total_samples)

        results_df.to_csv(experiment_dir / "window_metrics.csv", index=False)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=weighted_metrics,
            segment_test_metrics={"rolling": weighted_metrics},
            metadata={
                "num_windows": len(results_df),
                "mode": mode,
                "tune_hyperparameters": tune_hyperparameters,
                "feature_columns": list(GLOBAL_FEATURES),
            },
        )
