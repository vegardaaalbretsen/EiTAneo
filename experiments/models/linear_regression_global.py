"""Global linear regression baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import DEFAULT_STEP_SIZE_ROLLING_WINDOW, DEFAULT_TEST_SIZE_ROLLING_WINDOW, DEFAULT_TRAIN_RATIO, DEFAULT_TRAIN_SIZE_ROLLING_WINDOW, DEFAULT_VAL_RATIO, DEFAULT_VAL_SIZE_ROLLING_WINDOW, GLOBAL_FEATURES, TARGET_COLUMN
from experiments.metrics import regression_metrics, with_sample_count
from helpers.data_retrieval import chronological_split, rolling_window, split_features_target


class LinearRegressionGlobalExperiment(BaseExperiment):
    """Single global linear regression using all locations."""

    name = "linear_regression_global"

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

        X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

        model = LinearRegression()
        model.fit(X_train, y_train)

        train_metrics = regression_metrics(y_train, model.predict(X_train))
        val_metrics = regression_metrics(y_val, model.predict(X_val))
        test_metrics = regression_metrics(y_test, model.predict(X_test))
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name / "chronological"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.pkl"
        with open(model_path, "wb") as file_obj:
            pickle.dump(model, file_obj)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata={
                "feature_columns": list(GLOBAL_FEATURES),
                "model_path": str(model_path),
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
            X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
            X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
            X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

            model = LinearRegression()
            model.fit(X_train, y_train)

            test_metrics = regression_metrics(y_test, model.predict(X_test))

            # include n_samples per window for weighted aggregation
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
