from pathlib import Path
import pickle
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import (
    DEFAULT_STEP_SIZE_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE_ROLLING_WINDOW,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
    DEFAULT_VAL_RATIO,
    DEFAULT_VAL_SIZE_ROLLING_WINDOW,
    GLOBAL_FEATURES,
    TARGET_COLUMN,
)
from experiments.metrics import regression_metrics, with_sample_count
from helpers.data_retrieval import chronological_split, rolling_window, split_features_target


class RandomForestExperiment(BaseExperiment):
    name = "random_forest"

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:
        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(df, output_dir)
        if mode == RunModes.SLIDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="sliding_window", tune_hyperparameters=False)
        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="expanding_window", tune_hyperparameters=False)
        raise ValueError(
            f"Unsupported mode '{mode}' for Experiment. Use 'chronological', 'sliding_window', or 'expanding_window'."
        )

    def _get_param_grid(self) -> list[dict]:
        grid = {
            "n_estimators": [100, 300],
            "max_depth": [10, None],
            "min_samples_split": [2, 10],
        }

        keys = list(grid.keys())
        values = list(grid.values())

        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

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
        best_score = float("inf")  # lavest MAE/RMSE er best

        for params in self._get_param_grid():
            model = RandomForestRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            val_predictions = model.predict(X_val)
            val_metrics = regression_metrics(y_val, val_predictions)

            # Velg metrikk du ønsker å tune på
            score = val_metrics["mae"]

            if score < best_score:
                best_score = score
                best_model = model
                best_params = params
                best_val_metrics = val_metrics

        return best_model, best_params, best_val_metrics

    def _run_chronological(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

        model, best_params, val_metrics = self._select_best_model(X_train, y_train, X_val, y_val)

        train_metrics = regression_metrics(y_train, model.predict(X_train))
        test_metrics = regression_metrics(y_test, model.predict(X_test))
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name
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
                "best_params": best_params,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
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
                model, best_params, val_metrics = self._select_best_model(X_train, y_train, X_val, y_val)
            else:
                model = RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                best_params = {
                    "n_estimators": 300,
                }
                val_metrics = regression_metrics(y_val, model.predict(X_val))

            test_metrics = regression_metrics(y_test, model.predict(X_test))

            window_results.append({
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
            })

        if not window_results:
            raise ValueError("No rolling windows were generated.")

        results_df = pd.DataFrame(window_results)

        metric_cols = [
            c for c in results_df.columns
            if c not in [
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