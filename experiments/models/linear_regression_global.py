"""Global linear regression baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, GLOBAL_FEATURES, TARGET_COLUMN
from experiments.metrics import regression_metrics, with_sample_count
from helpers.data_retrieval import chronological_split, split_features_target


class LinearRegressionGlobalExperiment(BaseExperiment):
    """Single global linear regression using all locations."""

    name = "linear_regression_global"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
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
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )
