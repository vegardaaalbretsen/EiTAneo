from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, GLOBAL_FEATURES, TARGET_COLUMN
from experiments.metrics import regression_metrics, with_sample_count
from helpers.data_retrieval import chronological_split, split_features_target


class RandomForestExperiment(BaseExperiment):
    name = "random_forest"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df, train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO
        )

        X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        train_metrics = regression_metrics(y_train, model.predict(X_train))
        val_metrics = regression_metrics(y_val, model.predict(X_val))
        test_metrics = with_sample_count(regression_metrics(y_test, model.predict(X_test)), len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Save model artifact here if needed

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_metrics,
            segment_test_metrics={"all": test_metrics},
            metadata={"train_metrics": train_metrics, "val_metrics": val_metrics},
        )