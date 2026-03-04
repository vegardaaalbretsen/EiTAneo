"""Single SVR model trained on all locations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, GLOBAL_FEATURES, TARGET_COLUMN
from experiments.metrics import with_sample_count
from helpers.data_retrieval import chronological_split, split_features_target


class SVRGlobalExperiment(BaseExperiment):
    """Global Support Vector Regression model with location_id as a feature."""

    name = "svr_global"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        model = self._train_svr(
            train_df=train_df,
            val_df=val_df,
            feature_cols=GLOBAL_FEATURES,
            target_col=TARGET_COLUMN,
        )

        train_metrics = self._evaluate(model, train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        val_metrics = self._evaluate(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_metrics = self._evaluate(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.joblib"

        # use joblib to persist sklearn pipeline
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - informative error
            raise RuntimeError("joblib is required to save the SVR model. Install with `pip install joblib`.") from exc

        joblib.dump(model, str(model_path))

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata={
                "feature_columns": list(GLOBAL_FEATURES),
                "model_path": str(model_path),
                "best_iteration": -1,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )

    def _train_svr(self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols, target_col: str = TARGET_COLUMN, params: dict | None = None):
        """Train a scikit-learn SVR inside a small pipeline that scales features."""
        try:
            from sklearn.svm import SVR
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - informative error
            raise RuntimeError("scikit-learn is required for SVR experiments. Install with `pip install scikit-learn`.") from exc

        resolved = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}
        if params:
            resolved.update(params)

        X_train, y_train = split_features_target(train_df, feature_cols, target_col)

        pipeline = Pipeline([("scaler", StandardScaler()), ("svr", SVR(**resolved))])
        pipeline.fit(X_train, y_train)
        return pipeline

    def _evaluate(self, model, split_df: pd.DataFrame, feature_cols, target_col: str = TARGET_COLUMN) -> dict[str, float]:
        X_split, y_split = split_features_target(split_df, feature_cols, target_col)
        preds = model.predict(X_split)

        from experiments.metrics import regression_metrics

        return regression_metrics(y_split, preds)
