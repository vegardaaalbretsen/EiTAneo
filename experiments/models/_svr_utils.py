from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.config import TARGET_COLUMN
from experiments.metrics import with_sample_count
from helpers.data_retrieval import split_features_target

def train_svr(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols, target_col: str = TARGET_COLUMN, params: dict | None = None):
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

def evaluate(model, split_df: pd.DataFrame, feature_cols, target_col: str = TARGET_COLUMN) -> dict[str, float]:
        X_split, y_split = split_features_target(split_df, feature_cols, target_col)
        preds = model.predict(X_split)

        from experiments.metrics import regression_metrics

        return regression_metrics(y_split, preds)