
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import (
    TARGET_COLUMN,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_EARLY_STOPPING_ROUNDS,
)
from experiments.metrics import with_sample_count
from helpers.data_retrieval import chronological_split, split_features_target


def train_xgboost_regressor(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols,
        target_col: str = TARGET_COLUMN,
        params: dict | None = None,
        num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose_eval: int = 0,
    ):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is required for this experiment. Install with `pip install xgboost`."
            ) from exc

        # sensible defaults for XGBoost; user params override
        resolved = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        if params:
            resolved.update(params)

        X_train, y_train = split_features_target(train_df, feature_cols, target_col)
        X_val, y_val = split_features_target(val_df, feature_cols, target_col)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_cols))
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(feature_cols))

        evals = [(dtrain, "train"), (dval, "valid")]

        model = xgb.train(
            resolved,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        return model

def evaluate_xgboost(model, split_df: pd.DataFrame, feature_cols, target_col: str = TARGET_COLUMN) -> dict[str, float]:
        try:
            import xgboost as xgb  # noqa: F401 - keep for type expectations
        except Exception:
            pass

        X_split, y_split = split_features_target(split_df, feature_cols, target_col)
        dsplit = xgb.DMatrix(X_split, feature_names=list(feature_cols))
        best_it = getattr(model, "best_iteration", None)
        if best_it is not None and best_it >= 0:
            # prediction with best_iteration (iteration_range end is exclusive, so add 1)
            try:
                preds = model.predict(dsplit, iteration_range=(0, best_it + 1))
            except TypeError:
                # fallback to older ntree_limit parameter
                preds = model.predict(dsplit, ntree_limit=best_it)
        else:
            preds = model.predict(dsplit)

        from experiments.metrics import regression_metrics

        return regression_metrics(y_split, preds)