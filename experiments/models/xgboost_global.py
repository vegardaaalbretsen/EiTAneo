"""Single XGBoost model trained on all locations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import (
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    GLOBAL_FEATURES,
    TARGET_COLUMN,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_EARLY_STOPPING_ROUNDS,
)
from experiments.metrics import with_sample_count
from helpers.data_retrieval import chronological_split, split_features_target


class XGBoostGlobalExperiment(BaseExperiment):
    """Global XGBoost model with location_id as a feature."""

    name = "xgboost_global"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        model = self._train_xgboost_regressor(
            train_df=train_df,
            val_df=val_df,
            feature_cols=GLOBAL_FEATURES,
            target_col=TARGET_COLUMN,
        )

        train_metrics = self._evaluate_xgboost(model, train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        val_metrics = self._evaluate_xgboost(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_metrics = self._evaluate_xgboost(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.xgb"
        # xgboost Booster has save_model
        model.save_model(str(model_path))

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata={
                "feature_columns": list(GLOBAL_FEATURES),
                "model_path": str(model_path),
                "best_iteration": int(getattr(model, "best_iteration", -1)),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )

    def _train_xgboost_regressor(
        self,
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

    def _evaluate_xgboost(self, model, split_df: pd.DataFrame, feature_cols, target_col: str = TARGET_COLUMN) -> dict[str, float]:
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
