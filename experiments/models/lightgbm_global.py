"""Single LightGBM model trained on all locations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from experiments.config import DEFAULT_TEST_SIZE_ROLLING_WINDOW, DEFAULT_TRAIN_RATIO, DEFAULT_TRAIN_SIZE_ROLLING_WINDOW, DEFAULT_VAL_RATIO, DEFAULT_VAL_SIZE_ROLLING_WINDOW, GLOBAL_FEATURES, TARGET_COLUMN, DEFAULT_STEP_SIZE_ROLLING_WINDOW
from experiments.metrics import with_sample_count
from experiments.models._lightgbm_utils import evaluate_lightgbm, train_lightgbm_regressor
from helpers.data_retrieval import chronological_split, rolling_window
from helpers.plotter import Plotter


class LightGBMGlobalExperiment(BaseExperiment):
    """Global LightGBM model with location_id as a feature."""

    name = "lightgbm_global"

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:

        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(df, output_dir)
        
        if mode == RunModes.SLIDING_WINDOW:
            return self._run_sliding_window(df, output_dir, mode="sliding_window")
        
        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_sliding_window(df, output_dir, mode="expanding_window")
        
        raise ValueError(f"Unsupported mode '{mode}' for Experiment. Use 'chronological', 'sliding_window', or 'expanding_window'.")

        
    
    def _run_chronological(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
        )

        model = train_lightgbm_regressor(
            train_df=train_df,
            val_df=val_df,
            feature_cols=GLOBAL_FEATURES,
            target_col=TARGET_COLUMN,
        )

        train_metrics = evaluate_lightgbm(model, train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        val_metrics = evaluate_lightgbm(model, val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_metrics = evaluate_lightgbm(model, test_df, GLOBAL_FEATURES, TARGET_COLUMN)
        test_with_n = with_sample_count(test_metrics, len(test_df))

        experiment_dir = output_dir / self.name / "chronological"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_path = experiment_dir / "model.txt"
        model.save_model(str(model_path))

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_with_n,
            segment_test_metrics={"all": test_with_n},
            metadata={
                "feature_columns": list(GLOBAL_FEATURES),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )
    
    def _run_sliding_window(self, df: pd.DataFrame, output_dir: Path, mode: str) -> ExperimentResult:
        window_results = []
        experiment_dir = output_dir / self.name / mode
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for i, (train_df, val_df, test_df) in enumerate(
            rolling_window(
                df,
                train_size=DEFAULT_TRAIN_SIZE_ROLLING_WINDOW, # 5 citites, 24 hours, 20 days
                val_size=DEFAULT_VAL_SIZE_ROLLING_WINDOW,    # 5 cities, 24 hours, 3 days
                test_size=DEFAULT_TEST_SIZE_ROLLING_WINDOW,   # 5 cities, 24 hours, 1 day
                step_size=DEFAULT_STEP_SIZE_ROLLING_WINDOW, # 5 cities, 24 hours, 1 day step
                expanding=(mode == "expanding_window"),
            )
        ):
            model = train_lightgbm_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=GLOBAL_FEATURES,
                target_col=TARGET_COLUMN,
            )

            test_metrics = evaluate_lightgbm(
                model,
                test_df,
                GLOBAL_FEATURES,
                TARGET_COLUMN,
            )

            window_results.append({
                "window": i,
                "train_start": train_df.index.min(),
                "train_end": train_df.index.max(),
                "test_start": test_df.index.min(),
                "test_end": test_df.index.max(),
                **test_metrics,
            })
        results_df = pd.DataFrame(window_results)
        results_df.to_csv(experiment_dir / "window_metrics.csv", index=False)

        # Average metrics across windows
        avg_metrics = results_df.drop(columns=["window", "train_start", "train_end", "test_start", "test_end"]).mean().to_dict()

        Plotter.create_rolling_plots(df, results_df, mode=RunModes(mode), experiment_dir=experiment_dir)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=avg_metrics,
            segment_test_metrics={"rolling": avg_metrics},
            metadata={
                "num_windows": len(window_results),
                "mode": mode,
            },
        )
    
    
