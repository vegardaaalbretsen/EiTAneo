"""Location-specific LightGBM strategy: one model per location."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.base import BaseExperiment, ExperimentResult, RunModes
from helpers.data_retrieval import chronological_split, rolling_window
from experiments.config import (
    BASE_FEATURES,
    DEFAULT_LOCATION_NAMES,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    TARGET_COLUMN,
    DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
    DEFAULT_VAL_SIZE_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE_ROLLING_WINDOW,
    DEFAULT_STEP_SIZE_ROLLING_WINDOW,
)
from experiments.metrics import weighted_overall, with_sample_count
from experiments.models._lightgbm_utils import (
    evaluate_lightgbm,
    lightgbm_prediction_frame,
    save_lightgbm_prediction_artifacts,
    train_lightgbm_regressor,
)


class LightGBMLocationSpecificExperiment(BaseExperiment):
    """Train one LightGBM model per location_id."""

    name = "lightgbm_location_specific"

    def run(self, df: pd.DataFrame, mode: RunModes, output_dir: Path) -> ExperimentResult:

        if mode == RunModes.CHRONOLOGICAL:
            return self._run_chronological(df, output_dir)
        if mode == RunModes.SLIDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="sliding_window")
        if mode == RunModes.EXPANDING_WINDOW:
            return self._run_rolling_window(df, output_dir, mode="expanding_window")
        raise ValueError(f"Unsupported mode '{mode}' for Experiment. Use 'chronological', 'sliding_window', or 'expanding_window'.")


    def _run_chronological(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        segment_test_metrics: dict[str, dict[str, float]] = {}
        segment_metadata: dict[str, dict[str, object]] = {}

        for location_id in sorted(df["location_id"].unique()):
            location_df = df.loc[df["location_id"] == location_id].copy()
            if location_df.empty:
                continue

            train_df, val_df, test_df = chronological_split(
                location_df,
                train_ratio=DEFAULT_TRAIN_RATIO,
                val_ratio=DEFAULT_VAL_RATIO,
            )

            model = train_lightgbm_regressor(
                train_df=train_df,
                val_df=val_df,
                feature_cols=BASE_FEATURES,
                target_col=TARGET_COLUMN,
            )

            train_metrics = evaluate_lightgbm(model, train_df, BASE_FEATURES, TARGET_COLUMN)
            val_metrics = evaluate_lightgbm(model, val_df, BASE_FEATURES, TARGET_COLUMN)
            test_metrics = evaluate_lightgbm(model, test_df, BASE_FEATURES, TARGET_COLUMN)
            test_with_n = with_sample_count(test_metrics, len(test_df))

            location_name = DEFAULT_LOCATION_NAMES.get(int(location_id), f"location_{location_id}")
            segment_key = f"{int(location_id)}_{location_name.lower()}"
            segment_test_metrics[segment_key] = test_with_n

            model_path = experiment_dir / f"{segment_key}.txt"
            model.save_model(str(model_path))

            prediction_paths = save_lightgbm_prediction_artifacts(
                predictions_df=lightgbm_prediction_frame(
                    model,
                    test_df,
                    BASE_FEATURES,
                    TARGET_COLUMN,
                ),
                output_dir=experiment_dir,
                filename_stem=f"{segment_key}_test_predictions",
                title=f"LightGBM Location-Specific: {location_name} Test Set",
            )

            segment_metadata[segment_key] = {
                "location_id": int(location_id),
                "location_name": location_name,
                "feature_columns": list(BASE_FEATURES),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                **prediction_paths,
            }

        if not segment_test_metrics:
            raise ValueError("No location-specific models were trained.")

        overall_test_metrics = weighted_overall(segment_test_metrics)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=overall_test_metrics,
            segment_test_metrics=segment_test_metrics,
            metadata={"locations": segment_metadata},
        )
    
    def _run_rolling_window(self, df: pd.DataFrame, output_dir: Path, mode: str) -> ExperimentResult:
        experiment_dir = output_dir / self.name / mode
        experiment_dir.mkdir(parents=True, exist_ok=True)

        segment_test_metrics: dict[str, dict[str, float]] = {}
        segment_metadata: dict[str, dict[str, object]] = {}

        for location_id in sorted(df["location_id"].unique()):

            location_df = (
                df.loc[df["location_id"] == location_id]
                .sort_index()
                .copy()
            )

            if location_df.empty:
                continue

            window_results = []
            prediction_frames: list[pd.DataFrame] = []

            for i, (train_df, val_df, test_df) in enumerate(
                rolling_window(
                    location_df,
                    train_size=DEFAULT_TRAIN_SIZE_ROLLING_WINDOW,
                    val_size=DEFAULT_VAL_SIZE_ROLLING_WINDOW,
                    test_size=DEFAULT_TEST_SIZE_ROLLING_WINDOW,
                    step_size=DEFAULT_STEP_SIZE_ROLLING_WINDOW,
                    expanding=(mode == "expanding_window"),
                )
            ):
                model = train_lightgbm_regressor(
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=BASE_FEATURES,
                    target_col=TARGET_COLUMN,
                )

                test_metrics = evaluate_lightgbm(
                    model,
                    test_df,
                    BASE_FEATURES,
                    TARGET_COLUMN,
                )
                prediction_df = lightgbm_prediction_frame(
                    model,
                    test_df,
                    BASE_FEATURES,
                    TARGET_COLUMN,
                )
                prediction_df.insert(0, "window", i)
                prediction_frames.append(prediction_df)

                window_results.append({
                    "window": i,
                    "train_start": train_df.index.min(),
                    "train_end": train_df.index.max(),
                    "test_start": test_df.index.min(),
                    "test_end": test_df.index.max(),
                    "n_samples": len(test_df),
                    **test_metrics,
                })

            if not window_results:
                continue

            results_df = pd.DataFrame(window_results)

            location_name = DEFAULT_LOCATION_NAMES.get(
                int(location_id),
                f"location_{location_id}",
            )
            segment_key = f"{int(location_id)}_{location_name.lower()}"

            # Save window metrics
            results_df.to_csv(
                experiment_dir / f"{segment_key}_window_metrics.csv",
                index=False,
            )

            metric_cols = [
                c for c in results_df.columns
                if c not in ["window", "train_start", "train_end", "test_start", "test_end", "n_samples"]
            ]

            # weighted average across windows
            total_samples = results_df["n_samples"].sum()

            weighted_metrics = {}
            for col in metric_cols:
                weighted_metrics[col] = (
                    (results_df[col] * results_df["n_samples"]).sum()
                    / total_samples
                )

            weighted_metrics["n_samples"] = int(total_samples)

            segment_test_metrics[segment_key] = weighted_metrics

            prediction_metadata = {}
            if prediction_frames:
                prediction_metadata = save_lightgbm_prediction_artifacts(
                    predictions_df=pd.concat(prediction_frames),
                    output_dir=experiment_dir,
                    filename_stem=f"{segment_key}_test_predictions",
                    title=(
                        f"LightGBM Location-Specific: {location_name} "
                        f"({mode.replace('_', ' ').title()})"
                    ),
                )

            segment_metadata[segment_key] = {
                "location_id": int(location_id),
                "location_name": location_name,
                "num_windows": len(results_df),
                "mode": mode,
                **prediction_metadata,
            }

        if not segment_test_metrics:
            raise ValueError("No location-specific sliding windows were generated.")

        overall_test_metrics = weighted_overall(segment_test_metrics)

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=overall_test_metrics,
            segment_test_metrics=segment_test_metrics,
            metadata={
                "locations": segment_metadata,
                "mode": mode,
            },
        )
