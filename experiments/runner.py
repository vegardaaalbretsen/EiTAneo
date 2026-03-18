"""Runner logic for executing experiments and writing comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.base import ExperimentResult, RunModes
from experiments.registry import build_experiments
from helpers.data_retrieval import load_preprocessed_data


def _summary_row(result: ExperimentResult) -> dict[str, float | str]:
    row: dict[str, float | str] = {"experiment": result.experiment_name}
    row.update(result.overall_test_metrics)
    return row


def _grid_search_rows(results: list[ExperimentResult]) -> pd.DataFrame:
    trial_dfs: list[pd.DataFrame] = []

    for result in results:
        grid_search_path = result.metadata.get("grid_search_results_path")
        if not isinstance(grid_search_path, str):
            continue

        csv_path = Path(grid_search_path)
        if not csv_path.exists():
            continue

        trial_df = pd.read_csv(csv_path)
        if trial_df.empty:
            continue

        trial_df.insert(0, "experiment", result.experiment_name)
        trial_dfs.append(trial_df)

    if not trial_dfs:
        return pd.DataFrame()

    combined = pd.concat(trial_dfs, ignore_index=True)
    sort_cols = [col for col in ("experiment", "val_mae") if col in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    return combined


def run_experiments(
    experiment_names: Iterable[str],
    data_path: str | Path,
    mode: RunModes,
    output_dir: str | Path,
    experiment_options: dict[str, object] | None = None,
) -> tuple[list[ExperimentResult], pd.DataFrame, Path, Path, pd.DataFrame, Path | None]:
    """Run selected experiments and persist comparison outputs."""
    df = load_preprocessed_data(data_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    for experiment in build_experiments(experiment_names, options=experiment_options):
        result = experiment.run(df=df, mode=mode, output_dir=output_path)
        results.append(result)

        experiment_dir = output_path / result.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "result.json", "w", encoding="utf-8") as file_obj:
            json.dump(result.to_dict(), file_obj, indent=2)

    summary_df = pd.DataFrame(_summary_row(result) for result in results)
    if not summary_df.empty and "mae" in summary_df.columns:
        summary_df = summary_df.sort_values("mae", ascending=True).reset_index(drop=True)

    csv_path = output_path / "comparison.csv"
    json_path = output_path / "comparison.json"
    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary_df.to_dict(orient="records"), file_obj, indent=2)

    grid_trials_df = _grid_search_rows(results)
    grid_trials_csv: Path | None = None
    if not grid_trials_df.empty:
        grid_trials_csv = output_path / "grid_search_trials.csv"
        grid_trials_df.to_csv(grid_trials_csv, index=False)

    return results, summary_df, csv_path, json_path, grid_trials_df, grid_trials_csv
