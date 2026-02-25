"""Runner logic for executing experiments and writing comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.base import ExperimentResult
from experiments.registry import build_experiments
from helpers.data_retrieval import load_preprocessed_data


def _summary_row(result: ExperimentResult) -> dict[str, float | str]:
    row: dict[str, float | str] = {"experiment": result.experiment_name}
    row.update(result.overall_test_metrics)
    return row


def run_experiments(
    experiment_names: Iterable[str],
    data_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[ExperimentResult], pd.DataFrame, Path, Path]:
    """Run selected experiments and persist comparison outputs."""
    df = load_preprocessed_data(data_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    for experiment in build_experiments(experiment_names):
        result = experiment.run(df=df, output_dir=output_path)
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

    return results, summary_df, csv_path, json_path
