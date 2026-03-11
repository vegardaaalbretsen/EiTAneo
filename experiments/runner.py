"""Runner logic for executing experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.base import ExperimentResult, RunModes
from experiments.registry import build_experiments
from helpers.data_retrieval import load_preprocessed_data
from helpers.experiment_logger import ExperimentLogger


def run_experiments(
    experiment_names: Iterable[str],
    data_path: str | Path,
    mode: RunModes,
    output_dir: str | Path,
) -> tuple[list[ExperimentResult], pd.DataFrame, Path, Path]:

    df = load_preprocessed_data(data_path)

    logger = ExperimentLogger(Path(output_dir))

    results: list[ExperimentResult] = []

    for experiment in build_experiments(experiment_names):

        result = experiment.run(
            df=df,
            mode=mode,
            output_dir=logger.base_dir,
        )

        results.append(result)

        logger.save_result(result)

    summary_df = pd.DataFrame(
        ExperimentLogger.summary_row(r) for r in results
    )

    if not summary_df.empty and "mae" in summary_df.columns:
        summary_df = summary_df.sort_values("mae").reset_index(drop=True)

    csv_path, json_path = logger.save_comparison(summary_df)

    return results, summary_df, csv_path, json_path