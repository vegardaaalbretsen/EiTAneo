from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.base import ExperimentResult


class ExperimentLogger:
    """Handles persistence of experiment results and comparisons."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Per-experiment logging
    # -------------------------

    def experiment_dir(self, experiment_name: str) -> Path:
        path = self.base_dir / experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_result(self, result: ExperimentResult) -> Path:
        """Save full experiment result as JSON."""
        experiment_dir = self.experiment_dir(result.experiment_name)

        result_path = experiment_dir / "result.json"

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result_path

    # -------------------------
    # Comparison logging
    # -------------------------

    def save_comparison(self, summary_df: pd.DataFrame) -> tuple[Path, Path]:
        """Save comparison outputs."""
        csv_path = self.base_dir / "comparison.csv"
        json_path = self.base_dir / "comparison.json"

        summary_df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2)

        return csv_path, json_path

    # -------------------------
    # Helper
    # -------------------------

    @staticmethod
    def summary_row(result: ExperimentResult) -> dict[str, float | str]:
        row: dict[str, float | str] = {"experiment": result.experiment_name}
        row.update(result.overall_test_metrics)
        return row