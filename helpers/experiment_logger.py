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
        path = self.base_dir / experimen