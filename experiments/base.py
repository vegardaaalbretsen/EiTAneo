"""Base interfaces and data contracts for experiment execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ExperimentResult:
    """Container for experiment outputs used in reporting."""

    experiment_name: str
    overall_test_metrics: dict[str, float]
    segment_test_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseExperiment(ABC):
    """Base class each model experiment should implement."""

    name: str

    @abstractmethod
    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        """Train/evaluate a model strategy and return comparison-ready metrics."""
