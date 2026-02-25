"""Experiment registry and factory functions."""

from __future__ import annotations

from collections.abc import Iterable

from experiments.base import BaseExperiment
from experiments.models.lightgbm_global import LightGBMGlobalExperiment
from experiments.models.lightgbm_location_specific import LightGBMLocationSpecificExperiment
from experiments.models.lightgbm_two_group import LightGBMTwoGroupExperiment
from experiments.models.linear_regression_global import LinearRegressionGlobalExperiment
from experiments.models.random_forest import RandomForestExperiment


EXPERIMENT_REGISTRY: dict[str, type[BaseExperiment]] = {
    LightGBMGlobalExperiment.name: LightGBMGlobalExperiment,
    LightGBMTwoGroupExperiment.name: LightGBMTwoGroupExperiment,
    LightGBMLocationSpecificExperiment.name: LightGBMLocationSpecificExperiment,
    LinearRegressionGlobalExperiment.name: LinearRegressionGlobalExperiment,
    RandomForestExperiment.name: RandomForestExperiment
}


def available_experiments() -> list[str]:
    """Return sorted experiment names available to the runner."""
    return sorted(EXPERIMENT_REGISTRY)


def build_experiments(experiment_names: Iterable[str]) -> list[BaseExperiment]:
    """Instantiate experiment classes from user-selected names."""
    selected = list(dict.fromkeys(experiment_names))
    unknown = [name for name in selected if name not in EXPERIMENT_REGISTRY]
    if unknown:
        available = ", ".join(available_experiments())
        unknown_names = ", ".join(unknown)
        raise ValueError(f"Unknown experiments: {unknown_names}. Available: {available}")

    return [EXPERIMENT_REGISTRY[name]() for name in selected]
