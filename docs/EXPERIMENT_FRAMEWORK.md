# Experiment Framework Guide

This project uses a small experiment framework so multiple model strategies can be trained and compared the same way.

## 1. High-level flow

1. Run `python run_experiments.py ...`
2. CLI resolves selected experiment names (`--experiments` or `all`).
3. `experiments/runner.py` loads dataset once via `helpers/data_retrieval.py`.
4. Runner builds experiment objects from `experiments/registry.py`.
5. Each experiment executes `run(df, output_dir)` and returns an `ExperimentResult`.
6. Runner writes:
- per-experiment `result.json`
- global `comparison.csv` and `comparison.json` (sorted by MAE)

## 2. Main files

- `run_experiments.py`: CLI entrypoint
- `experiments/registry.py`: experiment name to class mapping
- `experiments/runner.py`: execution loop and report writing
- `experiments/base.py`: `BaseExperiment` and `ExperimentResult`
- `experiments/metrics.py`: shared MAE/RMSE/R2/MAPE calculation
- `helpers/data_retrieval.py`: data loading + validation + chronological splitting
- `experiments/config.py`: shared constants (features, split ratios, defaults)

## 3. Current experiment names

- `linear_regression_global`
- `lightgbm_global`
- `lightgbm_two_group`
- `lightgbm_location_specific`

List from CLI:

```bash
python run_experiments.py --list
```

## 4. Data contract

`helpers/data_retrieval.py` validates these required columns:

- `month_sin`
- `month_cos`
- `hour_sin`
- `hour_cos`
- `location_id`
- `temperature`
- `consumption`

Default path is `data/preprocessed_data.csv` with fallback to legacy `preprocessed_data.csv`.

## 5. Output layout

Default output directory: `results/experiments/`

- `results/experiments/comparison.csv`
- `results/experiments/comparison.json`
- `results/experiments/<experiment_name>/result.json`
- `results/experiments/<experiment_name>/...` model artifacts saved by each experiment

## 6. Add a new model experiment

### Step A: create a new experiment file

Add a new file under `experiments/models/`, for example:

- `experiments/models/random_forest_global.py`

### Step B: implement `BaseExperiment`

Your class must:

1. Set a unique `name` string (this is what CLI uses).
2. Implement `run(self, df, output_dir) -> ExperimentResult`.
3. Save artifacts inside `output_dir / self.name`.
4. Return `ExperimentResult` with:
- `overall_test_metrics` for comparison table
- optional `segment_test_metrics` for group/location breakdown
- optional `metadata` for diagnostics

Minimal example:

```python
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from experiments.base import BaseExperiment, ExperimentResult
from experiments.config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, GLOBAL_FEATURES, TARGET_COLUMN
from experiments.metrics import regression_metrics, with_sample_count
from helpers.data_retrieval import chronological_split, split_features_target


class RandomForestGlobalExperiment(BaseExperiment):
    name = "random_forest_global"

    def run(self, df: pd.DataFrame, output_dir: Path) -> ExperimentResult:
        train_df, val_df, test_df = chronological_split(
            df, train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO
        )

        X_train, y_train = split_features_target(train_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_val, y_val = split_features_target(val_df, GLOBAL_FEATURES, TARGET_COLUMN)
        X_test, y_test = split_features_target(test_df, GLOBAL_FEATURES, TARGET_COLUMN)

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        train_metrics = regression_metrics(y_train, model.predict(X_train))
        val_metrics = regression_metrics(y_val, model.predict(X_val))
        test_metrics = with_sample_count(regression_metrics(y_test, model.predict(X_test)), len(test_df))

        experiment_dir = output_dir / self.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Save model artifact here if needed

        return ExperimentResult(
            experiment_name=self.name,
            overall_test_metrics=test_metrics,
            segment_test_metrics={"all": test_metrics},
            metadata={"train_metrics": train_metrics, "val_metrics": val_metrics},
        )
```

### Step C: register it

Update `experiments/registry.py`:

1. Add import:
- `from experiments.models.random_forest_global import RandomForestGlobalExperiment`
2. Add entry to `EXPERIMENT_REGISTRY`:
- `"random_forest_global": RandomForestGlobalExperiment` (or using `.name`)

### Step D: run it

```bash
python run_experiments.py --experiments random_forest_global
```

or with others:

```bash
python run_experiments.py --experiments random_forest_global lightgbm_global
```

## 7. Optional patterns for segmented models

If you train multiple sub-models in one experiment (for example one per location):

1. Compute test metrics per segment.
2. Add `n_samples` to each segment via `with_sample_count(...)`.
3. Use `weighted_overall(segment_metrics)` for fair overall comparison.
4. Put sub-model metrics and paths into `metadata` for traceability.

## 8. Troubleshooting

- `Unknown experiments: ...`: name is not in `EXPERIMENT_REGISTRY`.
- `Could not find dataset...`: check `--data-path` or dataset location.
- `LightGBM is required...`: install dependencies (`pip install -r requirements.txt`).
- `Dataset is missing required columns...`: ensure input CSV follows the data contract above.
