# EiT Aneo 2026

## Experiment Framework

Use the runner to train and compare multiple model strategies with shared data retrieval/splitting.

Detailed documentation: `docs/EXPERIMENT_FRAMEWORK.md`

### List available experiments

```bash
python run_experiments.py --list
```

### Run all experiments

```bash
python run_experiments.py
```

### Run selected experiments

```bash
python run_experiments.py --experiments linear_regression_global lightgbm_global
```

### Run with a specific mode

Run modes control how the dataset is split and how experiments are executed over time. Available modes are:

- `chronological` — single chronological train/test split (default)
- `sliding_window` — repeated training on sliding windows
- `expanding_window` — training on an expanding window over time

Example — run experiments using the sliding window evaluation:

```bash
python run_experiments.py --mode sliding_window
```

You can combine `--mode` with `--experiments` to run a subset of experiments with the chosen evaluation strategy:

```bash
python run_experiments.py --mode sliding_window --experiments lightgbm_global
```


### Outputs

By default, results are written to `results/experiments/`:

- `comparison.csv`: Comparable test metrics for all runs.
- `comparison.json`: Same summary in JSON format.
- `results/experiments/<experiment_name>/result.json`: Detailed per-experiment metrics/metadata.
- Model files for each experiment (for example `model.txt`, `model.pkl`, location/group files).

### How it runs

- `run_experiments.py` parses CLI args and resolves selected experiment names.
- `experiments/registry.py` maps experiment names to classes.
- `experiments/runner.py` loads data once, runs each experiment class, writes outputs, and builds comparison tables.
- Each experiment class returns an `ExperimentResult` with overall test metrics, segmented metrics, and metadata.
