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
