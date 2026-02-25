# EiT Aneo 2026

## Experiment Framework

Use the runner to train and compare multiple model strategies with shared data retrieval/splitting.

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
