"""CLI entrypoint for running and comparing model experiments."""

from __future__ import annotations

import argparse

from experiments.config import DEFAULT_OUTPUT_DIR
from experiments.registry import available_experiments
from experiments.runner import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more model experiments and compare test metrics."
    )
    parser.add_argument(
        "--data-path",
        default="data/preprocessed_data.csv",
        help="Path to the preprocessed dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for run outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Experiment names to run, or use 'all'.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    return parser.parse_args()


def resolve_experiment_names(raw_names: list[str]) -> list[str]:
    if any(name.lower() == "all" for name in raw_names):
        return available_experiments()
    return raw_names


def main() -> None:
    args = parse_args()
    available = available_experiments()

    if args.list:
        print("Available experiments:")
        for experiment_name in available:
            print(f" - {experiment_name}")
        return

    selected = resolve_experiment_names(args.experiments)
    try:
        results, summary_df, comparison_csv, comparison_json = run_experiments(
            experiment_names=selected,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Ran {len(results)} experiment(s).")
    print(f"Comparison CSV:  {comparison_csv}")
    print(f"Comparison JSON: {comparison_json}")

    if not summary_df.empty:
        print("\nResults (sorted by MAE):")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
