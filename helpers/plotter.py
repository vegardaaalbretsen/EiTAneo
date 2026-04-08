from pathlib import Path
import re
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.base import RunModes
from experiments.metrics import regression_metrics


class Plotter:
    @staticmethod
    def _slugify_filename(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "plot"

    @staticmethod
    def create_daily_line_plots_by_location(
        predictions_df: pd.DataFrame,
        output_dir: Path,
        filename_stem: str,
        title_prefix: str,
        location_names: Mapping[int, str] | None = None,
        samples_per_day: int = 24,
    ) -> dict[str, str]:
        required_cols = {"actual", "predicted", "location_id"}
        missing_cols = required_cols.difference(predictions_df.columns)
        if missing_cols:
            return {}

        if samples_per_day <= 0:
            raise ValueError("samples_per_day must be greater than 0.")

        output_dir.mkdir(parents=True, exist_ok=True)

        working_df = predictions_df.reset_index()
        row_col = working_df.columns[0]
        saved_paths: dict[str, str] = {}

        for raw_location_id, location_df in working_df.groupby("location_id", sort=True):
            resolved_location_id = int(raw_location_id)
            location_name = (
                location_names.get(resolved_location_id, f"location_{resolved_location_id}")
                if location_names is not None
                else f"location_{resolved_location_id}"
            )
            safe_location_name = Plotter._slugify_filename(location_name)

            plot_df = (
                location_df.sort_values(row_col)
                .drop_duplicates(subset=[row_col], keep="last")
                .reset_index(drop=True)
            )
            if plot_df.empty:
                continue

            plot_df["sample_step"] = np.arange(len(plot_df))
            day_starts = np.arange(0, len(plot_df), samples_per_day)
            num_days = int(np.ceil(len(plot_df) / samples_per_day))
            metrics = regression_metrics(plot_df["actual"], plot_df["predicted"])

            fig, ax = plt.subplots(figsize=(16, 5))
            ax.plot(
                plot_df["sample_step"],
                plot_df["actual"],
                label="Actual",
                linewidth=1.8,
                color="black",
                alpha=0.85,
            )
            ax.plot(
                plot_df["sample_step"],
                plot_df["predicted"],
                label="Predicted",
                linewidth=1.8,
                color="orange",
                alpha=0.8,
            )

            for day_start in day_starts[1:]:
                ax.axvline(day_start - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

            tick_stride = max(1, int(np.ceil(num_days / 12)))
            tick_positions = day_starts[::tick_stride]
            tick_labels = [f"Day {(position // samples_per_day) + 1}" for position in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(0, max(len(plot_df) - 1, 1))
            if location_name.lower() in title_prefix.lower():
                plot_title = title_prefix
            else:
                plot_title = f"{title_prefix}: {location_name}"

            ax.set_xlabel("Day in Test Horizon")
            ax.set_ylabel("Consumption (MW)")
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5))


            metric_text = (
                f"Days: {num_days}\n"
                f"Samples: {len(plot_df):,}\n"
                f"MAE: {metrics['mae']:.3f}\n"
                f"RMSE: {metrics['rmse']:.3f}"
            )
            ax.text(
                0.01,
                0.98,
                metric_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )

            plt.tight_layout()
            plot_path = (
                output_dir
                / f"{filename_stem}_{resolved_location_id}_{safe_location_name}_daily_lines.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            saved_paths[f"{resolved_location_id}_{safe_location_name}"] = str(plot_path)

        return saved_paths

    @staticmethod
    def create_actual_vs_predicted_plot(
        predictions_df: pd.DataFrame,
        output_path: Path,
        title: str,
        sample_size: int = 3000,
    ) -> None:
        required_cols = {"actual", "predicted"}
        missing_cols = required_cols.difference(predictions_df.columns)
        if missing_cols:
            missing = ", ".join(sorted(missing_cols))
            raise ValueError(f"Predictions dataframe is missing required columns: {missing}")

        full_plot_df = predictions_df.loc[:, ["actual", "predicted"]].dropna()
        if full_plot_df.empty:
            raise ValueError("Predictions dataframe does not contain any plottable rows.")

        if len(full_plot_df) > sample_size:
            plot_df = full_plot_df.sample(n=sample_size, random_state=42)
            sampled = True
        else:
            plot_df = full_plot_df
            sampled = False

        metrics = regression_metrics(full_plot_df["actual"], full_plot_df["predicted"])
        actual = plot_df["actual"].to_numpy()
        predicted = plot_df["predicted"].to_numpy()

        min_value = float(min(actual.min(), predicted.min()))
        max_value = float(max(actual.max(), predicted.max()))
        if np.isclose(min_value, max_value):
            padding = max(abs(min_value) * 0.05, 1.0)
            min_value -= padding
            max_value += padding

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            actual,
            predicted,
            alpha=0.5,
            s=20,
            color="steelblue",
            edgecolors="none",
        )
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
            linewidth=2,
            color="crimson",
        )
        ax.set_xlabel("Actual Consumption (MW)")
        ax.set_ylabel("Predicted Consumption (MW)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        sample_note = ""
        if sampled:
            sample_note = f"\nScatter sampled: {len(plot_df):,} of {len(full_plot_df):,} rows"
        metric_text = (
            f"MAE: {metrics['mae']:.3f}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"R2: {metrics['r2']:.3f}"
            f"{sample_note}"
        )
        ax.text(
            0.05,
            0.95,
            metric_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def create_rolling_plots(df, results_df, mode: RunModes, experiment_dir: Path):

        # -------------------------
        # 1. Window Coverage Plot
        # -------------------------
        fig, ax = plt.subplots(figsize=(12, 6))

        y_positions = results_df["window"].values

        # Train segments
        ax.hlines(
            y=y_positions,
            xmin=results_df["train_start"],
            xmax=results_df["train_end"],
            color=sns.color_palette("tab10")[0],
            linewidth=8,
            label="Train",
        )

        # Test segments
        ax.hlines(
            y=y_positions,
            xmin=results_df["test_start"],
            xmax=results_df["test_end"],
            color=sns.color_palette("tab10")[1],
            linewidth=8,
            linestyles="dashed",
            label="Test",
        )

        # Formatting
        ax.set_title("Sliding / Expanding Window Coverage")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Window")

        # Earliest window at top
        ax.invert_yaxis()

        # Clean y-axis ticks
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"W{i}" for i in y_positions])

        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(experiment_dir / "window_coverage.png")
        plt.close()


        # -------------------------
        # 2. Metric Over Time
        # -------------------------
        metric_cols = [c for c in results_df.columns if c not in 
                    ["window", "train_start", "train_end", "test_start", "test_end"]]

        for metric in metric_cols:
            plt.figure(figsize=(10, 5))
            plt.plot(results_df["window"], results_df[metric], marker="o")
            plt.title(f"{metric} over {mode.value.capitalize()} Windows")
            plt.xlabel("Window")
            plt.ylabel(metric)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(experiment_dir / f"{metric}_over_time.png")
            plt.close()
