from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.base import RunModes


class Plotter:

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