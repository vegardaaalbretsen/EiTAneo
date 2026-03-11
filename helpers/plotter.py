# plotter.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from experiments.base import ExperimentResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS: dict[str, str] = {
    "mae": "MAE",
    "rmse": "RMSE",
    "nrmse": "NRMSE",
    "mape": "MAPE (%)",
    "r2": "R²",
    "smape": "SMAPE (%)",
}

SEGMENT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric.lower(), metric)


def _segment_color_map(segments: list[str]) -> dict[str, str]:
    return {seg: SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i, seg in enumerate(segments)}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plotter
# ─────────────────────────────────────────────────────────────────────────────

class Plotter:
    """
    Generates diagnostic plots for day-ahead forecasting experiments.

    Works for:
    - Model types : global, two_group, location_specific (and any new ones)
    - Training modes: chronological, sliding_window, expanding_window

    Usage
    -----
    plotter = Plotter(output_dir=Path("plots"))

    # After a chronological run:
    plotter.plot_chronological(result, actuals_df)

    # After a rolling/expanding run (pass the window_metrics CSVs):
    plotter.plot_rolling(result, window_csv_dir=experiment_dir)
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    # ── Public entry-points ───────────────────────────────────────────────────

    def plot_chronological(
        self,
        result: ExperimentResult,
        actuals_df: pd.DataFrame | None = None,
        predictions: dict[str, pd.Series] | None = None,
    ) -> None:
        """
        Plots for a single chronological train/val/test split.

        Parameters
        ----------
        result      : ExperimentResult returned by the experiment.
        actuals_df  : (optional) DataFrame with a DatetimeIndex and a column
                      matching TARGET_COLUMN, used for time-series residual plots.
        predictions : (optional) {segment_key: pd.Series of predictions}
                      keyed the same way as result.segment_test_metrics.
        """
        exp_dir = self.output_dir / result.experiment_name / "chronological"

        self._plot_segment_metric_bar(
            segment_metrics=result.segment_test_metrics,
            title=f"{result.experiment_name} — Test metrics per segment",
            save_path=exp_dir / "segment_metrics_bar.png",
        )

        self._plot_metric_comparison_table(
            segment_metrics=result.segment_test_metrics,
            overall_metrics=result.overall_test_metrics,
            save_path=exp_dir / "metrics_table.png",
        )

        if predictions and actuals_df is not None:
            for seg_key, preds in predictions.items():
                self._plot_actual_vs_predicted(
                    actuals=actuals_df,
                    predictions=preds,
                    title=f"{result.experiment_name} [{seg_key}] — Actual vs Predicted",
                    save_path=exp_dir / f"{seg_key}_actual_vs_predicted.png",
                )
                self._plot_residuals(
                    actuals=actuals_df,
                    predictions=preds,
                    title=f"{result.experiment_name} [{seg_key}] — Residuals",
                    save_path=exp_dir / f"{seg_key}_residuals.png",
                )

    def plot_rolling(
        self,
        result: "ExperimentResult",
        window_csv_dir: Path,
        mode: str = "sliding_window",
    ) -> None:
        """
        Plots for sliding- or expanding-window experiments.

        Parameters
        ----------
        result          : ExperimentResult returned by the experiment.
        window_csv_dir  : Directory that contains the per-window CSV files
                          written by the experiment (e.g. `{segment}_window_metrics.csv`
                          or `window_metrics.csv` for global models).
        mode            : 'sliding_window' or 'expanding_window'
        """
        exp_dir = self.output_dir / result.experiment_name / mode
        window_csv_dir = Path(window_csv_dir)

        # ── Load all window CSVs ──────────────────────────────────────────────
        segment_window_data: dict[str, pd.DataFrame] = {}

        # Global model writes a single "window_metrics.csv"
        global_csv = window_csv_dir / "window_metrics.csv"
        if global_csv.exists():
            segment_window_data["all"] = pd.read_csv(global_csv)
        else:
            # Location-specific / two-group: one CSV per segment
            for csv_path in sorted(window_csv_dir.glob("*_window_metrics.csv")):
                seg_key = csv_path.stem.replace("_window_metrics", "")
                segment_window_data[seg_key] = pd.read_csv(csv_path)

        if not segment_window_data:
            raise FileNotFoundError(f"No window metric CSVs found in {window_csv_dir}")

        # ── Per-segment rolling metric line plots ─────────────────────────────
        for seg_key, df in segment_window_data.items():
            self._plot_rolling_metrics_over_windows(
                df=df,
                segment_name=seg_key,
                mode=mode,
                save_path=exp_dir / f"{seg_key}_rolling_metrics.png",
            )

        # ── All segments on one chart (e.g. MAE comparison) ──────────────────
        self._plot_rolling_metric_across_segments(
            segment_window_data=segment_window_data,
            metric="mae",
            mode=mode,
            title=f"{result.experiment_name} — MAE across segments ({mode})",
            save_path=exp_dir / "all_segments_mae.png",
        )

        # ── Final averaged metrics bar chart ──────────────────────────────────
        self._plot_segment_metric_bar(
            segment_metrics=result.segment_test_metrics,
            title=f"{result.experiment_name} — Avg test metrics per segment ({mode})",
            save_path=exp_dir / "segment_metrics_bar.png",
        )

        # ── Window-count / train-size growth (expanding only) ─────────────────
        if mode == "expanding_window":
            for seg_key, df in segment_window_data.items():
                if "train_start" in df.columns and "train_end" in df.columns:
                    self._plot_expanding_train_size(
                        df=df,
                        segment_name=seg_key,
                        save_path=exp_dir / f"{seg_key}_train_size_growth.png",
                    )

    # ── Core plot builders ────────────────────────────────────────────────────

    def _plot_segment_metric_bar(
        self,
        segment_metrics: dict[str, dict[str, float]],
        title: str,
        save_path: Path,
        metrics: list[str] | None = None,
    ) -> None:
        """Grouped bar chart — one group per segment, one bar per metric."""
        if metrics is None:
            sample_vals = next(iter(segment_metrics.values()))
            metrics = [k for k in sample_vals if k != "n_samples"]

        segments = list(segment_metrics.keys())
        n_seg = len(segments)
        n_met = len(metrics)
        colors = _segment_color_map(segments)

        fig, axes = plt.subplots(
            1, n_met, figsize=(4 * n_met, max(4, 0.6 * n_seg)), squeeze=False
        )

        for col_idx, metric in enumerate(metrics):
            ax = axes[0][col_idx]
            values = [segment_metrics[s].get(metric, float("nan")) for s in segments]
            bars = ax.barh(
                segments,
                values,
                color=[colors[s] for s in segments],
                edgecolor="white",
                height=0.6,
            )
            ax.set_xlabel(_label(metric))
            ax.set_title(_label(metric))
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, save_path)

    def _plot_rolling_metrics_over_windows(
        self,
        df: pd.DataFrame,
        segment_name: str,
        mode: str,
        save_path: Path,
        metrics: list[str] | None = None,
    ) -> None:
        """Line plot of each metric over successive rolling windows."""
        meta_cols = {"window", "train_start", "train_end", "test_start", "test_end", "n_samples"}
        if metrics is None:
            metrics = [c for c in df.columns if c not in meta_cols]

        n = len(metrics)
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

        x = df["window"] if "window" in df.columns else range(len(df))
        x_label = "Window index"

        # Optionally use test_start as x-axis if it's a date
        if "test_start" in df.columns:
            try:
                x = pd.to_datetime(df["test_start"])
                x_label = "Test window start"
            except Exception:
                pass

        for idx, metric in enumerate(metrics):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            if metric not in df.columns:
                ax.set_visible(False)
                continue
            ax.plot(x, df[metric], marker="o", linewidth=1.8, markersize=4, color="#4C72B0")
            # Rolling mean overlay
            if len(df) >= 3:
                roll = df[metric].rolling(3, min_periods=1).mean()
                ax.plot(x, roll, linewidth=1.2, linestyle="--", color="#DD8452", alpha=0.8, label="3-win avg")
                ax.legend(fontsize=7)
            ax.set_title(_label(metric), fontsize=9)
            ax.set_xlabel(x_label, fontsize=7)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.spines[["top", "right"]].set_visible(False)

        # Hide any empty subplots
        for idx in range(len(metrics), rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            f"{segment_name} — metrics over windows ({mode})",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, save_path)

    def _plot_rolling_metric_across_segments(
        self,
        segment_window_data: dict[str, pd.DataFrame],
        metric: str,
        mode: str,
        title: str,
        save_path: Path,
    ) -> None:
        """One line per segment on a single axes — good for MAE / RMSE comparison."""
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = _segment_color_map(list(segment_window_data.keys()))

        for seg_key, df in segment_window_data.items():
            if metric not in df.columns:
                continue
            x = range(len(df))
            if "test_start" in df.columns:
                try:
                    x = pd.to_datetime(df["test_start"])
                except Exception:
                    pass
            ax.plot(x, df[metric], marker="o", markersize=3, linewidth=1.5,
                    color=colors[seg_key], label=seg_key)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Window" if not isinstance(x, pd.DatetimeIndex) else "Test window start")
        ax.set_ylabel(_label(metric))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, fontsize=8)
        fig.tight_layout()
        _save(fig, save_path)

    def _plot_actual_vs_predicted(
        self,
        actuals: pd.DataFrame,
        predictions: pd.Series,
        title: str,
        save_path: Path,
        target_col: str = "target",
    ) -> None:
        """Two subplots: time-series overlay + scatter with identity line."""
        y_true = actuals[target_col] if target_col in actuals.columns else actuals.iloc[:, 0]
        y_pred = predictions.reindex(y_true.index)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                        gridspec_kw={"height_ratios": [3, 2]})

        # Time-series panel
        ax1.plot(y_true.index, y_true.values, label="Actual", linewidth=1.2, color="#4C72B0")
        ax1.plot(y_pred.index, y_pred.values, label="Predicted", linewidth=1.2,
                 color="#DD8452", linestyle="--")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.spines[["top", "right"]].set_visible(False)

        # Scatter panel
        common = y_true.dropna().index.intersection(y_pred.dropna().index)
        vmin = min(y_true[common].min(), y_pred[common].min())
        vmax = max(y_true[common].max(), y_pred[common].max())
        ax2.scatter(y_true[common], y_pred[common], alpha=0.3, s=10, color="#55A868")
        ax2.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="Identity")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.legend(fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout()
        _save(fig, save_path)

    def _plot_residuals(
        self,
        actuals: pd.DataFrame,
        predictions: pd.Series,
        title: str,
        save_path: Path,
        target_col: str = "target",
    ) -> None:
        """Residuals over time + histogram."""
        y_true = actuals[target_col] if target_col in actuals.columns else actuals.iloc[:, 0]
        y_pred = predictions.reindex(y_true.index)
        residuals = y_true - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(residuals.index, residuals.values, linewidth=0.8, color="#C44E52", alpha=0.8)
        ax1.axhline(0, color="black", linewidth=1, linestyle="--")
        ax1.set_ylabel("Residual (actual − predicted)")
        ax1.set_xlabel("Time")
        ax1.set_title("Residuals over time")
        ax1.spines[["top", "right"]].set_visible(False)

        ax2.hist(residuals.dropna(), bins=50, color="#8172B3", edgecolor="white", alpha=0.85)
        ax2.axvline(0, color="black", linewidth=1, linestyle="--")
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Count")
        ax2.set_title("Residual distribution")
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout()
        _save(fig, save_path)

    def _plot_expanding_train_size(
        self,
        df: pd.DataFrame,
        segment_name: str,
        save_path: Path,
    ) -> None:
        """Bar chart showing training-set row count growth across expanding windows."""
        if "n_samples" not in df.columns:
            return

        x = df["window"] if "window" in df.columns else range(len(df))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(x, df["n_samples"], color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Window index")
        ax.set_ylabel("Train set size (rows)")
        ax.set_title(f"{segment_name} — expanding train size", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        _save(fig, save_path)

    def _plot_metric_comparison_table(
        self,
        segment_metrics: dict[str, dict[str, float]],
        overall_metrics: dict[str, float],
        save_path: Path,
    ) -> None:
        """Render a styled table of all segment + overall metrics as a PNG."""
        rows = list(segment_metrics.items()) + [("⬛ overall", overall_metrics)]
        sample_vals = next(iter(segment_metrics.values()))
        metric_cols = [k for k in sample_vals if k != "n_samples"]

        table_data = []
        for seg, vals in rows:
            row = [seg] + [f"{vals.get(m, float('nan')):.4f}" for m in metric_cols]
            if "n_samples" in vals:
                row.append(str(int(vals["n_samples"])))
            table_data.append(row)

        col_labels = ["Segment"] + [_label(m) for m in metric_cols]
        if any("n_samples" in v for v in segment_metrics.values()):
            col_labels.append("n")

        fig, ax = plt.subplots(
            figsize=(max(8, 2 * len(col_labels)), 0.45 * len(table_data) + 1.2)
        )
        ax.axis("off")
        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)

        # Highlight overall row
        for c in range(len(col_labels)):
            tbl[len(rows), c].set_facecolor("#DDEBF7")
            tbl[len(rows), c].set_text_props(fontweight="bold")

        fig.tight_layout()
        _save(fig, save_path)