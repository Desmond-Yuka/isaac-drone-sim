"""Compare several runs: a metrics table (Markdown + CSV) and overlaid figures.

Typical use: the same trajectory flown by different controllers or gains, e.g.
``python -m isaac_drone compare runs/A runs/B --labels geometric pid``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .metrics import METRIC_KEYS, compute_metrics
from .plots import load_basic_csv, phase_starts

TABLE_KEYS = ("success", "completion_time_s", *METRIC_KEYS)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def comparison_rows(run_dirs, labels=None, phase=None) -> list[dict]:
    """One row per run with TABLE_KEYS, overall or for one mission ``phase``."""
    labels = labels or [Path(run).name for run in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("Give one label per run")
    rows = []
    for label, run in zip(labels, run_dirs):
        metrics = compute_metrics(run)
        values = metrics["overall"] if phase is None else metrics["phases"].get(phase)
        if values is None:
            raise ValueError(f"{run} has no phase {phase!r}; available: {sorted(metrics['phases'])}")
        rows.append({"label": label, "run": str(run), "controller": metrics["run"]["controller"],
                     "trajectory": metrics["run"]["trajectory"], "success": metrics["run"]["success"],
                     "completion_time_s": metrics["run"]["completion_time_s"], **values})
    return rows


def markdown_table(rows) -> str:
    keys = ("label", "controller", "trajectory", *TABLE_KEYS)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(_fmt(row.get(key)) for key in keys) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(rows, path) -> Path:
    keys = ("label", "run", "controller", "trajectory", *TABLE_KEYS)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


def comparison_figures(run_dirs, labels, output_dir) -> list[Path]:
    """Position-error, tilt and top-view path overlays of all runs (matplotlib, headless)."""
    from matplotlib.figure import Figure

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = [load_basic_csv(Path(run) / "basic.csv") for run in run_dirs]
    paths = []

    figure = Figure(figsize=(10, 7), layout="constrained")
    error_axis, tilt_axis = figure.subplots(2, 1, sharex=True)
    for label, columns in zip(labels, data):
        t = columns["sample_time_s"]
        error_axis.plot(t, columns["position_error_norm_m"], linewidth=1.1, label=label)
        x, y = columns["quaternion_wxyz_x"], columns["quaternion_wxyz_y"]
        tilt_axis.plot(t, np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))), linewidth=1.0, label=label)
    first = Path(run_dirs[0])
    config = json.loads((first / "config.json").read_text()) if (first / "config.json").is_file() else {}
    metadata = json.loads((first / "metadata.json").read_text()) if (first / "metadata.json").is_file() else {}
    for index, (start, name) in enumerate(phase_starts(config, metadata)):
        for axis in (error_axis, tilt_axis):
            axis.axvline(start, color="0.6", linestyle=":", linewidth=0.8)
        error_axis.annotate(name, (start, 1.0), xycoords=("data", "axes fraction"), fontsize=8, color="0.4",
                            xytext=(2, -10 - 10 * (index % 2)), textcoords="offset points")
    error_axis.set_ylabel("|position error| [m]")
    tilt_axis.set_ylabel("tilt [deg]")
    tilt_axis.set_xlabel("Time [s]")
    for axis in (error_axis, tilt_axis):
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Run comparison")
    paths.append(output / "compare_errors.png")
    figure.savefig(paths[-1], dpi=150)

    figure = Figure(figsize=(7, 7), layout="constrained")
    axis = figure.subplots()
    columns = data[0]
    axis.plot(columns["target_position_w_x_m"], columns["target_position_w_y_m"], "k--", linewidth=1.0,
              label=f"target ({labels[0]})")
    for label, columns in zip(labels, data):
        axis.plot(columns["position_w_x_m"], columns["position_w_y_m"], linewidth=1.0, label=label)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    axis.set_title("Top view (world frame)")
    paths.append(output / "compare_top_view.png")
    figure.savefig(paths[-1], dpi=150)
    return paths


def compare_runs(run_dirs, output_dir, labels=None, phase=None, figures=True) -> dict:
    """Write comparison.md/.csv (and figures) into ``output_dir``; return the table rows and paths."""
    if len(run_dirs) < 1:
        raise ValueError("Give at least one run directory")
    labels = labels or [Path(run).name for run in run_dirs]
    rows = comparison_rows(run_dirs, labels, phase)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    title = "overall" if phase is None else f"phase {phase}"
    table = markdown_table(rows)
    (output / "comparison.md").write_text(f"# Run comparison ({title})\n\n{table}\n", encoding="utf-8")
    write_csv(rows, output / "comparison.csv")
    paths = comparison_figures(run_dirs, labels, output) if figures else []
    return {"rows": rows, "table": table, "files": [output / "comparison.md", output / "comparison.csv", *paths]}


__all__ = ["TABLE_KEYS", "compare_runs", "comparison_figures", "comparison_rows", "markdown_table", "write_csv"]

