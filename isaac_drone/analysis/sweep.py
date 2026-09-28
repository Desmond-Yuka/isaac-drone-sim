"""Parameter sweeps on the synthetic backend: every grid point is a full recorded run.

``--grid key.path=[v1, v2, ...]`` axes are combined as a Cartesian product and
applied like ``--set`` overrides on top of the experiment config. All points
are validated before any flight starts. Runs execute in parallel worker
processes and are written under ``<sweep dir>/points/``; ``summary.csv``,
``summary.json`` and ``summary.md`` rank them by one metric.
"""
from __future__ import annotations

import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from isaac_drone.config import ConfigurationError, load_config
from isaac_drone.config.loader import parse_yaml


def parse_grid(items) -> list[tuple[str, list]]:
    """``["a.b=[1, 2]", ...]`` -> ``[("a.b", [1, 2]), ...]``; each axis needs a nonempty YAML list."""
    grid = []
    for item in items:
        if "=" not in item:
            raise ConfigurationError(f"Grid axis {item!r} must look like key.path=[value, ...]")
        key, text = item.split("=", 1)
        values = parse_yaml(text, f"--grid {key}")
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"Grid axis {key!r} needs a nonempty YAML list of values")
        grid.append((key.strip(), values))
    if len({key for key, _ in grid}) != len(grid):
        raise ConfigurationError("Each grid key may appear only once")
    return grid


def grid_points(grid) -> list[dict]:
    """Every combination as {key: value}."""
    keys = [key for key, _ in grid]
    return [dict(zip(keys, values)) for values in itertools.product(*(values for _, values in grid))]


def _overrides(point: dict) -> list[str]:
    return [f"{key}={json.dumps(value)}" for key, value in point.items()]


def _run_point(job) -> dict:
    """Worker: fly one point and return its flat metrics row (errors are reported, not raised)."""
    index, config_path, base, point, points_dir, figures = job
    from isaac_drone.analysis.metrics import compute_metrics, flatten
    from isaac_drone.sim.synthetic import run_simulation

    row = {"index": index, **{f"param:{key}": json.dumps(value) for key, value in point.items()}}
    try:
        config = load_config(config_path, overrides=[*base, *_overrides(point)])
        result = run_simulation(config, figures=figures, log=lambda *_: None, log_root=points_dir)
        metrics_path = result.run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else compute_metrics(result.run_dir)
        row.update({"run_dir": str(result.run_dir), "error": None, **flatten(metrics)})
    except Exception as error:  # one diverging point must not abort the sweep
        row.update({"run_dir": None, "error": f"{type(error).__name__}: {error}", "success": False})
    return row


def _sort_key(rank):
    def key(row):
        value = row.get(rank)
        return (row.get("error") is not None, not row.get("success"), value is None, value if value is not None else 0)
    return key


def run_sweep(config_path, grid, *, base_overrides=(), output_dir=None, workers=None, rank="position_error_rms_m",
              figures=False, log=print, runs_root=None) -> dict:
    points = grid_points(grid)
    for point in points:  # fail before flying anything
        load_config(config_path, overrides=[*base_overrides, *_overrides(point)])
    if output_dir is None:
        from isaac_drone.runtime.runner import REPO_ROOT
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(runs_root or REPO_ROOT / "runs") / f"sweep_{stamp}"
    output = Path(output_dir)
    (output / "points").mkdir(parents=True, exist_ok=True)
    log(f"Sweep: {len(points)} runs of {config_path} -> {output}")
    jobs = [(index, str(config_path), list(base_overrides), point, str(output / "points"), figures)
            for index, point in enumerate(points)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_run_point, jobs):
            status = "error" if row["error"] else ("ok" if row.get("success") else "failed")
            log(f"  [{row['index'] + 1}/{len(points)}] {status}: "
                + ", ".join(f"{key[6:]}={value}" for key, value in row.items() if key.startswith("param:"))
                + ("" if row.get(rank) is None else f"  {rank}={row[rank]:.4g}"))
            rows.append(row)
    rows.sort(key=_sort_key(rank))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({"config": str(config_path), "base_overrides": list(base_overrides),
                                                     "grid": [[key, values] for key, values in grid], "rank": rank,
                                                     "rows": rows}, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    shown = [key for key in columns if key.startswith("param:")] + ["success", rank, "position_error_max_m",
                                                                     "completion_time_s", "error"]
    lines = ["| " + " | ".join(key.removeprefix("param:") for key in shown) + " |", "|" + "---|" * len(shown)]
    for row in rows:
        cells = []
        for key in shown:
            value = row.get(key)
            cells.append("-" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (output / "summary.md").write_text(f"# Sweep of {config_path} (ranked by {rank})\n\n{table}\n", encoding="utf-8")
    return {"output_dir": output, "rows": rows, "table": table}
