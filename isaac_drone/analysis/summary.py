"""Per-phase console summary of one run (computed with analysis.metrics)."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import compute_metrics, load_records


def latest_telemetry(runs_dir) -> Path:
    candidates = list(Path(runs_dir).glob("*/telemetry.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No run with telemetry.jsonl under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _cell(value, width, digits):
    return f"{'-':>{width}}" if value is None else f"{value:{width}.{digits}f}"


def summarize_run(path) -> str:
    """Table of position error, tilt, rates, motor speeds and saturation per mission phase.

    ``path`` is a run directory or its telemetry.jsonl.
    """
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    records, end = load_records(directory)
    metrics = compute_metrics(directory, (records, end))
    run = metrics["run"]
    lines = [f"run: {directory}  controller={run['controller']}  trajectory={run['trajectory']}  "
             f"backend={run['backend']}",
             f"{'phase':9} {'t[s]':>13} {'err_max':>8} {'err_rms':>8} {'err_end':>8} {'tilt_max':>8} "
             f"{'w_max':>6} {'rps_min':>7} {'rps_max':>7} {'sat%':>5}"]
    starts = {}
    for record in records:
        starts.setdefault(record.get("mission_phase", "unknown"), record["basic"]["sample_time_s"])
    for name, values in [*metrics["phases"].items(), ("overall", metrics["overall"])]:
        start = starts.get(name, records[0]["basic"]["sample_time_s"])
        span = f"{start:6.2f}-{start + values['duration_s'] - records[0]['basic']['interval_dt_s']:6.2f}"
        lines.append(f"{name:9} {span:>13} {_cell(values['position_error_max_m'], 8, 3)} "
                     f"{_cell(values['position_error_rms_m'], 8, 3)} {_cell(values['position_error_final_m'], 8, 3)} "
                     f"{_cell(values['tilt_max_deg'], 7, 1)}° {_cell(values['angular_speed_max_rad_s'], 6, 2)} "
                     f"{_cell(values['motor_speed_min_rps'], 7, 1)} {_cell(values['motor_speed_max_rps'], 7, 1)} "
                     f"{100 * values['saturation_fraction']:5.1f}")
    last = records[-1]["basic"]
    lines.append(f"final position: {[round(v, 3) for v in last['position_w_m']]}  "
                 f"target: {[round(v, 3) for v in last['target_position_w_m']]}")
    if run["completion_time_s"] is not None:
        lines.append(f"completion criterion first met at {run['completion_time_s']:.2f} s")
    if end:
        keys = ("event", "success", "physics_steps", "simulated_time_s", "error")
        lines.append("end event: " + json.dumps({key: end.get(key) for key in keys}))
    return "\n".join(lines)
