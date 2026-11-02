"""Standard run metrics from telemetry.jsonl: overall and per mission phase.

One definition of "how well did this run fly" shared by the per-run summary,
multi-run comparison and parameter sweeps. Errors are ideal minus actual as
recorded (docs/telemetry.md); nothing is re-simulated or re-sampled here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

METRICS_FILE = "metrics.json"
# Column order used by tables; every key is present for every phase (None when unavailable).
METRIC_KEYS = (
    "duration_s",
    "position_error_max_m",
    "position_error_rms_m",
    "position_error_final_m",
    "velocity_error_rms_m_s",
    "attitude_error_max_deg",
    "attitude_error_rms_deg",
    "tilt_max_deg",
    "speed_max_m_s",
    "angular_speed_max_rad_s",
    "motor_speed_min_rps",
    "motor_speed_max_rps",
    "saturation_fraction",
    "mean_total_thrust_n",
)


def _telemetry_path(run) -> Path:
    path = Path(run)
    return path / "telemetry.jsonl" if path.is_dir() else path


def load_records(run) -> tuple[list[dict], dict | None]:
    """Step records that carry ``basic`` data, and the final finished/aborted event (if any)."""
    records, end = [], None
    with _telemetry_path(run).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") in ("finished", "aborted"):
                end = record
            elif record.get("basic") is not None:
                records.append(record)
    return records, end


def _rms(values) -> float | None:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def _max(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(max(values)) if values else None


def _tilt_deg(quaternion) -> float:
    _, x, y, _ = quaternion
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def phase_metrics(records: list[dict]) -> dict:
    """Metrics over a list of step records (see METRIC_KEYS for names and units)."""
    basics = [record["basic"] for record in records]
    times = [basic["sample_time_s"] for basic in basics]
    speeds = [basic["motor_speed_rps"] for basic in basics if basic.get("motor_speed_rps") is not None]
    thrusts = [basic["motor_thrust_applied_n"] for basic in basics if basic.get("motor_thrust_applied_n") is not None]
    attitude = [
        None if basic.get("attitude_error_angle_rad") is None else math.degrees(basic["attitude_error_angle_rad"])
        for basic in basics
    ]
    return {
        "duration_s": float(times[-1] - basics[0]["step_start_time_s"]),
        "position_error_max_m": _max(basic["position_error_norm_m"] for basic in basics),
        "position_error_rms_m": _rms(basic["position_error_norm_m"] for basic in basics),
        "position_error_final_m": float(basics[-1]["position_error_norm_m"]),
        "velocity_error_rms_m_s": _rms(basic.get("velocity_error_norm_m_s") for basic in basics),
        "attitude_error_max_deg": _max(attitude),
        "attitude_error_rms_deg": _rms(attitude),
        "tilt_max_deg": _max(_tilt_deg(basic["quaternion_wxyz"]) for basic in basics),
        "speed_max_m_s": _max(float(np.linalg.norm(basic["velocity_w_m_s"])) for basic in basics),
        "angular_speed_max_rad_s": _max(float(np.linalg.norm(basic["angular_velocity_b_rad_s"])) for basic in basics),
        "motor_speed_min_rps": float(np.min(speeds)) if speeds else None,
        "motor_speed_max_rps": float(np.max(speeds)) if speeds else None,
        "saturation_fraction": float(np.mean([bool(record.get("allocation_saturated")) for record in records])),
        "mean_total_thrust_n": float(np.mean(np.sum(thrusts, axis=1))) if thrusts else None,
    }


def compute_metrics(run, loaded=None) -> dict:
    """Metrics for one run directory (or telemetry.jsonl path); ``loaded`` reuses load_records() output."""
    path = _telemetry_path(run)
    records, end = load_records(path) if loaded is None else loaded
    if not records:
        raise ValueError(f"{path} has no step records")
    directory = path.parent
    config = (
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if (directory / "config.json").is_file()
        else {}
    )
    metadata = (
        json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (directory / "metadata.json").is_file()
        else {}
    )
    phases: dict[str, list] = {}
    for record in records:
        phases.setdefault(record.get("mission_phase", "unknown"), []).append(record)
    completion = next(
        (record["basic"]["sample_time_s"] for record in records if (record.get("mission") or {}).get("achieved")), None
    )
    run_info = {
        "name": directory.name,
        "success": None if end is None else end.get("success"),
        "end_event": None if end is None else end.get("event"),
        "backend": metadata.get("backend_kind"),
        "controller": config.get("controller", {}).get("kind"),
        "trajectory": config.get("trajectory", {}).get("kind"),
        "simulated_time_s": records[-1]["basic"]["sample_time_s"],
        "completion_time_s": completion,
    }
    return {
        "run": run_info,
        "overall": phase_metrics(records),
        "phases": {name: phase_metrics(items) for name, items in phases.items()},
    }


def write_metrics(run) -> Path:
    directory = _telemetry_path(run).parent
    target = directory / METRICS_FILE
    target.write_text(json.dumps(compute_metrics(directory), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def flatten(metrics: dict, phases=None) -> dict:
    """One flat row: run info, overall metrics, and ``<phase>.<metric>`` for the chosen phases."""
    row = dict(metrics["run"])
    row.update(metrics["overall"])
    for phase, values in metrics["phases"].items():
        if phases is None or phase in phases:
            row.update({f"{phase}.{key}": value for key, value in values.items()})
    return row
