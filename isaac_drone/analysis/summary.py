"""Per-phase console summary of one run's telemetry.jsonl."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def latest_telemetry(runs_dir) -> Path:
    candidates = list(Path(runs_dir).glob("*/telemetry.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No run with telemetry.jsonl under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_run(path) -> str:
    """Table of position error, tilt, rates, motor speeds and saturation per mission phase.

    ``path`` is a run directory or its telemetry.jsonl.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "telemetry.jsonl"
    phases, finished = {}, None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") in ("finished", "aborted"):
                finished = record
                continue
            basic = record.get("basic")
            if basic is None:
                continue
            w, x, y, z = basic["quaternion_wxyz"]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
            phases.setdefault(record.get("mission_phase", "?"), []).append((
                basic["sample_time_s"], basic["position_error_norm_m"], tilt,
                float(np.linalg.norm(record["post_step_state"]["angular_velocity_b"])),
                min(basic["motor_speed_rps"]), max(basic["motor_speed_rps"]),
                bool(record.get("allocation_saturated")), basic["position_w_m"], basic["target_position_w_m"]))
    lines = [f"run: {path.parent}",
             f"{'phase':8} {'t[s]':>11} {'err_max':>8} {'err_rms':>8} {'err_end':>8} {'tilt_max':>8} "
             f"{'w_max':>6} {'rps_min':>7} {'rps_max':>7} {'sat%':>5}"]
    rows = []
    for name, rows in phases.items():
        t, err, tilt, omega, lo, hi, sat = (np.array([r[i] for r in rows]) for i in range(7))
        lines.append(f"{name:8} {t[0]:5.2f}-{t[-1]:5.2f} {err.max():8.3f} {math.sqrt((err**2).mean()):8.3f} "
                     f"{err[-1]:8.3f} {tilt.max():7.1f}° {omega.max():6.2f} {lo.min():7.1f} {hi.max():7.1f} "
                     f"{100*sat.mean():5.1f}")
    if rows:
        last = rows[-1]
        lines.append(f"final position  : {np.round(last[7], 3)}  target: {np.round(last[8], 3)}")
    if finished:
        keys = ("event", "success", "physics_steps", "simulated_time_s", "error")
        lines.append("end event: " + json.dumps({key: finished.get(key) for key in keys}))
        if finished.get("mission"):
            mission = {key: (round(value, 3) if isinstance(value, float) else value)
                       for key, value in finished["mission"].items()}
            lines.append(f"mission: {mission}")
    return "\n".join(lines)
