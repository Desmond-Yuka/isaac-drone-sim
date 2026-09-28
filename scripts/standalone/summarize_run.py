#!/usr/bin/env python3
"""Per-phase summary of a run's telemetry.jsonl (latest run under runs/ by default)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(RUNS.glob("*/telemetry.jsonl"), key=lambda p: p.stat().st_mtime)
print("run:", path.parent)
phases, finished = {}, None
for line in path.open():
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

print(f"{'phase':8} {'t[s]':>11} {'err_max':>8} {'err_rms':>8} {'err_end':>8} {'tilt_max':>8} {'w_max':>6} {'rps_min':>7} {'rps_max':>7} {'sat%':>5}")
for name, rows in phases.items():
    t, err, tilt, omega, lo, hi, sat = (np.array([r[i] for r in rows]) for i in range(7))
    print(f"{name:8} {t[0]:5.2f}-{t[-1]:5.2f} {err.max():8.3f} {math.sqrt((err**2).mean()):8.3f} {err[-1]:8.3f} "
          f"{tilt.max():7.1f}° {omega.max():6.2f} {lo.min():7.1f} {hi.max():7.1f} {100*sat.mean():5.1f}")
last = rows[-1]
print("final position  :", np.round(last[7], 3), " target:", np.round(last[8], 3))
if finished:
    print("end event:", json.dumps({k: finished.get(k) for k in ("event", "success", "physics_steps", "simulated_time_s", "error")}))
    if finished.get("mission"):
        m = finished["mission"]
        print("mission:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
