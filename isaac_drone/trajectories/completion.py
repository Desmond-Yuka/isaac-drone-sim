"""Measured endpoint-hover completion, independent of reference elapsed time."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from isaac_drone.core.rotations import quaternion_to_matrix
from isaac_drone.core.types import TrajectorySetpoint, VehicleState


def validate_completion_config(config):
    fields = {"position_tolerance_m", "speed_tolerance_m_s", "yaw_tolerance_rad",
              "angular_speed_tolerance_rad_s", "dwell_time_s", "max_hold_time_s"}
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ValueError(f"trajectory.completion requires exactly {sorted(fields)}")
    for name, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"trajectory.completion.{name} must be finite and positive")
    if config["yaw_tolerance_rad"] > math.pi:
        raise ValueError("yaw_tolerance_rad cannot exceed pi")
    if config["max_hold_time_s"] < config["dwell_time_s"]:
        raise ValueError("max_hold_time_s must be at least dwell_time_s")


class SpiralMissionMonitor:
    """Require a continuous measured dwell; success is revoked if hover is lost.

    A deadline is latched as a failure if no qualifying dwell has occurred by
    that time. Achieving the reference endpoint alone is never success. The
    runner continues applying endpoint feedback after successful completion.
    """
    def __init__(self, config):
        validate_completion_config(config)
        self.config = dict(config)
        self.reset()

    def reset(self):
        self._last_time = None
        self._hold_start = None
        self._within_since = None
        self._ever_achieved = False
        self._timed_out = False
        self.status = None

    def update(self, state: VehicleState, target: TrajectorySetpoint, phase: str):
        time = state.time_s
        if self._last_time is not None and time <= self._last_time:
            raise ValueError("mission state timestamps must strictly increase")
        self._last_time = time
        rotation = quaternion_to_matrix(state.quaternion_wxyz)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        yaw_error = math.atan2(math.sin(target.yaw_rad-yaw), math.cos(target.yaw_rad-yaw))
        errors = {"position_error_m": float(np.linalg.norm(target.position_w-state.position_w)),
                  "speed_m_s": float(np.linalg.norm(state.linear_velocity_w)),
                  "yaw_error_rad": float(abs(yaw_error)),
                  "angular_speed_rad_s": float(np.linalg.norm(state.angular_velocity_b))}
        tolerances = ("position_tolerance_m", "speed_tolerance_m_s", "yaw_tolerance_rad",
                      "angular_speed_tolerance_rad_s")
        within = phase == "hold" and all(v <= self.config[k] for v, k in zip(errors.values(), tolerances))
        if phase != "hold":
            self._hold_start = self._within_since = None
            self._ever_achieved = self._timed_out = False
        elif self._hold_start is None:
            self._hold_start = time
        if within:
            if self._within_since is None:
                self._within_since = time
        else:
            self._within_since = None
        dwell = 0.0 if self._within_since is None else time-self._within_since
        achieved = within and dwell + 1e-12 >= self.config["dwell_time_s"]
        hold_time = 0.0 if self._hold_start is None else time-self._hold_start
        # A late observation must not retroactively turn an already expired
        # deadline into success. The same-sample equality is allowed.
        completion_time = None if self._within_since is None else self._within_since+self.config["dwell_time_s"]
        deadline = None if self._hold_start is None else self._hold_start+self.config["max_hold_time_s"]
        if achieved and completion_time <= deadline + 1e-12 and not self._timed_out:
            self._ever_achieved = True
        if deadline is not None and time >= deadline and not self._ever_achieved:
            self._timed_out = True
        self.status = {"phase": phase, **errors, "within_tolerances": bool(within),
                       "continuous_dwell_s": dwell, "hold_elapsed_s": hold_time,
                       "achieved": bool(achieved and not self._timed_out),
                       "ever_achieved": self._ever_achieved, "timed_out": self._timed_out}
        return dict(self.status)
