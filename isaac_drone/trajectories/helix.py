"""Vertical takeoff followed by a constant-radius 3D cylindrical helix, then endpoint hold.

The cylinder axis is offset so the helix starts exactly where the vertical
takeoff ends. Positive turns are counterclockwise viewed from world +Z;
negative and fractional turns are allowed. In ``tangent`` yaw mode the heading
follows the helix tangent (after turning to it smoothly during takeoff); in
``fixed`` mode it keeps the initial heading plus ``yaw_offset_rad``.
See docs/spiral_math.md for the derivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from isaac_drone.core.types import TrajectorySetpoint

from .base import TRAJECTORIES, Segment, SegmentedTrajectory


@dataclass(frozen=True)
class HelixParams:
    """``trajectory: {kind: helix}``. Durations may be null: minimum time from the actual limits.

    thrust_utilization in (0.5, 1] is required when any duration is null: the
    planned nominal rotor thrust stays within that central fraction of each
    rotor's thrust range, leaving the rest to feedback.
    """

    takeoff_height_m: float
    takeoff_duration_s: float | None
    radius_m: float
    turns: float
    climb_height_m: float
    helix_duration_s: float | None
    initial_phase_rad: float
    yaw_mode: Literal["fixed", "tangent"]
    yaw_offset_rad: float
    rest_speed_tolerance_m_s: float
    rest_angular_speed_tolerance_rad_s: float
    thrust_utilization: float | None = None

    def __post_init__(self):
        for name in ("takeoff_height_m", "radius_m", "climb_height_m", "takeoff_duration_s", "helix_duration_s"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("rest_speed_tolerance_m_s", "rest_angular_speed_tolerance_rad_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.turns == 0:
            raise ValueError("turns must be nonzero")
        planned = self.takeoff_duration_s is None or self.helix_duration_s is None
        if self.thrust_utilization is None:
            if planned:
                raise ValueError("null durations are planned for minimum time and require thrust_utilization")
        elif not 0.5 < self.thrust_utilization <= 1.0:
            raise ValueError("thrust_utilization must satisfy 0.5 < value <= 1")
        # Reject unrepresentable differentiation scales instead of emitting NaN/Inf in flight.
        angle = 2.0 * math.pi * self.turns
        scales = [angle, self.initial_phase_rad + angle, self.takeoff_height_m + self.climb_height_m]
        try:
            for duration in (self.takeoff_duration_s, self.helix_duration_s):
                if duration is not None:
                    scales.extend((1.0 / duration) ** order for order in range(1, 5))
        except OverflowError as error:
            raise ValueError("helix parameters produce nonfinite derivative scales") from error
        if not np.all(np.isfinite(scales)):
            raise ValueError("helix parameters produce nonfinite derivative scales")


class HelixTrajectory(SegmentedTrajectory):
    """Segments ``takeoff`` (vertical line, yaw turns to the helix heading) and ``helix``."""

    def __init__(self, params: HelixParams):
        super().__init__(thrust_utilization=params.thrust_utilization,
                         rest_speed_tolerance_m_s=params.rest_speed_tolerance_m_s,
                         rest_angular_speed_tolerance_rad_s=params.rest_angular_speed_tolerance_rad_s)
        self.params = params
        self.turn_angle_rad = 2.0 * math.pi * params.turns

    def _build(self, position_w, yaw_rad):
        p = self.params
        phase = p.initial_phase_rad
        center = position_w[:2] - p.radius_m * np.array([math.cos(phase), math.sin(phase)])
        takeoff_end = position_w + np.array([0., 0., p.takeoff_height_m])
        final_phase = phase + self.turn_angle_rad
        endpoint = np.r_[center + p.radius_m * np.array([math.cos(final_phase), math.sin(final_phase)]),
                         takeoff_end[2] + p.climb_height_m]
        if p.yaw_mode == "fixed":
            yaw_helix = yaw_rad + p.yaw_offset_rad
        else:
            tangent = phase + math.copysign(math.pi / 2, self.turn_angle_rad) + p.yaw_offset_rad
            difference = tangent - yaw_rad
            if not np.isfinite(difference):
                raise ValueError("Tangent heading exceeds finite numerical range")
            yaw_helix = yaw_rad + math.atan2(math.sin(difference), math.cos(difference))
        helix_yaw_change = self.turn_angle_rad if p.yaw_mode == "tangent" else 0.0
        if not np.all(np.isfinite(np.r_[center, takeoff_end, yaw_helix])):
            raise ValueError("Helix positions or headings exceed finite numerical range")
        self._center_xy, self._takeoff_end = center, takeoff_end
        self._yaw_initial, self._yaw_helix = yaw_rad, yaw_helix
        takeoff = Segment("takeoff", self._takeoff_setpoint, abs(p.takeoff_height_m), yaw_helix - yaw_rad,
                          p.takeoff_duration_s, revolutions=0.0, hermite_samples=33)
        helix = Segment("helix", self._helix_setpoint, math.hypot(p.radius_m * self.turn_angle_rad, p.climb_height_m),
                        helix_yaw_change, p.helix_duration_s, revolutions=abs(p.turns), hermite_samples=65)
        return [takeoff, helix], endpoint, yaw_helix + helix_yaw_change

    def _takeoff_setpoint(self, progress) -> TrajectorySetpoint:
        """Vertical takeoff reference for progress [sigma, sigma', ..., sigma''''] in [0, 1]."""
        height = self.params.takeoff_height_m
        derivatives = np.zeros((4, 3))
        position = self._start_position + np.array([0., 0., height * progress[0]])
        derivatives[:, 2] = float(height) * progress[1:]
        yaw_change = self._yaw_helix - self._yaw_initial
        yaw = self._yaw_initial + yaw_change * progress[0]
        yaw_rate, yaw_acceleration = yaw_change * progress[1:3]
        return TrajectorySetpoint(position_w=position, velocity_w=derivatives[0], acceleration_w=derivatives[1],
                                  yaw_rad=float(yaw), yaw_rate_rad_s=float(yaw_rate),
                                  yaw_acceleration_rad_s2=float(yaw_acceleration),
                                  jerk_w=derivatives[2], snap_w=derivatives[3])

    def _helix_setpoint(self, progress) -> TrajectorySetpoint:
        """Helix reference for progress [sigma, sigma', ..., sigma''''] in [0, 1]."""
        p = self.params
        derivatives = np.zeros((4, 3))
        yaw_rate = yaw_acceleration = 0.0
        phase = float(p.initial_phase_rad) + self.turn_angle_rad * progress[0]
        theta1, theta2, theta3, theta4 = self.turn_angle_rad * np.asarray(progress[1:])
        radial = np.array([math.cos(phase), math.sin(phase), 0.])
        tangent = np.array([-math.sin(phase), math.cos(phase), 0.])
        radius = float(p.radius_m)
        position = np.r_[self._center_xy, self._takeoff_end[2]] + radius * radial
        position[2] += float(p.climb_height_m) * progress[0]
        if progress[0] == 0:
            position = self._takeoff_end.copy()
        derivatives[0] = radius * theta1 * tangent
        derivatives[1] = radius * (theta2 * tangent - theta1**2 * radial)
        derivatives[2] = radius * ((theta3 - theta1**3) * tangent - 3.0 * theta1 * theta2 * radial)
        derivatives[3] = radius * ((theta4 - 6.0 * theta1**2 * theta2) * tangent
                                   + (theta1**4 - 3.0 * theta2**2 - 4.0 * theta1 * theta3) * radial)
        derivatives[:, 2] = float(p.climb_height_m) * np.asarray(progress[1:])
        yaw = self._yaw_helix
        if p.yaw_mode == "tangent":
            yaw += self.turn_angle_rad * progress[0]
            yaw_rate, yaw_acceleration = theta1, theta2
        return TrajectorySetpoint(position_w=position, velocity_w=derivatives[0], acceleration_w=derivatives[1],
                                  yaw_rad=float(yaw), yaw_rate_rad_s=float(yaw_rate),
                                  yaw_acceleration_rad_s2=float(yaw_acceleration),
                                  jerk_w=derivatives[2], snap_w=derivatives[3])


@TRAJECTORIES.register("helix", params=HelixParams)
def build_helix(params: HelixParams) -> HelixTrajectory:
    """Vertical takeoff, constant-radius 3D helix climb, endpoint hold."""
    return HelixTrajectory(params)
