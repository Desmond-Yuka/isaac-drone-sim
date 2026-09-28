"""Analytic C4 delay/takeoff/constant-radius 3D helix/hold references.

Each motion phase is a fixed geometric path traversed by a progress time law
(see ``timing``). A configured duration selects the ninth-degree Hermite law,
which satisfies endpoint position plus four zero time derivatives; it is not a
claimed minimum-snap optimum. A null duration leaves the phase to the
minimum-time planner, which assigns an accelerate/cruise/decelerate law from
the actual vehicle limits after reset. Actual flight remains subject to the
controller, motor dynamics, contacts, disturbances and actuator constraints.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np

from isaac_drone.core.rotations import quaternion_to_matrix
from isaac_drone.core.types import TrajectorySetpoint, VehicleState
from isaac_drone.core.validation import finite_array, finite_scalar

from .timing import CruiseTimeLaw, HermiteTimeLaw

_FIELDS = {
    "start_delay_s", "takeoff_height_m", "takeoff_duration_s", "radius_m", "turns",
    "climb_height_m", "spiral_duration_s", "initial_phase_rad", "yaw_mode", "yaw_offset_rad",
    "initial_speed_tolerance_m_s", "initial_angular_speed_tolerance_rad_s",
}
# Optional: required only when a phase duration is null (minimum-time planning).
_OPTIONAL_FIELDS = {"thrust_utilization"}
# Phase name -> duration field. A null duration is planned from vehicle limits.
_PHASE_DURATIONS = {"takeoff": "takeoff_duration_s", "spiral": "spiral_duration_s"}
_POSITIVE_FIELDS = {
    "takeoff_height_m", "takeoff_duration_s", "radius_m", "climb_height_m", "spiral_duration_s",
}
_NONNEGATIVE_FIELDS = {
    "start_delay_s", "initial_speed_tolerance_m_s", "initial_angular_speed_tolerance_rad_s",
}


def validate_spiral_config(config: Mapping) -> None:
    """Reject missing/unknown fields and nonfinite or incompatible SI values.

    Turns may be any nonzero signed real number; fractional turns are allowed.
    Positive turns are counterclockwise viewed from world +Z. Initial speed
    tolerances are acceptance thresholds, not replacements for state values.
    takeoff_duration_s / spiral_duration_s may be null: that phase is then
    planned for minimum time, and thrust_utilization in (0.5, 1] is required.
    """
    if not isinstance(config, Mapping):
        raise ValueError("trajectory.spiral must be a mapping")
    if not all(isinstance(key, str) for key in config):
        raise ValueError("trajectory.spiral keys must be strings")
    missing, unknown = _FIELDS - set(config), set(config) - _FIELDS - _OPTIONAL_FIELDS
    if missing or unknown:
        raise ValueError(f"trajectory.spiral: missing={sorted(missing)}, unknown={sorted(unknown)}")
    planned = {name for name in _PHASE_DURATIONS.values() if config[name] is None}
    numeric = _FIELDS - {"yaw_mode"} - planned
    for name in numeric:
        if not isinstance(config[name], Real):
            raise ValueError(f"trajectory.spiral.{name} must be a real number")
    values = {name: finite_scalar(config[name], f"trajectory.spiral.{name}") for name in numeric}
    for name in _POSITIVE_FIELDS - planned:
        if values[name] <= 0:
            raise ValueError(f"trajectory.spiral.{name} must be positive")
    for name in _NONNEGATIVE_FIELDS:
        if values[name] < 0:
            raise ValueError(f"trajectory.spiral.{name} must be nonnegative")
    if values["turns"] == 0:
        raise ValueError("trajectory.spiral.turns must be nonzero")
    if config["yaw_mode"] not in ("fixed", "tangent"):
        raise ValueError("trajectory.spiral.yaw_mode must be fixed or tangent")
    utilization = config.get("thrust_utilization")
    if utilization is None:
        if planned:
            raise ValueError("trajectory.spiral: null phase durations are planned for minimum time and "
                             "require thrust_utilization")
    else:
        utilization = finite_scalar(utilization, "trajectory.spiral.thrust_utilization")
        if not 0.5 < utilization <= 1.0:
            raise ValueError("trajectory.spiral.thrust_utilization must satisfy 0.5 < value <= 1")
    angle = 2.0 * math.pi * values["turns"]
    angle_end = values["initial_phase_rad"] + angle
    if not np.isfinite(angle_end):
        raise ValueError("Spiral mission duration/angle exceeds finite numerical range")
    if not planned:
        boundaries = np.cumsum([values["start_delay_s"], values["takeoff_duration_s"],
                                values["spiral_duration_s"]])
        if not np.all(np.isfinite(boundaries)):
            raise ValueError("Spiral mission duration/angle exceeds finite numerical range")
        if boundaries[1] <= boundaries[0] or boundaries[2] <= boundaries[1]:
            raise ValueError("Spiral phase durations must remain distinguishable in floating-point time")
    # Reject unrepresentable differentiation scales instead of emitting NaN/Inf
    # during flight. These are representability checks, not invented vehicle limits.
    try:
        scales = [angle, values["takeoff_height_m"] + values["climb_height_m"]]
        for name in set(_PHASE_DURATIONS.values()) - planned:
            inverse = 1.0 / values[name]
            scales.extend(inverse**order for order in range(1, 5))
        if not np.all(np.isfinite(scales)):
            raise ValueError("Spiral parameters produce nonfinite derivative scales")
    except OverflowError as error:
        raise ValueError("Spiral parameters produce nonfinite derivative scales") from error


class SpiralTrajectory:
    """Smooth takeoff followed by a constant-radius 3D helix and endpoint hold.

    The historical SpiralTrajectory name is retained for compatibility; this
    curve is a cylindrical helix, not an expanding planar spiral. The identical
    implementation is also exported as HelixTrajectory.

    Times passed to sample/phase are absolute simulation times. reset establishes
    the time origin from initial_state.time_s, the actual CoM position, and the
    actual projected body-X heading. Sampling is pure and order-independent;
    no trajectory state is advanced by logging or repeated calls.

    During delay the reference holds the initial pose. During takeoff, heading
    smoothly reaches either initial_yaw+yaw_offset (fixed mode) or the nearest
    equivalent tangent heading (tangent mode). Helical yaw is then unwrapped.
    Initial nonzero rates exceeding the explicit tolerances are rejected; a
    caller must settle the vehicle or use a different transition planner.

    A phase with a null configured duration has no time law until
    ``set_time_laws`` (normally the minimum-time planner) supplies one after
    every reset; geometry is available before that, sampling is not.
    """

    def __init__(self, config: Mapping):
        validate_spiral_config(config)
        self.config = deepcopy(dict(config))
        for name in _FIELDS - {"yaw_mode"} | _OPTIONAL_FIELDS:
            if self.config.get(name) is not None:
                self.config[name] = float(self.config[name])
        self._delay = self.config["start_delay_s"]
        self._turn_angle = 2.0*math.pi*self.config["turns"]
        self._configured_laws = {
            phase: None if self.config[field] is None else HermiteTimeLaw(self.config[field])
            for phase, field in _PHASE_DURATIONS.items()}
        self._laws = dict(self._configured_laws)
        self._initialized = False
        self._timed = False

    @property
    def planned_phases(self) -> tuple[str, ...]:
        """Phases whose null configured duration must be planned after reset."""
        return tuple(phase for phase, law in self._configured_laws.items() if law is None)

    @property
    def time_laws(self) -> dict:
        return dict(self._laws)

    @property
    def mission_duration_s(self) -> float:
        """Delay + takeoff + spiral duration; endpoint hold has no time limit."""
        if any(law is None for law in self._laws.values()):
            self._require_timing()  # raises with the planning hint
        return self._delay+self._laws["takeoff"].duration_s+self._laws["spiral"].duration_s

    @property
    def phase_boundaries_s(self) -> list[float]:
        """Absolute starts of delay, takeoff, spiral and hold."""
        self._require_timing()
        return self._absolute_boundaries.tolist()

    @property
    def endpoint_position_w(self) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("reset spiral trajectory before requesting its endpoint")
        return self._endpoint.copy()

    @property
    def takeoff_yaw_change_rad(self) -> float:
        self._require_geometry()
        return self._yaw_spiral-self._yaw_initial

    @property
    def turn_angle_rad(self) -> float:
        return self._turn_angle

    @property
    def path_lengths_m(self) -> dict:
        """Path length of each motion phase: speed = length * progress rate."""
        return {"takeoff": abs(self.config["takeoff_height_m"]),
                "spiral": math.hypot(self.config["radius_m"]*self._turn_angle, self.config["climb_height_m"])}

    def segments(self) -> list[tuple[str, str, float, float]]:
        """(phase, segment, absolute start, absolute end) of every time-law segment."""
        self._require_timing()
        result = []
        for index, phase in enumerate(_PHASE_DURATIONS):
            start = self._absolute_boundaries[index+1]
            result.extend((phase, name, start+begin, start+end) for name, begin, end in self._laws[phase].segments)
        return result

    def timing_report(self) -> dict:
        """Phase boundaries plus each phase's law and peak speed along its path."""
        self._require_timing()
        lengths = self.path_lengths_m
        phases = {}
        for phase in _PHASE_DURATIONS:
            law = self._laws[phase]
            phases[phase] = {**law.describe(), "planned": phase in self.planned_phases,
                             "path_length_m": lengths[phase], "peak_speed_m_s": lengths[phase]*law.peak_rate}
        return {"phase_boundaries_s": self.phase_boundaries_s, "mission_duration_s": self.mission_duration_s,
                "thrust_utilization": self.config.get("thrust_utilization"), "phases": phases}

    def set_time_laws(self, **laws) -> None:
        """Assign time laws to the planned (null-duration) phases after reset."""
        self._require_geometry()
        unknown = set(laws)-set(self.planned_phases)
        if unknown:
            raise ValueError(f"Only null-duration phases can be planned; got {sorted(unknown)}")
        for phase, law in laws.items():
            if not isinstance(law, (HermiteTimeLaw, CruiseTimeLaw)):
                raise TypeError(f"{phase} time law must be a HermiteTimeLaw or CruiseTimeLaw")
            self._laws[phase] = law
        self._timed = False
        if all(law is not None for law in self._laws.values()):
            self._finalize_timing()

    def reset(self, initial_state: VehicleState) -> None:
        self._initialized = self._timed = False
        self._laws = dict(self._configured_laws)
        position = finite_array(initial_state.position_w, (3,), "initial CoM position")
        velocity = finite_array(initial_state.linear_velocity_w, (3,), "initial CoM velocity")
        omega = finite_array(initial_state.angular_velocity_b, (3,), "initial angular velocity")
        if np.linalg.norm(velocity) > self.config["initial_speed_tolerance_m_s"]:
            raise ValueError("Spiral reset requires settled CoM speed within initial_speed_tolerance_m_s")
        if np.linalg.norm(omega) > self.config["initial_angular_speed_tolerance_rad_s"]:
            raise ValueError("Spiral reset requires settled angular speed within initial_angular_speed_tolerance_rad_s")
        rotation = quaternion_to_matrix(initial_state.quaternion_wxyz)
        if np.linalg.norm(rotation[:2, 0]) < 1e-10:
            raise ValueError("Initial body-X heading is undefined at a vertical body-X orientation")
        origin = finite_scalar(initial_state.time_s, "initial_state.time_s")
        if origin < 0:
            raise ValueError("Initial simulation time must be nonnegative")
        if self._delay > 0 and origin+self._delay <= origin:
            raise ValueError("Start delay is not representable at the initial simulation time")
        yaw_initial = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        phase = float(self.config["initial_phase_rad"])
        radius = float(self.config["radius_m"])
        center = position[:2]-radius*np.array([math.cos(phase), math.sin(phase)])
        takeoff_position = position+np.array([0., 0., self.config["takeoff_height_m"]])
        final_phase = phase+self._turn_angle
        endpoint = np.r_[center+radius*np.array([math.cos(final_phase), math.sin(final_phase)]),
                         takeoff_position[2]+self.config["climb_height_m"]]
        if self.config["yaw_mode"] == "fixed":
            yaw_spiral = yaw_initial+float(self.config["yaw_offset_rad"])
        else:
            tangent = phase+math.copysign(math.pi/2, self._turn_angle)+float(self.config["yaw_offset_rad"])
            difference = tangent-yaw_initial
            if not np.isfinite(difference):
                raise ValueError("Tangent heading exceeds finite numerical range")
            yaw_spiral = yaw_initial+math.atan2(math.sin(difference), math.cos(difference))
        yaw_final = yaw_spiral + (self._turn_angle if self.config["yaw_mode"] == "tangent" else 0.0)
        if not np.all(np.isfinite(np.r_[center, takeoff_position, endpoint, yaw_spiral, yaw_final])):
            raise ValueError("Mission positions or headings exceed finite numerical range")
        self._time_origin = origin
        self._initial_position = position
        self._takeoff_position = takeoff_position
        self._center_xy = center
        self._endpoint = endpoint
        self._yaw_initial, self._yaw_spiral, self._yaw_final = yaw_initial, yaw_spiral, yaw_final
        self._initialized = True
        if not self.planned_phases:
            self._finalize_timing()

    def _finalize_timing(self) -> None:
        origin = self._time_origin
        takeoff_end = self._delay+self._laws["takeoff"].duration_s
        relative_boundaries = (0.0, self._delay, takeoff_end, takeoff_end+self._laws["spiral"].duration_s)
        absolute = np.array([origin+value for value in relative_boundaries])
        if not np.all(np.isfinite(absolute)) or absolute[2] <= absolute[1] or absolute[3] <= absolute[2]:
            raise ValueError("Mission phases are not representable at the initial simulation time")
        self._absolute_boundaries = absolute
        self._timed = True

    def _require_geometry(self) -> None:
        if not self._initialized:
            raise RuntimeError("reset spiral trajectory before sampling")

    def _require_timing(self) -> None:
        self._require_geometry()
        if not self._timed:
            raise RuntimeError(f"Plan the null-duration phases {list(self.planned_phases)} "
                               "(minimum-time planner) after reset before sampling")

    def _elapsed(self, time_s: float) -> float:
        self._require_timing()
        time = finite_scalar(time_s, "trajectory time_s")
        if time < self._time_origin:
            raise ValueError("Trajectory sample time precedes its reset time")
        return time-self._time_origin

    def phase(self, time_s: float) -> str:
        """Return delay, takeoff, spiral, or hold at an absolute simulation time."""
        self._elapsed(time_s)
        if time_s < self._absolute_boundaries[1]:
            return "delay"
        if time_s < self._absolute_boundaries[2]:
            return "takeoff"
        if time_s < self._absolute_boundaries[3]:
            return "spiral"
        return "hold"

    def takeoff_setpoint(self, progress: np.ndarray) -> TrajectorySetpoint:
        """Vertical takeoff reference for progress [sigma, sigma', ..., sigma''''] in [0, 1]."""
        self._require_geometry()
        derivatives = np.zeros((4, 3))
        position = self._initial_position+np.array([0., 0., self.config["takeoff_height_m"]*progress[0]])
        derivatives[:, 2] = float(self.config["takeoff_height_m"])*progress[1:]
        yaw_change = self._yaw_spiral-self._yaw_initial
        yaw = self._yaw_initial+yaw_change*progress[0]
        yaw_rate, yaw_acceleration = yaw_change*progress[1:3]
        return TrajectorySetpoint(position_w=position, velocity_w=derivatives[0], acceleration_w=derivatives[1],
                                  yaw_rad=float(yaw), yaw_rate_rad_s=float(yaw_rate),
                                  yaw_acceleration_rad_s2=float(yaw_acceleration),
                                  jerk_w=derivatives[2], snap_w=derivatives[3])

    def spiral_setpoint(self, progress: np.ndarray) -> TrajectorySetpoint:
        """Helix reference for progress [sigma, sigma', ..., sigma''''] in [0, 1]."""
        self._require_geometry()
        derivatives = np.zeros((4, 3))
        yaw_rate = yaw_acceleration = 0.0
        phase = float(self.config["initial_phase_rad"])+self._turn_angle*progress[0]
        theta1, theta2, theta3, theta4 = self._turn_angle*np.asarray(progress[1:])
        radial = np.array([math.cos(phase), math.sin(phase), 0.])
        tangent = np.array([-math.sin(phase), math.cos(phase), 0.])
        radius = float(self.config["radius_m"])
        position = np.r_[self._center_xy, self._takeoff_position[2]]+radius*radial
        position[2] += float(self.config["climb_height_m"])*progress[0]
        if progress[0] == 0:
            position = self._takeoff_position.copy()
        derivatives[0] = radius*theta1*tangent
        derivatives[1] = radius*(theta2*tangent-theta1**2*radial)
        derivatives[2] = radius*((theta3-theta1**3)*tangent-3.0*theta1*theta2*radial)
        derivatives[3] = radius*((theta4-6.0*theta1**2*theta2)*tangent
                                 +(theta1**4-3.0*theta2**2-4.0*theta1*theta3)*radial)
        derivatives[:, 2] = float(self.config["climb_height_m"])*np.asarray(progress[1:])
        yaw = self._yaw_spiral
        if self.config["yaw_mode"] == "tangent":
            yaw += self._turn_angle*progress[0]
            yaw_rate, yaw_acceleration = theta1, theta2
        return TrajectorySetpoint(position_w=position, velocity_w=derivatives[0], acceleration_w=derivatives[1],
                                  yaw_rad=float(yaw), yaw_rate_rad_s=float(yaw_rate),
                                  yaw_acceleration_rad_s2=float(yaw_acceleration),
                                  jerk_w=derivatives[2], snap_w=derivatives[3])

    def sample(self, time_s: float) -> TrajectorySetpoint:
        mission_phase = self.phase(time_s)
        if mission_phase == "takeoff":
            return self.takeoff_setpoint(self._laws["takeoff"].derivatives(time_s-self._absolute_boundaries[1]))
        if mission_phase == "spiral":
            return self.spiral_setpoint(self._laws["spiral"].derivatives(time_s-self._absolute_boundaries[2]))
        if mission_phase == "delay":
            position, yaw = self._initial_position.copy(), self._yaw_initial
        else:
            position, yaw = self._endpoint.copy(), self._yaw_final
        zero = np.zeros(3)
        return TrajectorySetpoint(position_w=position, velocity_w=zero.copy(), acceleration_w=zero.copy(),
                                  yaw_rad=float(yaw), yaw_rate_rad_s=0.0, yaw_acceleration_rad_s2=0.0,
                                  jerk_w=zero.copy(), snap_w=zero.copy())


# Preserve the historical factory/config import while making the geometry explicit.
HelixTrajectory = SpiralTrajectory
