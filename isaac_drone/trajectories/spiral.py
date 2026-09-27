"""Analytic C4 delay/takeoff/constant-radius 3D helix/hold references.

The ninth-degree progress polynomial satisfies endpoint position plus four
zero time derivatives; it is a Hermite interpolant, not a claimed minimum-snap
optimum. Actual flight remains subject to the controller, motor dynamics,
contacts, disturbances and actuator constraints.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from isaac_drone.control.math import finite_array, finite_scalar, quaternion_to_matrix
from isaac_drone.types import TrajectorySetpoint, VehicleState


_FIELDS = {
    "start_delay_s", "takeoff_height_m", "takeoff_duration_s", "radius_m", "turns",
    "climb_height_m", "spiral_duration_s", "initial_phase_rad", "yaw_mode", "yaw_offset_rad",
    "initial_speed_tolerance_m_s", "initial_angular_speed_tolerance_rad_s",
}
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
    """
    if not isinstance(config, Mapping):
        raise ValueError("trajectory.spiral must be a mapping")
    if not all(isinstance(key, str) for key in config):
        raise ValueError("trajectory.spiral keys must be strings")
    missing, unknown = _FIELDS - set(config), set(config) - _FIELDS
    if missing or unknown:
        raise ValueError(f"trajectory.spiral: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for name in _FIELDS - {"yaw_mode"}:
        if not isinstance(config[name], Real):
            raise ValueError(f"trajectory.spiral.{name} must be a real number")
    values = {name: finite_scalar(config[name], f"trajectory.spiral.{name}")
              for name in _FIELDS - {"yaw_mode"}}
    for name in _POSITIVE_FIELDS:
        if values[name] <= 0:
            raise ValueError(f"trajectory.spiral.{name} must be positive")
    for name in _NONNEGATIVE_FIELDS:
        if values[name] < 0:
            raise ValueError(f"trajectory.spiral.{name} must be nonnegative")
    if values["turns"] == 0:
        raise ValueError("trajectory.spiral.turns must be nonzero")
    if config["yaw_mode"] not in ("fixed", "tangent"):
        raise ValueError("trajectory.spiral.yaw_mode must be fixed or tangent")
    angle = 2.0 * math.pi * values["turns"]
    angle_end = values["initial_phase_rad"] + angle
    boundaries = np.cumsum([values["start_delay_s"], values["takeoff_duration_s"],
                             values["spiral_duration_s"]])
    if not np.all(np.isfinite(boundaries)) or not np.isfinite(angle_end):
        raise ValueError("Spiral mission duration/angle exceeds finite numerical range")
    if boundaries[1] <= boundaries[0] or boundaries[2] <= boundaries[1]:
        raise ValueError("Spiral phase durations must remain distinguishable in floating-point time")
    # Reject unrepresentable differentiation scales instead of emitting NaN/Inf
    # during flight. These are representability checks, not invented vehicle limits.
    try:
        scales = [angle, values["takeoff_height_m"] + values["climb_height_m"]]
        for name in ("takeoff_duration_s", "spiral_duration_s"):
            inverse = 1.0 / values[name]
            scales.extend(inverse**order for order in range(1, 5))
        if not np.all(np.isfinite(scales)):
            raise ValueError("Spiral parameters produce nonfinite derivative scales")
    except OverflowError as error:
        raise ValueError("Spiral parameters produce nonfinite derivative scales") from error


def _progress_derivatives(u: float, duration_s: float) -> np.ndarray:
    """Return S and d^kS/dt^k (k=1..4), exactly zero outside transition.

    Symmetry S(1-u)=1-S(u) avoids cancellation in the ninth-degree position
    polynomial near the endpoint. Factored derivative forms retain their
    endpoint zeros without subtracting large polynomial terms.
    """
    if u <= 0:
        return np.zeros(5)
    if u >= 1:
        return np.array([1., 0., 0., 0., 0.])
    q = min(u, 1.0-u)
    position = q**5 * (126.0 + q*(-420.0 + q*(540.0 + q*(-315.0 + 70.0*q))))
    if u > .5:
        position = 1.0-position
    v = 1.0-u
    first = 630.0*u**4*v**4
    second = 2520.0*u**3*v**3*(1.0-2.0*u)
    third = 2520.0*u**2*v**2*(3.0-14.0*u+14.0*u*u)
    fourth = 15120.0*u*v*(1.0-2.0*u)*(1.0-7.0*u+7.0*u*u)
    inverse = 1.0/duration_s
    return np.array([position, first*inverse, second*inverse**2, third*inverse**3, fourth*inverse**4])


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
    """

    def __init__(self, config: Mapping):
        validate_spiral_config(config)
        self.config = deepcopy(dict(config))
        self.config.update({name: float(config[name]) for name in _FIELDS - {"yaw_mode"}})
        self._delay = float(config["start_delay_s"])
        self._takeoff_duration = float(config["takeoff_duration_s"])
        self._spiral_duration = float(config["spiral_duration_s"])
        self._takeoff_end = self._delay+self._takeoff_duration
        self._duration = self._takeoff_end+self._spiral_duration
        self._turn_angle = 2.0*math.pi*float(config["turns"])
        self._initialized = False

    @property
    def mission_duration_s(self) -> float:
        """Delay + takeoff + spiral duration; endpoint hold has no time limit."""
        return self._duration

    @property
    def endpoint_position_w(self) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("reset spiral trajectory before requesting its endpoint")
        return self._endpoint.copy()

    def reset(self, initial_state: VehicleState) -> None:
        self._initialized = False
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
        relative_boundaries = (0.0, self._delay, self._takeoff_end, self._duration)
        absolute = np.array([origin+value for value in relative_boundaries])
        if not np.all(np.isfinite(absolute)) or absolute[2] <= absolute[1] or absolute[3] <= absolute[2]:
            raise ValueError("Mission phases are not representable at the initial simulation time")
        if self._delay > 0 and absolute[1] <= absolute[0]:
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
        self._absolute_boundaries = absolute
        self._initial_position = position
        self._takeoff_position = takeoff_position
        self._center_xy = center
        self._endpoint = endpoint
        self._yaw_initial, self._yaw_spiral, self._yaw_final = yaw_initial, yaw_spiral, yaw_final
        self._initialized = True

    def _elapsed(self, time_s: float) -> float:
        if not self._initialized:
            raise RuntimeError("reset spiral trajectory before sampling")
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

    def sample(self, time_s: float) -> TrajectorySetpoint:
        mission_phase = self.phase(time_s)
        derivatives = np.zeros((4, 3))
        yaw_rate = yaw_acceleration = 0.0
        if mission_phase == "delay":
            position, yaw = self._initial_position.copy(), self._yaw_initial
        elif mission_phase == "takeoff":
            progress = _progress_derivatives((time_s-self._absolute_boundaries[1])/self._takeoff_duration, self._takeoff_duration)
            position = self._initial_position+np.array([0., 0., self.config["takeoff_height_m"]*progress[0]])
            derivatives[:, 2] = float(self.config["takeoff_height_m"])*progress[1:]
            yaw_change = self._yaw_spiral-self._yaw_initial
            yaw = self._yaw_initial+yaw_change*progress[0]
            yaw_rate, yaw_acceleration = yaw_change*progress[1:3]
        elif mission_phase == "spiral":
            progress = _progress_derivatives((time_s-self._absolute_boundaries[2])/self._spiral_duration, self._spiral_duration)
            phase = float(self.config["initial_phase_rad"])+self._turn_angle*progress[0]
            theta1, theta2, theta3, theta4 = self._turn_angle*progress[1:]
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
            derivatives[:, 2] = float(self.config["climb_height_m"])*progress[1:]
            yaw = self._yaw_spiral
            if self.config["yaw_mode"] == "tangent":
                yaw += self._turn_angle*progress[0]
                yaw_rate, yaw_acceleration = theta1, theta2
        else:
            position, yaw = self._endpoint.copy(), self._yaw_final
        return TrajectorySetpoint(position_w=position, velocity_w=derivatives[0], acceleration_w=derivatives[1],
                                  yaw_rad=float(yaw), yaw_rate_rad_s=float(yaw_rate),
                                  yaw_acceleration_rad_s2=float(yaw_acceleration),
                                  jerk_w=derivatives[2], snap_w=derivatives[3])


# Preserve the historical factory/config import while making the geometry explicit.
HelixTrajectory = SpiralTrajectory
