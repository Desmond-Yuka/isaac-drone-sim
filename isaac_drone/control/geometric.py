"""COM position and full-inertia geometric attitude control for +Z quadrotors.

Actual mass, full COM inertia, gravity and allocation geometry must come from
the vehicle configuration/backend. Rotor dynamics, batteries and disturbances
act outside this module and are observed through state/allocation feedback.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isaac_drone.core.rotations import desired_attitude_kinematics, quaternion_to_matrix, vee
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench
from isaac_drone.core.validation import finite_array, finite_scalar

from .allocation import AllocationResult


class GeometricController:
    """Position PID plus Lee-style SO(3) attitude control with feedforward.

    World position gains generate acceleration: kp [s^-2], kd [s^-1], ki
    [s^-3]. Integral state [m s] is bounded per axis. A zero ki explicitly
    disables integral action; no undocumented integral gain is introduced.

    Attitude gains follow the local Isaac Lab Lee implementation's direct
    torque convention: kp [N m/rad], kd [N m s/rad]. Gyroscopic coupling and
    desired angular acceleration use the full positive-definite COM inertia.
    The smooth SO(3) error has an unstable equilibrium at exactly 180 degrees;
    this controller targets upright flight, not inverted recovery.

    Heading derivatives are analytical. When jerk/snap are supplied, their
    force contribution is analytical; feedback-force derivatives use variable-
    dt backward differences. If force limits activate, the full projected
    force is differentiated numerically. Unknown feedback derivatives start
    at zero on reset; these are not exact derivatives of the measured plant.

    max_tilt_rad is measured from world +Z and must be below pi/2. Desired
    force is projected onto that positive-thrust cone. max_acceleration_m_s2
    limits net translational acceleration before gravity compensation. Yaw
    feedforward rate is clipped; acceleration of a rate outside the interval
    is zero. At its boundary only inward acceleration is admitted. This is
    not a slew-rate limiter for a discontinuous yaw angle reference.
    Call notify_allocation after every allocation to prevent integral windup.
    """

    def __init__(self, config: dict, mass_properties: MassProperties, gravity_w: np.ndarray):
        if not isinstance(config, Mapping):
            raise ValueError("control config must be a mapping")
        required = (
            "position_kp", "velocity_kd", "position_ki", "integral_limit_m_s",
            "attitude_kp", "angular_rate_kd", "max_tilt_rad", "max_yaw_rate_rad_s",
            "max_acceleration_m_s2",
        )
        missing = set(required) - set(config)
        if missing:
            raise ValueError(f"missing control configuration: {', '.join(sorted(missing))}")
        for name in required[:6]:
            value = finite_array(config[name], (3,), name)
            if np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
            value.setflags(write=False)
            setattr(self, name, value)
        self.max_tilt_rad = finite_scalar(config["max_tilt_rad"], "max_tilt_rad")
        if not 0 <= self.max_tilt_rad < np.pi / 2:
            raise ValueError("max_tilt_rad must satisfy 0 <= max_tilt_rad < pi/2")
        self.max_yaw_rate_rad_s = finite_scalar(config["max_yaw_rate_rad_s"], "max_yaw_rate_rad_s")
        if self.max_yaw_rate_rad_s < 0:
            raise ValueError("max_yaw_rate_rad_s must be nonnegative")
        self.max_acceleration_m_s2 = config["max_acceleration_m_s2"]
        if self.max_acceleration_m_s2 is not None:
            self.max_acceleration_m_s2 = finite_scalar(self.max_acceleration_m_s2, "max_acceleration_m_s2")
            if self.max_acceleration_m_s2 <= 0:
                raise ValueError("max_acceleration_m_s2 must be positive or None")
        self.mass_kg = finite_scalar(mass_properties.mass_kg, "mass_kg")
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        self.inertia_com_b = finite_array(mass_properties.inertia_com_b, (3, 3), "inertia_com_b")
        if not np.allclose(self.inertia_com_b, self.inertia_com_b.T, atol=1e-12, rtol=1e-9):
            raise ValueError("inertia_com_b must be symmetric")
        try:
            np.linalg.cholesky(self.inertia_com_b)
        except np.linalg.LinAlgError as exc:
            raise ValueError("inertia_com_b must be positive definite") from exc
        finite_array(mass_properties.com_b, (3,), "com_b")
        self.gravity_w = finite_array(gravity_w, (3,), "gravity_w")
        self.inertia_com_b.setflags(write=False)
        self.gravity_w.setflags(write=False)
        self.reset()

    def reset(self) -> None:
        """Clear integral, allocation feedback, and reference derivative history."""
        self._integral = np.zeros(3)
        self._integral_before_step = np.zeros(3)
        self._force_history: list[np.ndarray] = []
        self._feedback_force_history: list[np.ndarray] = []
        self._limited_history: list[bool] = []
        self.force_derivative_source = "uninitialized"
        self._previous_dt: float | None = None
        self._allocation_limited = False
        self._has_computed = False
        self._last_time: float | None = None
        self._desired_rotation_w = np.eye(3)
        self._desired_angular_velocity_d = np.zeros(3)
        self._desired_angular_acceleration_d = np.zeros(3)

    @property
    def integral_error_m_s(self) -> np.ndarray:
        return self._integral.copy()

    @property
    def desired_rotation_w(self) -> np.ndarray:
        return self._desired_rotation_w.copy()

    @property
    def desired_angular_velocity_d(self) -> np.ndarray:
        return self._desired_angular_velocity_d.copy()

    @property
    def desired_angular_acceleration_d(self) -> np.ndarray:
        return self._desired_angular_acceleration_d.copy()

    def notify_allocation(self, result: AllocationResult) -> None:
        """Suspend integral growth when allocation cannot achieve the command.

        Roll back newly growing components and allow old integral components
        to unwind. An exact command on a bound does not freeze integration.
        Regularization residual counts too: that wrench was not applied.
        """
        residual = finite_array(result.residual, (6,), "allocation residual")
        self._allocation_limited = bool(np.linalg.norm(residual) > 1e-8)
        if self._allocation_limited and self._has_computed:
            growing = np.abs(self._integral) > np.abs(self._integral_before_step)
            self._integral[growing] = self._integral_before_step[growing]

    def _limited_force(self, acceleration: np.ndarray) -> tuple[np.ndarray, bool]:
        limited = False
        magnitude = np.linalg.norm(acceleration)
        if self.max_acceleration_m_s2 is not None and magnitude > self.max_acceleration_m_s2:
            acceleration = acceleration * (self.max_acceleration_m_s2 / magnitude)
            limited = True
        force = self.mass_kg * (acceleration - self.gravity_w)
        horizontal = np.linalg.norm(force[:2])
        sine, cosine = np.sin(self.max_tilt_rad), np.cos(self.max_tilt_rad)
        if force[2] >= 0 and horizontal * cosine <= force[2] * sine:
            return force, limited
        # Orthogonal projection onto a circular cone, including its apex.
        projection = max(0.0, sine * horizontal + cosine * force[2])
        direction = np.zeros(3)
        if horizontal > 0:
            direction[:2] = sine * force[:2] / horizontal
        direction[2] = cosine
        return projection * direction, True

    def _force_derivatives(self, force: np.ndarray, dt: float, history=None) -> tuple[np.ndarray, np.ndarray]:
        history = self._force_history if history is None else history
        if not history:
            return np.zeros(3), np.zeros(3)
        first = (force - history[-1]) / dt
        if len(history) == 1:
            return first, np.zeros(3)
        previous_dt = self._previous_dt
        assert previous_dt is not None
        previous_first = (history[-1] - history[-2]) / previous_dt
        second = 2.0 * (first - previous_first) / (dt + previous_dt)
        return first + 0.5 * dt * second, second

    def compute(self, state: VehicleState, setpoint: TrajectorySetpoint, dt_s: float) -> Wrench:
        dt = finite_scalar(dt_s, "dt_s")
        if dt <= 0:
            raise ValueError("dt_s must be positive")
        time = finite_scalar(state.time_s, "state.time_s")
        if self._last_time is not None and time <= self._last_time:
            raise ValueError("state.time_s must increase between compute calls; reset after a simulation reset")
        position = finite_array(state.position_w, (3,), "state.position_w")
        velocity = finite_array(state.linear_velocity_w, (3,), "state.linear_velocity_w")
        omega = finite_array(state.angular_velocity_b, (3,), "state.angular_velocity_b")
        rotation = quaternion_to_matrix(state.quaternion_wxyz)
        position_error = finite_array(setpoint.position_w, (3,), "setpoint.position_w") - position
        velocity_error = finite_array(setpoint.velocity_w, (3,), "setpoint.velocity_w") - velocity
        reference_acceleration = finite_array(setpoint.acceleration_w, (3,), "setpoint.acceleration_w")
        yaw = finite_scalar(setpoint.yaw_rad, "setpoint.yaw_rad")
        raw_yaw_rate = finite_scalar(setpoint.yaw_rate_rad_s, "setpoint.yaw_rate_rad_s")
        yaw_acceleration = finite_scalar(setpoint.yaw_acceleration_rad_s2, "setpoint.yaw_acceleration_rad_s2")
        yaw_rate = float(np.clip(raw_yaw_rate, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s))
        if abs(raw_yaw_rate) > self.max_yaw_rate_rad_s or (
            abs(raw_yaw_rate) == self.max_yaw_rate_rad_s and raw_yaw_rate * yaw_acceleration > 0
        ):
            yaw_acceleration = 0.0
        if self.max_yaw_rate_rad_s == 0:
            yaw_acceleration = 0.0
        self._integral_before_step = self._integral.copy()
        proposal = np.clip(self._integral + position_error * dt, -self.integral_limit_m_s, self.integral_limit_m_s)
        proposal[self.position_ki == 0] = 0.0
        if self._allocation_limited:
            proposal = np.where(np.abs(proposal) <= np.abs(self._integral), proposal, self._integral)
        self._integral = proposal
        acceleration = self.position_kp * position_error + self.velocity_kd * velocity_error + reference_acceleration
        force, internally_limited = self._limited_force(acceleration + self.position_ki * self._integral)
        if internally_limited:
            growing = np.abs(self._integral) > np.abs(self._integral_before_step)
            self._integral[growing] = self._integral_before_step[growing]
            force, _ = self._limited_force(acceleration + self.position_ki * self._integral)
        feedback_force = force - self.mass_kg * reference_acceleration
        if setpoint.jerk_w is not None and not internally_limited and not any(self._limited_history):
            dfeedback, ddfeedback = self._force_derivatives(feedback_force, dt, self._feedback_force_history)
            dforce = self.mass_kg * setpoint.jerk_w + dfeedback
            ddforce = self.mass_kg * setpoint.snap_w + ddfeedback
            self.force_derivative_source = "analytic_reference_and_numerical_feedback"
        else:
            dforce, ddforce = self._force_derivatives(force, dt)
            self.force_derivative_source = "numerical_total_force"
        desired, omega_desired, alpha_desired = desired_attitude_kinematics(
            force, dforce, ddforce, yaw, yaw_rate, yaw_acceleration,
            self._desired_rotation_w if self._has_computed else rotation,
        )
        relative = rotation.T @ desired
        attitude_error = 0.5 * vee(relative.T - relative)
        desired_rate_current_body = relative @ omega_desired
        rate_error = omega - desired_rate_current_body
        feedforward_acceleration = relative @ alpha_desired - np.cross(omega, desired_rate_current_body)
        torque = (
            -self.attitude_kp * attitude_error - self.angular_rate_kd * rate_error
            + np.cross(omega, self.inertia_com_b @ omega)
            + self.inertia_com_b @ feedforward_acceleration
        )
        thrust = max(0.0, float(force @ rotation[:, 2]))
        wrench = Wrench(np.array([0.0, 0.0, thrust]), torque)
        finite_array(wrench.vector, (6,), "computed wrench")
        self._desired_rotation_w = desired
        self._desired_angular_velocity_d = omega_desired
        self._desired_angular_acceleration_d = alpha_desired
        self._force_history.append(force.copy())
        self._force_history = self._force_history[-2:]
        self._feedback_force_history.append(feedback_force.copy())
        self._feedback_force_history = self._feedback_force_history[-2:]
        self._limited_history.append(internally_limited)
        self._limited_history = self._limited_history[-2:]
        self._previous_dt = dt
        self._last_time = time
        self._has_computed = True
        return wrench
