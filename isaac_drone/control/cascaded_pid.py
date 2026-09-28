"""Classic cascaded PID: position -> velocity -> attitude -> body rate (``kind: cascaded_pid``).

Each loop feeds the next one's setpoint, as in common flight stacks:

1. velocity setpoint ``v_sp = v_ref + Kp_pos (p_ref - p)`` (correction optionally norm-limited);
2. acceleration ``a = a_ref + Kp_vel (v_sp - v) + I_vel`` with a bounded integral term;
3. thrust vector ``m (a - g)`` limited by ``limits`` (acceleration norm, tilt cone),
   desired attitude from its direction and the reference yaw;
4. body-rate setpoint ``w_sp = -Kp_att e_R + (yaw-rate feedforward)``, optionally clipped;
5. angular acceleration ``alpha = Kp_rate (w_sp - w) + I_rate``, torque ``J alpha + w x J w``.

All gains are in 1/s (integral gains 1/s^2): the rate loop is normalized by the
actual full COM inertia, so gains are loop bandwidths independent of vehicle
scale. Unlike the geometric controller it uses no jerk/snap/attitude-rate
feedforward, which makes it a baseline for comparison, not a tracking optimum.
Integrals do not grow while the thrust vector or the allocation is saturated.
The attitude error has the SO(3) half-turn ambiguity; upright flight only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaac_drone.core.params import Vec3, params_to_dict
from isaac_drone.core.rotations import desired_attitude_kinematics, quaternion_to_matrix, vee
from isaac_drone.core.types import TrajectorySetpoint, VehicleState, Wrench
from isaac_drone.core.validation import finite_array, finite_scalar

from .allocation import AllocationResult
from .base import CONTROLLERS, ControllerDiagnostics, FlightLimits, VehicleModel
from .shaping import limited_thrust_vector


@dataclass(frozen=True)
class CascadedPidGains:
    """``controller: {kind: cascaded_pid}``; limits come from the shared ``limits:`` section."""

    position_kp: Vec3  # [1/s] position error -> velocity correction
    velocity_kp: Vec3  # [1/s] velocity error -> acceleration
    velocity_ki: Vec3  # [1/s^2]
    velocity_integral_limit_m_s2: Vec3  # bound of the integral acceleration term per axis
    attitude_kp: Vec3  # [1/s] attitude error -> body-rate setpoint
    rate_kp: Vec3  # [1/s] body-rate error -> angular acceleration
    rate_ki: Vec3  # [1/s^2]
    rate_integral_limit_rad_s2: Vec3  # bound of the integral angular-acceleration term per axis
    max_velocity_correction_m_s: float | None = None
    max_body_rate_rad_s: Vec3 | None = None

    def __post_init__(self):
        for name, value in params_to_dict(self).items():
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            if min(values) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.max_velocity_correction_m_s is not None and self.max_velocity_correction_m_s <= 0:
            raise ValueError("max_velocity_correction_m_s must be positive or null")
        if self.max_body_rate_rad_s is not None and min(self.max_body_rate_rad_s) <= 0:
            raise ValueError("max_body_rate_rad_s must be positive or null")


class CascadedPidController:
    output = "wrench"

    def __init__(self, gains: CascadedPidGains, vehicle: VehicleModel, limits: FlightLimits):
        self.gains = gains
        self.limits = limits
        self.mass_kg = finite_scalar(vehicle.mass_properties.mass_kg, "mass_kg")
        self.inertia = finite_array(vehicle.mass_properties.inertia_com_b, (3, 3), "inertia_com_b")
        self.gravity_w = finite_array(vehicle.gravity_w, (3,), "gravity_w")
        self._array = {
            name: None if value is None else np.asarray(value, dtype=float)
            for name, value in params_to_dict(gains).items()
            if isinstance(value, list) or value is None
        }
        self.reset()

    def reset(self) -> None:
        self._velocity_integral = np.zeros(3)
        self._rate_integral = np.zeros(3)
        self._previous = (np.zeros(3), np.zeros(3))
        self._desired_rotation = None
        self._desired_rate_d = np.zeros(3)
        self._details = {}

    def _integrate(self, integral, error, gain, limit, dt, frozen):
        proposal = np.clip(integral + gain * error * dt, -limit, limit)
        if frozen:  # anti-windup: keep components that shrink, block growth
            proposal = np.where(np.abs(proposal) <= np.abs(integral), proposal, integral)
        return proposal

    def compute(self, state: VehicleState, setpoint: TrajectorySetpoint, dt_s: float) -> Wrench:
        dt = finite_scalar(dt_s, "dt_s")
        if dt <= 0:
            raise ValueError("dt_s must be positive")
        g = self._array
        rotation = quaternion_to_matrix(state.quaternion_wxyz)
        self._previous = (self._velocity_integral.copy(), self._rate_integral.copy())

        correction = g["position_kp"] * (setpoint.position_w - state.position_w)
        limit = self.gains.max_velocity_correction_m_s
        if limit is not None and np.linalg.norm(correction) > limit:
            correction *= limit / np.linalg.norm(correction)
        velocity_setpoint = setpoint.velocity_w + correction
        velocity_error = velocity_setpoint - state.linear_velocity_w
        acceleration = setpoint.acceleration_w + g["velocity_kp"] * velocity_error + self._velocity_integral
        force, limited = limited_thrust_vector(
            acceleration, self.mass_kg, self.gravity_w, self.limits.max_tilt_rad, self.limits.max_acceleration_m_s2
        )
        self._velocity_integral = self._integrate(
            self._velocity_integral, velocity_error, g["velocity_ki"], g["velocity_integral_limit_m_s2"], dt, limited
        )

        yaw_rate = float(
            np.clip(setpoint.yaw_rate_rad_s, -self.limits.max_yaw_rate_rad_s, self.limits.max_yaw_rate_rad_s)
        )
        fallback = rotation if self._desired_rotation is None else self._desired_rotation
        desired, desired_rate_d, _ = desired_attitude_kinematics(
            force, np.zeros(3), np.zeros(3), setpoint.yaw_rad, yaw_rate, 0.0, fallback
        )
        relative = rotation.T @ desired
        attitude_error = 0.5 * vee(relative.T - relative)
        rate_setpoint = -g["attitude_kp"] * attitude_error + relative @ desired_rate_d
        if g["max_body_rate_rad_s"] is not None:
            rate_setpoint = np.clip(rate_setpoint, -g["max_body_rate_rad_s"], g["max_body_rate_rad_s"])
        omega = state.angular_velocity_b
        rate_error = rate_setpoint - omega
        angular_acceleration = g["rate_kp"] * rate_error + self._rate_integral
        self._rate_integral = self._integrate(
            self._rate_integral, rate_error, g["rate_ki"], g["rate_integral_limit_rad_s2"], dt, False
        )
        torque = self.inertia @ angular_acceleration + np.cross(omega, self.inertia @ omega)
        thrust = max(0.0, float(force @ rotation[:, 2]))

        self._desired_rotation = desired
        self._desired_rate_d = relative.T @ rate_setpoint  # rate setpoint expressed in desired axes
        self._details = {
            "velocity_setpoint_w_m_s": velocity_setpoint.tolist(),
            "rate_setpoint_b_rad_s": rate_setpoint.tolist(),
            "thrust_vector_limited": bool(limited),
        }
        return Wrench(np.array([0.0, 0.0, thrust]), torque)

    def notify_allocation(self, result: AllocationResult) -> None:
        """Undo this step's integral growth when the allocated wrench differs from the command."""
        if np.linalg.norm(finite_array(result.residual, (6,), "allocation residual")) > 1e-8:
            for name, previous in zip(("_velocity_integral", "_rate_integral"), self._previous):
                current = getattr(self, name)
                setattr(self, name, np.where(np.abs(current) > np.abs(previous), previous, current))

    def diagnostics(self) -> ControllerDiagnostics:
        rotation = None if self._desired_rotation is None else self._desired_rotation.copy()
        return ControllerDiagnostics(
            rotation, None if rotation is None else self._desired_rate_d.copy(), dict(self._details)
        )


@CONTROLLERS.register("cascaded_pid", params=CascadedPidGains)
def build_cascaded_pid(params: CascadedPidGains, *, vehicle: VehicleModel, limits: FlightLimits):
    """Cascaded PID: position -> velocity -> attitude -> body rate (no jerk/snap feedforward)."""
    return CascadedPidController(params, vehicle, limits)
