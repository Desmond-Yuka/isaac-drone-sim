"""Time-aligned basic flight signals derived from actual simulator states.

Kinematic acceleration and net wrench are interval averages inferred from
motion, not IMU or load-cell measurements. No acceleration is fabricated at
reset: the first sample uses both ends of the first completed physics step.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isaac_drone.core.rotations import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_log,
    rotation_to_rpy,
    validate_rotation,
    wrap_angle,
)
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench
from isaac_drone.core.validation import finite_array


def _optional_vector(telemetry: Mapping, key: str, size: int):
    value = telemetry.get(key)
    return None if value is None else finite_array(value, (size,), key)


def _difference(ideal, actual):
    """Ideal minus actual, or None when either side is unavailable."""
    return None if ideal is None or actual is None else ideal - actual


def _norm(vector):
    return None if vector is None else float(np.linalg.norm(vector))


def build_basic_record(
    before: VehicleState,
    after: VehicleState,
    reference: TrajectorySetpoint,
    reference_sample_time_s: float,
    mass: MassProperties,
    backend_telemetry: Mapping,
    commanded: Wrench,
    allocated: Wrench,
    disturbance: Wrench,
    *,
    desired_rotation_w=None,
    desired_angular_velocity_d=None,
    motor_command_rps=None,
) -> dict:
    """Build one completed-physics-interval record in explicit SI units.

    Position, velocity and motor speed are actual simulation quantities. The
    target is the control reference held over this interval, sampled at
    ``reference_sample_time_s``; error is that target minus the endpoint CoM
    position. Sampling a trajectory again for logging could change controller
    behavior and would conceal which target was actually commanded.

    Every tracked quantity is logged as ideal, actual and ideal-minus-actual
    error. The attitude and angular-velocity ideals are the controller's held
    desired attitude R_d and desired-body rate; the rate is rotated into the
    endpoint body axes, as the controller does. Motor speed, rotor thrust and
    body force/torque ideals are the commands for this interval, compared with
    the motor model's output. Omitted ideals leave their fields None.

    Acceleration is Δv_world/Δt, including gravity/contact effects reflected in
    the motion. Mean net force is m Δv_world/Δt. Mean net torque is
    Δ(R J_com Rᵀ ω_world)/Δt about the moving whole-vehicle CoM. Differencing
    world angular momentum includes gyroscopic effects without assuming a
    diagonal inertia or subtracting body components in mismatched frames.

    Command, allocation, motor, disturbance and submitted wrenches retain the
    body frame used at interval start. Submitted wrench excludes gravity,
    collisions and implicit native damping, which PhysX adds separately.
    """
    if not isinstance(backend_telemetry, Mapping):
        raise TypeError("backend telemetry must be a mapping")
    dt = after.time_s - before.time_s
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Basic telemetry requires a positive completed physics interval")
    if (
        isinstance(reference_sample_time_s, bool)
        or not np.isfinite(reference_sample_time_s)
        or reference_sample_time_s < 0
        or reference_sample_time_s > before.time_s + 1e-12
    ):
        raise ValueError("Control reference must have been sampled no later than the interval start")
    rotation_before = quaternion_to_matrix(before.quaternion_wxyz)
    rotation_after = quaternion_to_matrix(after.quaternion_wxyz)
    acceleration_w = (after.linear_velocity_w - before.linear_velocity_w) / dt
    omega_before_w = rotation_before @ before.angular_velocity_b
    omega_after_w = rotation_after @ after.angular_velocity_b
    angular_acceleration_w = (omega_after_w - omega_before_w) / dt
    angular_acceleration_b = rotation_after.T @ angular_acceleration_w
    angular_momentum_before_w = rotation_before @ mass.inertia_com_b @ before.angular_velocity_b
    angular_momentum_after_w = rotation_after @ mass.inertia_com_b @ after.angular_velocity_b
    torque_net_w = (angular_momentum_after_w - angular_momentum_before_w) / dt
    error_w = reference.position_w - after.position_w
    velocity_error_w = reference.velocity_w - after.linear_velocity_w
    acceleration_error_w = reference.acceleration_w - acceleration_w
    attitude_rpy = rotation_to_rpy(rotation_after)

    target_quaternion = target_rpy = attitude_error_rpy = attitude_error_angle = None
    target_rate_b = None
    if desired_rotation_w is not None:
        desired = validate_rotation(desired_rotation_w, "desired_rotation_w")
        relative = rotation_after.T @ desired
        target_quaternion = matrix_to_quaternion(desired)
        target_rpy = rotation_to_rpy(desired)
        attitude_error_rpy = wrap_angle(target_rpy - attitude_rpy)
        attitude_error_angle = float(np.linalg.norm(rotation_log(relative)))
        if desired_angular_velocity_d is not None:
            target_rate_b = relative @ finite_array(desired_angular_velocity_d, (3,), "desired_angular_velocity_d")
    rate_error_b = _difference(target_rate_b, after.angular_velocity_b)

    motor_speed = _optional_vector(backend_telemetry, "motor_speed_rps", 4)
    speed_command = None if motor_command_rps is None else finite_array(motor_command_rps, (4,), "motor_command_rps")
    speed_error = _difference(speed_command, motor_speed)
    thrust_command = _optional_vector(backend_telemetry, "commanded_thrust_n", 4)
    thrust_applied = _optional_vector(backend_telemetry, "applied_thrust_n", 4)

    motor = _optional_vector(backend_telemetry, "motor_wrench_b", 6)
    submitted = None
    if backend_telemetry.get("has_submitted_wrench", True):
        submitted = _optional_vector(backend_telemetry, "total_wrench_b", 6)
    force_error = None if motor is None else commanded.force_b - motor[:3]
    torque_error = None if motor is None else commanded.torque_b - motor[3:]
    result = {
        "step_start_time_s": float(before.time_s),
        "sample_time_s": float(after.time_s),
        "interval_dt_s": float(dt),
        "reference_sample_time_s": float(reference_sample_time_s),
        "position_w_m": after.position_w.copy(),
        "velocity_w_m_s": after.linear_velocity_w.copy(),
        "acceleration_w_m_s2": acceleration_w,
        "acceleration_native_w_m_s2": _optional_vector(backend_telemetry, "linear_acceleration_com_w_m_s2", 3),
        "acceleration_native_source": backend_telemetry.get("linear_acceleration_source", "unavailable"),
        "quaternion_wxyz": after.quaternion_wxyz.copy(),
        "attitude_rpy_rad": attitude_rpy,
        "angular_velocity_b_rad_s": after.angular_velocity_b.copy(),
        "angular_velocity_w_rad_s": omega_after_w,
        "angular_acceleration_w_rad_s2": angular_acceleration_w,
        "angular_acceleration_b_rad_s2": angular_acceleration_b,
        "target_position_w_m": reference.position_w.copy(),
        "target_velocity_w_m_s": reference.velocity_w.copy(),
        "target_acceleration_w_m_s2": reference.acceleration_w.copy(),
        "target_quaternion_wxyz": target_quaternion,
        "target_attitude_rpy_rad": target_rpy,
        "target_angular_velocity_b_rad_s": target_rate_b,
        "position_error_w_m": error_w,
        "position_error_norm_m": float(np.linalg.norm(error_w)),
        "velocity_error_w_m_s": velocity_error_w,
        "velocity_error_norm_m_s": _norm(velocity_error_w),
        "acceleration_error_w_m_s2": acceleration_error_w,
        "acceleration_error_norm_m_s2": _norm(acceleration_error_w),
        "attitude_error_rpy_rad": attitude_error_rpy,
        "attitude_error_angle_rad": attitude_error_angle,
        "angular_velocity_error_b_rad_s": rate_error_b,
        "angular_velocity_error_norm_rad_s": _norm(rate_error_b),
        "motor_speed_rps": motor_speed,
        "motor_speed_rpm": _optional_vector(backend_telemetry, "motor_speed_rpm", 4),
        "motor_speed_rad_s": _optional_vector(backend_telemetry, "motor_speed_rad_s", 4),
        "motor_speed_command_rps": speed_command,
        "motor_speed_command_rpm": None if speed_command is None else speed_command * 60.0,
        "motor_speed_error_rps": speed_error,
        "motor_speed_error_rpm": None if speed_error is None else speed_error * 60.0,
        "motor_speed_source": backend_telemetry.get("motor_speed_source", "unavailable"),
        "motor_thrust_command_n": thrust_command,
        "motor_thrust_applied_n": thrust_applied,
        "motor_thrust_error_n": _difference(thrust_command, thrust_applied),
        "force_command_b_n": commanded.force_b.copy(),
        "torque_command_b_nm": commanded.torque_b.copy(),
        "force_allocated_b_n": allocated.force_b.copy(),
        "torque_allocated_b_nm": allocated.torque_b.copy(),
        "force_motor_b_n": None if motor is None else motor[:3],
        "torque_motor_b_nm": None if motor is None else motor[3:],
        "force_disturbance_b_n": disturbance.force_b.copy(),
        "torque_disturbance_b_nm": disturbance.torque_b.copy(),
        "force_submitted_b_n": None if submitted is None else submitted[:3],
        "torque_submitted_b_nm": None if submitted is None else submitted[3:],
        "force_net_w_n": mass.mass_kg * acceleration_w,
        "torque_net_w_nm": torque_net_w,
        "force_error_b_n": force_error,
        "force_error_norm_n": _norm(force_error),
        "torque_error_b_nm": torque_error,
        "torque_error_norm_nm": _norm(torque_error),
    }
    for key, value in result.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite derived flight signal: {key}")
    if not np.isfinite(result["position_error_norm_m"]):
        raise FloatingPointError("Non-finite position error norm")
    return result
