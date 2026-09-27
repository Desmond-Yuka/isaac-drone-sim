"""Reject evidently infeasible nominal helix references before flight.

This is a finite sampled rigid-body feedforward check, not a closed-loop
stability proof, motor-dynamics simulation, or continuous-time certificate.
All masses, complete inertias, geometry and actuator bounds are supplied by the
caller from the actual vehicle. No symmetric rotor layout is substituted.
"""
from __future__ import annotations

import math

import numpy as np

from isaac_drone.control.allocation import BoundedAllocator
from isaac_drone.control.math import desired_attitude_kinematics, finite_array, finite_scalar
from isaac_drone.types import MassProperties, Wrench
from .spiral import SpiralTrajectory, validate_spiral_config


_RESIDUAL_RTOL = 1e-7
_RESIDUAL_ATOL = 1e-9


def validate_helix_feasibility(
    trajectory: SpiralTrajectory,
    control_config: dict,
    mass_properties: MassProperties,
    gravity_w: np.ndarray,
    allocation_matrix_b: np.ndarray,
    lower_thrust_n: np.ndarray,
    upper_thrust_n: np.ndarray,
) -> dict:
    """Check hover, exact yaw-rate peaks, and sampled nominal feedforward.

    The trajectory must already be reset to its intended actual starting state.
    Two motion phases are sampled at 33 and 65 uniform times respectively,
    including their boundaries and midpoints. All sample times are returned.
    Reference acceleration, required tilt and instantaneous rotor thrust bounds
    are checked without silently clipping any trajectory value.

    At each point the ideal tracking wrench is ``f = ||m(a_d-g)||`` and
    ``M = J alpha_d + omega_d x (J omega_d)``. Analytic desired attitude
    derivatives come from jerk/snap and yaw derivatives. This ideal tracking
    state is a *necessary planning screen*, not a claim the real controller has
    zero errors or the real motors have instantaneous response. A finite grid
    can miss violations between samples, and no feedback/disturbance reserve
    is assumed. Exact yaw maxima follow the known ninth-degree progress law.
    """
    if not isinstance(trajectory, SpiralTrajectory):
        raise TypeError("Helix feasibility requires the analytic HelixTrajectory implementation")
    validate_spiral_config(trajectory.config)
    # This module is part of the trajectory package and uses the implementation's
    # established absolute reset origin, never assuming every mission starts at 0.
    trajectory.endpoint_position_w  # Raises clearly if reset has not completed.
    origin = trajectory._time_origin
    cfg = trajectory.config
    gravity = finite_array(gravity_w, (3,), "gravity_w")
    matrix = finite_array(allocation_matrix_b, (6, 4), "allocation_matrix_b")
    lower = finite_array(lower_thrust_n, (4,), "lower_thrust_n")
    upper = finite_array(upper_thrust_n, (4,), "upper_thrust_n")
    if np.any(lower < 0) or np.any(upper < lower):
        raise ValueError("Helix feasibility thrust bounds must satisfy 0 <= lower <= upper")
    if not np.allclose(matrix[:3], np.tile([[0.], [0.], [1.]], (1, 4)), rtol=0, atol=1e-7):
        raise ValueError("Helix controller requires every thrust axis along body +Z")
    if np.linalg.matrix_rank(matrix[[2, 3, 4, 5]]) != 4:
        raise ValueError("Helix allocation lacks four independent collective/attitude channels")
    mass = finite_scalar(mass_properties.mass_kg, "mass_kg")
    inertia = finite_array(mass_properties.inertia_com_b, (3, 3), "inertia_com_b")
    if mass <= 0 or not np.allclose(inertia, inertia.T, rtol=1e-9, atol=1e-12):
        raise ValueError("Helix feasibility requires positive mass and symmetric full COM inertia")
    try:
        np.linalg.cholesky(inertia)
    except np.linalg.LinAlgError as error:
        raise ValueError("Helix feasibility requires positive-definite COM inertia") from error
    maximum_tilt = finite_scalar(control_config["max_tilt_rad"], "max_tilt_rad")
    maximum_yaw_rate = finite_scalar(control_config["max_yaw_rate_rad_s"], "max_yaw_rate_rad_s")
    if not 0 <= maximum_tilt < math.pi/2 or maximum_yaw_rate < 0:
        raise ValueError("Helix controller tilt/yaw limits are invalid")
    maximum_acceleration = control_config["max_acceleration_m_s2"]
    if maximum_acceleration is not None:
        maximum_acceleration = finite_scalar(maximum_acceleration, "max_acceleration_m_s2")
        if maximum_acceleration <= 0:
            raise ValueError("max_acceleration_m_s2 must be positive or None")

    takeoff_start = origin+cfg["start_delay_s"]
    helix_start = origin+cfg["start_delay_s"]+cfg["takeoff_duration_s"]
    mission_end = origin+trajectory.mission_duration_s
    first = trajectory.sample(takeoff_start)
    at_helix = trajectory.sample(helix_start)
    peak_progress = 315.0/128.0
    yaw_takeoff_peak = abs(at_helix.yaw_rad-first.yaw_rad)*peak_progress/cfg["takeoff_duration_s"]
    yaw_helix_peak = (abs(2.0*math.pi*cfg["turns"])*peak_progress/cfg["spiral_duration_s"]
                      if cfg["yaw_mode"] == "tangent" else 0.0)
    yaw_peak = max(yaw_takeoff_peak, yaw_helix_peak)
    if yaw_peak > maximum_yaw_rate+1e-12*max(1.0, maximum_yaw_rate):
        raise ValueError(f"Analytic helix/takeoff yaw-rate peak {yaw_peak:.9g} rad/s exceeds "
                         f"control max_yaw_rate_rad_s={maximum_yaw_rate:.9g}; lengthen the phase or explicitly revise its limit")

    allocator = BoundedAllocator(matrix, np.ones(6), regularization=0.0)

    def allocate_exact(wrench: Wrench, label: str):
        result = allocator.allocate(wrench, lower, upper)
        tolerance = _RESIDUAL_ATOL+_RESIDUAL_RTOL*np.maximum(1.0, np.abs(wrench.vector))
        if np.any(np.abs(result.residual) > tolerance):
            raise ValueError(f"{label}: actuator bounds cannot realize the required nominal wrench; "
                             f"requested={wrench.vector.tolist()}, residual={result.residual.tolist()}")
        return result

    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= 1e-12:
        raise ValueError("Helix hover feasibility requires a nonzero configured gravity vector")
    hover_tilt = float(np.arctan2(np.linalg.norm(gravity[:2]), -gravity[2]))
    if hover_tilt > maximum_tilt+1e-10:
        raise ValueError("Static hover requires a tilt beyond the configured maximum")
    hover_wrench = Wrench([0, 0, mass*gravity_norm], np.zeros(3))
    hover = allocate_exact(hover_wrench, "Static hover infeasible")

    times = np.unique(np.r_[origin, np.linspace(takeoff_start, helix_start, 33),
                            np.linspace(helix_start, mission_end, 65)])
    sampled_thrusts, acceleration_norms, tilts, angular_speeds, torque_norms, residuals = [], [], [], [], [], []
    for time in times:
        reference = trajectory.sample(float(time))
        if reference.jerk_w is None or reference.snap_w is None:
            raise ValueError("Analytic helix feasibility requires reference jerk and snap")
        acceleration_norm = float(np.linalg.norm(reference.acceleration_w))
        if maximum_acceleration is not None and acceleration_norm > maximum_acceleration+1e-10:
            raise ValueError(f"Reference acceleration {acceleration_norm:.9g} m/s^2 at t={time:.9g} s "
                             f"exceeds control max_acceleration_m_s2={maximum_acceleration:.9g}")
        force_w = mass*(reference.acceleration_w-gravity)
        thrust = float(np.linalg.norm(force_w))
        if not np.isfinite(thrust) or thrust <= 1e-10:
            raise ValueError(f"Nominal thrust direction is degenerate at t={time:.9g} s")
        tilt = float(np.arctan2(np.linalg.norm(force_w[:2]), force_w[2]))
        if tilt > maximum_tilt+1e-10:
            raise ValueError(f"Nominal required tilt {tilt:.9g} rad at t={time:.9g} s "
                             f"exceeds control max_tilt_rad={maximum_tilt:.9g}")
        _, omega, alpha = desired_attitude_kinematics(
            force_w, mass*reference.jerk_w, mass*reference.snap_w,
            reference.yaw_rad, reference.yaw_rate_rad_s, reference.yaw_acceleration_rad_s2)
        torque = inertia@alpha+np.cross(omega, inertia@omega)
        result = allocate_exact(Wrench([0, 0, thrust], torque), f"Helix feedforward infeasible at t={time:.9g} s")
        sampled_thrusts.append(result.thrusts_n)
        acceleration_norms.append(acceleration_norm)
        tilts.append(tilt)
        angular_speeds.append(float(np.linalg.norm(omega)))
        torque_norms.append(float(np.linalg.norm(torque)))
        residuals.append(result.residual)
    thrusts = np.array(sampled_thrusts)
    return {
        "status": "passed_nominal_sampled_checks",
        "continuous_time_certificate": False,
        "closed_loop_guarantee": False,
        "motor_dynamic_feasibility_checked": False,
        "sample_count": len(times),
        "sample_times_s": times.tolist(),
        "exact_takeoff_yaw_rate_peak_rad_s": float(yaw_takeoff_peak),
        "exact_helix_yaw_rate_peak_rad_s": float(yaw_helix_peak),
        "exact_yaw_rate_peak_rad_s": float(yaw_peak),
        "sampled_max_reference_acceleration_m_s2": max(acceleration_norms),
        "sampled_max_required_tilt_rad": max(tilts),
        "sampled_max_desired_angular_speed_rad_s": max(angular_speeds),
        "sampled_max_required_torque_norm_nm": max(torque_norms),
        "sampled_min_rotor_thrust_n": thrusts.min(axis=0).tolist(),
        "sampled_max_rotor_thrust_n": thrusts.max(axis=0).tolist(),
        "sampled_min_lower_thrust_margin_n": (thrusts-lower).min(axis=0).tolist(),
        "sampled_min_upper_thrust_margin_n": (upper-thrusts).min(axis=0).tolist(),
        "sampled_max_absolute_allocation_residual": np.abs(residuals).max(axis=0).tolist(),
        "hover_thrusts_n": hover.thrusts_n.tolist(),
        "hover_allocation_residual": hover.residual.tolist(),
        "lower_thrust_n": lower.tolist(),
        "upper_thrust_n": upper.tolist(),
        "mass_kg": mass,
        "limitations": [
            "Finite samples can miss violations between sample times; only yaw-rate peaks are checked analytically.",
            "Nominal perfect-reference rigid-body feedforward is checked, not feedback-error or disturbance authority.",
            "Actual supplied actuator bounds are instantaneous; lag, command slew and motor/battery dynamics require separate verification.",
            "Passing does not establish tracking accuracy, closed-loop stability, contact-free motion or hardware safety.",
        ],
    }
