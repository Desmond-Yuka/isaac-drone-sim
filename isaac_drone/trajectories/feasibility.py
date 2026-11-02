"""Reject evidently infeasible nominal trajectory references before flight.

This is a finite sampled rigid-body feedforward check, not a closed-loop
stability proof, motor-dynamics simulation, or continuous-time certificate.
All masses, complete inertias, geometry and actuator bounds are supplied by the
caller from the actual vehicle. No symmetric rotor layout is substituted.
"""

from __future__ import annotations

import math

import numpy as np

from isaac_drone.control.allocation import BoundedAllocator
from isaac_drone.control.base import FlightLimits
from isaac_drone.core.rotations import desired_attitude_kinematics
from isaac_drone.core.types import MassProperties, Wrench
from isaac_drone.core.validation import finite_array, finite_scalar

from .base import SegmentedTrajectory
from .timing import HermiteTimeLaw

_RESIDUAL_RTOL = 1e-7
_RESIDUAL_ATOL = 1e-9


def nominal_requirement(reference, mass: float, inertia: np.ndarray, gravity: np.ndarray, where: str = ""):
    """Ideal-tracking collective thrust [N], body torque [N m], tilt [rad], body rate.

    ``f = ||m(a_d-g)||`` and ``M = J alpha_d + omega_d x (J omega_d)``, with the
    desired attitude derivatives taken analytically from jerk, snap and yaw.
    """
    if reference.jerk_w is None or reference.snap_w is None:
        raise ValueError("Analytic feasibility checks require reference jerk and snap")
    force_w = mass * (reference.acceleration_w - gravity)
    thrust = float(np.linalg.norm(force_w))
    if not np.isfinite(thrust) or thrust <= 1e-10:
        raise ValueError(f"Nominal thrust direction is degenerate at {where}")
    tilt = float(np.arctan2(np.linalg.norm(force_w[:2]), force_w[2]))
    _, omega, alpha = desired_attitude_kinematics(
        force_w,
        mass * reference.jerk_w,
        mass * reference.snap_w,
        reference.yaw_rad,
        reference.yaw_rate_rad_s,
        reference.yaw_acceleration_rad_s2,
    )
    torque = inertia @ alpha + np.cross(omega, inertia @ omega)
    return thrust, torque, tilt, omega


def _sample_count(segment, law, name: str, start: float, end: float) -> int:
    """Hermite laws use the segment's fixed count; cruise pieces 16 per revolution; ramps 33."""
    if isinstance(law, HermiteTimeLaw):
        return segment.hermite_samples
    if name == "cruise":
        revolutions = segment.revolutions * law.cruise_rate * (end - start)
        return max(33, math.ceil(16 * revolutions) + 1)
    return 33


def validate_trajectory_feasibility(
    trajectory: SegmentedTrajectory,
    limits: FlightLimits,
    mass_properties: MassProperties,
    gravity_w: np.ndarray,
    allocation_matrix_b: np.ndarray,
    lower_thrust_n: np.ndarray,
    upper_thrust_n: np.ndarray,
) -> dict:
    """Check hover, exact yaw-rate peaks, and sampled nominal feedforward.

    The trajectory must already be reset (and any null-duration segment planned)
    at its intended actual starting state. Every time-law piece is sampled
    uniformly including its boundaries: a Hermite law at its segment's
    ``hermite_samples``, each ramp of a cruise law at 33, and a cruise at no
    fewer than 16 samples per revolution. All sample times are returned.
    Reference acceleration, required tilt and instantaneous rotor thrust bounds
    are checked without silently clipping any trajectory value.

    At each point the ideal tracking wrench is ``f = ||m(a_d-g)||`` and
    ``M = J alpha_d + omega_d x (J omega_d)``. Analytic desired attitude
    derivatives come from jerk/snap and yaw derivatives. This ideal tracking
    state is a *necessary planning screen*, not a claim the real controller has
    zero errors or the real motors have instantaneous response. A finite grid
    can miss violations between samples, and no feedback/disturbance reserve
    is assumed. Yaw is affine in each segment's progress, so the exact yaw-rate
    peak is |yaw change| times the law's exact peak progress rate.
    """
    if not isinstance(trajectory, SegmentedTrajectory):
        raise TypeError("Feasibility checks require a SegmentedTrajectory")
    boundaries = trajectory.phase_boundaries_s  # Raises clearly if unreset or unplanned.
    origin = trajectory.start_time_s
    gravity = finite_array(gravity_w, (3,), "gravity_w")
    matrix = finite_array(allocation_matrix_b, (6, 4), "allocation_matrix_b")
    lower = finite_array(lower_thrust_n, (4,), "lower_thrust_n")
    upper = finite_array(upper_thrust_n, (4,), "upper_thrust_n")
    if np.any(lower < 0) or np.any(upper < lower):
        raise ValueError("Feasibility thrust bounds must satisfy 0 <= lower <= upper")
    if not np.allclose(matrix[:3], np.tile([[0.0], [0.0], [1.0]], (1, 4)), rtol=0, atol=1e-7):
        raise ValueError("The upright controllers require every thrust axis along body +Z")
    if np.linalg.matrix_rank(matrix[[2, 3, 4, 5]]) != 4:
        raise ValueError("Allocation lacks four independent collective/attitude channels")
    mass = finite_scalar(mass_properties.mass_kg, "mass_kg")
    inertia = finite_array(mass_properties.inertia_com_b, (3, 3), "inertia_com_b")
    if mass <= 0 or not np.allclose(inertia, inertia.T, rtol=1e-9, atol=1e-12):
        raise ValueError("Feasibility requires positive mass and symmetric full COM inertia")
    try:
        np.linalg.cholesky(inertia)
    except np.linalg.LinAlgError as error:
        raise ValueError("Feasibility requires positive-definite COM inertia") from error
    maximum_tilt = limits.max_tilt_rad
    maximum_yaw_rate = limits.max_yaw_rate_rad_s
    maximum_acceleration = limits.max_acceleration_m_s2

    laws = trajectory.time_laws
    segments = {segment.name: segment for segment in trajectory.segments}
    yaw_peaks = {name: abs(segment.yaw_change_rad) * laws[name].peak_rate for name, segment in segments.items()}
    yaw_peak = max(yaw_peaks.values(), default=0.0)
    for name, peak in yaw_peaks.items():
        if peak > maximum_yaw_rate + 1e-12 * max(1.0, maximum_yaw_rate):
            raise ValueError(
                f"Analytic {name} yaw-rate peak {peak:.9g} rad/s exceeds "
                f"limits.max_yaw_rate_rad_s={maximum_yaw_rate:.9g}; lengthen the segment or "
                "explicitly revise the limit"
            )

    allocator = BoundedAllocator(matrix, np.ones(6), regularization=0.0)

    def allocate_exact(wrench: Wrench, label: str):
        result = allocator.allocate(wrench, lower, upper)
        tolerance = _RESIDUAL_ATOL + _RESIDUAL_RTOL * np.maximum(1.0, np.abs(wrench.vector))
        if np.any(np.abs(result.residual) > tolerance):
            raise ValueError(
                f"{label}: actuator bounds cannot realize the required nominal wrench; "
                f"requested={wrench.vector.tolist()}, residual={result.residual.tolist()}"
            )
        return result

    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= 1e-12:
        raise ValueError("Hover feasibility requires a nonzero configured gravity vector")
    hover_tilt = float(np.arctan2(np.linalg.norm(gravity[:2]), -gravity[2]))
    if hover_tilt > maximum_tilt + 1e-10:
        raise ValueError("Static hover requires a tilt beyond the configured maximum")
    hover_wrench = Wrench([0, 0, mass * gravity_norm], np.zeros(3))
    hover = allocate_exact(hover_wrench, "Static hover infeasible")

    pieces = [
        np.linspace(start, end, _sample_count(segments[phase], laws[phase], name, start, end))
        for phase, name, start, end in trajectory.law_segments()
    ]
    times = np.unique(np.concatenate([[origin], *pieces]))
    sampled_thrusts, acceleration_norms, tilts, angular_speeds, torque_norms, residuals = [], [], [], [], [], []
    for time in times:
        reference = trajectory.sample(float(time))
        acceleration_norm = float(np.linalg.norm(reference.acceleration_w))
        if maximum_acceleration is not None and acceleration_norm > maximum_acceleration + 1e-10:
            raise ValueError(
                f"Reference acceleration {acceleration_norm:.9g} m/s^2 at t={time:.9g} s "
                f"exceeds limits.max_acceleration_m_s2={maximum_acceleration:.9g}"
            )
        thrust, torque, tilt, omega = nominal_requirement(reference, mass, inertia, gravity, f"t={time:.9g} s")
        if tilt > maximum_tilt + 1e-10:
            raise ValueError(
                f"Nominal required tilt {tilt:.9g} rad at t={time:.9g} s exceeds limits.max_tilt_rad={maximum_tilt:.9g}"
            )
        result = allocate_exact(Wrench([0, 0, thrust], torque), f"Feedforward infeasible at t={time:.9g} s")
        sampled_thrusts.append(result.thrusts_n)
        acceleration_norms.append(acceleration_norm)
        tilts.append(tilt)
        angular_speeds.append(float(np.linalg.norm(omega)))
        torque_norms.append(float(np.linalg.norm(torque)))
        residuals.append(result.residual)
    thrusts = np.array(sampled_thrusts)
    thrust_fraction = thrusts.max(axis=0) / upper
    return {
        "status": "passed_nominal_sampled_checks",
        "continuous_time_certificate": False,
        "closed_loop_guarantee": False,
        "motor_dynamic_feasibility_checked": False,
        "sample_count": len(times),
        "sample_times_s": times.tolist(),
        "exact_yaw_rate_peak_by_phase_rad_s": {name: float(peak) for name, peak in yaw_peaks.items()},
        "exact_yaw_rate_peak_rad_s": float(yaw_peak),
        "sampled_max_reference_acceleration_m_s2": max(acceleration_norms),
        "sampled_max_required_tilt_rad": max(tilts),
        "sampled_max_desired_angular_speed_rad_s": max(angular_speeds),
        "sampled_max_required_torque_norm_nm": max(torque_norms),
        "sampled_min_rotor_thrust_n": thrusts.min(axis=0).tolist(),
        "sampled_max_rotor_thrust_n": thrusts.max(axis=0).tolist(),
        "sampled_min_lower_thrust_margin_n": (thrusts - lower).min(axis=0).tolist(),
        "sampled_min_upper_thrust_margin_n": (upper - thrusts).min(axis=0).tolist(),
        "sampled_max_absolute_allocation_residual": np.abs(residuals).max(axis=0).tolist(),
        # With one fixed k_f per rotor, n/n_max = sqrt(f/f_max).
        "sampled_max_thrust_fraction_of_max": thrust_fraction.tolist(),
        "sampled_max_motor_speed_fraction_of_max": np.sqrt(thrust_fraction).tolist(),
        "phase_boundaries_s": [start for start, _ in boundaries],
        "phase_names": [name for _, name in boundaries],
        "timing": trajectory.timing_report(),
        "hover_thrusts_n": hover.thrusts_n.tolist(),
        "hover_allocation_residual": hover.residual.tolist(),
        "lower_thrust_n": lower.tolist(),
        "upper_thrust_n": upper.tolist(),
        "mass_kg": mass,
        "limitations": [
            "Finite samples can miss violations between sample times; only yaw-rate peaks are checked analytically.",
            "Nominal perfect-reference rigid-body feedforward is checked, not feedback-error or disturbance authority.",
            "Actual supplied actuator bounds are instantaneous; lag, command slew and motor/battery dynamics require "
            "separate verification.",
            "Passing does not establish tracking accuracy, closed-loop stability, contact-free motion or hardware "
            "safety.",
        ],
    }
