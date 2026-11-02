"""Estimation, control and total errors, all ideal minus actual like the rest of the telemetry.

- estimation error: truth − estimate (the estimate's ideal is the truth);
- control error: reference − estimate (what the controller acted on);
- total error: reference − truth (``basic.position_error_*`` already records it),
  so total = control − estimation for every quantity.

Attitude errors use the rotation vector of R_actualᵀ R_ideal as in
``telemetry.measurements``, split into the tilt error (angle between the two
body Z axes) and the yaw error (wrapped heading difference).
"""

from __future__ import annotations

import numpy as np

from isaac_drone.core.rotations import quaternion_to_matrix, rotation_log, rotation_to_rpy, wrap_angle
from isaac_drone.core.types import TrajectorySetpoint, VehicleState


def attitude_errors(ideal_quaternion_wxyz, actual_quaternion_wxyz) -> dict:
    """Rotation vector Log(R_actualᵀ R_ideal) [rad] with its angle, tilt and yaw parts."""
    ideal = quaternion_to_matrix(ideal_quaternion_wxyz)
    actual = quaternion_to_matrix(actual_quaternion_wxyz)
    vector = rotation_log(actual.T @ ideal)
    z_ideal, z_actual = ideal[:, 2], actual[:, 2]
    tilt = float(np.arctan2(np.linalg.norm(np.cross(z_actual, z_ideal)), np.dot(z_actual, z_ideal)))
    yaw = float(wrap_angle(rotation_to_rpy(ideal)[2] - rotation_to_rpy(actual)[2]))
    return {
        "attitude_error_rad": vector,
        "attitude_error_angle_rad": float(np.linalg.norm(vector)),
        "tilt_error_rad": tilt,
        "yaw_error_rad": yaw,
    }


def estimation_errors(truth: VehicleState, estimate: VehicleState) -> dict:
    """truth − estimate for position, velocity, angular velocity and the attitude split."""
    position = truth.position_w - estimate.position_w
    velocity = truth.linear_velocity_w - estimate.linear_velocity_w
    rate = truth.angular_velocity_b - estimate.angular_velocity_b
    return {
        "position_w_m": position,
        "position_norm_m": float(np.linalg.norm(position)),
        "velocity_w_m_s": velocity,
        "velocity_norm_m_s": float(np.linalg.norm(velocity)),
        "angular_velocity_b_rad_s": rate,
        "angular_velocity_norm_rad_s": float(np.linalg.norm(rate)),
        **attitude_errors(truth.quaternion_wxyz, estimate.quaternion_wxyz),
    }


def control_errors(reference: TrajectorySetpoint, estimate: VehicleState) -> dict:
    """reference − estimate for position and velocity (the controller's own error signals)."""
    position = reference.position_w - estimate.position_w
    velocity = reference.velocity_w - estimate.linear_velocity_w
    return {
        "position_w_m": position,
        "position_norm_m": float(np.linalg.norm(position)),
        "velocity_w_m_s": velocity,
        "velocity_norm_m_s": float(np.linalg.norm(velocity)),
    }
