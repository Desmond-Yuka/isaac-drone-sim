"""Validated SO(3) helpers; quaternions are scalar-first body-to-world.

All arrays use float64. Bad sensor quaternions are rejected, not silently fixed.
"""

from __future__ import annotations

import numpy as np

from .validation import finite_array, finite_scalar


def hat(vector: np.ndarray) -> np.ndarray:
    """Cross-product matrix: ``hat(a) @ b == cross(a, b)``."""
    x, y, z = finite_array(vector, (3,), "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(skew_matrix: np.ndarray) -> np.ndarray:
    """Axial vector of the skew part, ignoring symmetric roundoff."""
    matrix = finite_array(skew_matrix, (3, 3), "skew_matrix")
    return 0.5 * np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )


def quaternion_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert a unit wxyz quaternion without hiding a non-unit input."""
    q = finite_array(quaternion_wxyz, (4,), "quaternion_wxyz")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"quaternion_wxyz must have unit norm; got {norm}")
    w, x, y, z = q / norm
    return np.array(
        [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion of a rotation matrix, with the scalar part nonnegative."""
    m = validate_rotation(rotation)
    trace = np.trace(m)
    # Shepperd: divide by the largest of the four candidate components.
    if trace >= max(m[0, 0], m[1, 1], m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + trace)
        q = np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw [rad] of a body-to-world rotation, Z-Y-X (yaw first) convention."""
    m = finite_array(rotation, (3, 3), "rotation")
    return np.array([np.arctan2(m[2, 1], m[2, 2]), np.arcsin(np.clip(-m[2, 0], -1.0, 1.0)),
                     np.arctan2(m[1, 0], m[0, 0])])


def wrap_angle(angle):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def validate_rotation(rotation: np.ndarray, name: str = "rotation") -> np.ndarray:
    result = finite_array(rotation, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1e-7) or not np.isclose(
        np.linalg.det(result), 1.0, rtol=0.0, atol=1e-7
    ):
        raise ValueError(f"{name} must be a proper orthonormal rotation matrix")
    return result


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Principal rotation vector, stable at zero and at a half-turn.

    At exactly pi the sign is intrinsically ambiguous. The largest component
    is chosen positive when the skew component cannot determine the sign.
    """
    matrix = validate_rotation(rotation)
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)))
    axial = vee(matrix)
    if angle < 1e-7:
        return axial * (1.0 + angle * angle / 6.0)
    if np.pi - angle < 1e-5:
        _, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        axis = eigenvectors[:, -1]
        if np.linalg.norm(axial) > 1e-10:
            if np.dot(axis, axial) < 0:
                axis = -axis
        elif axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        return axis * angle
    return axial * (angle / np.sin(angle))


def normalized_derivatives(
    vector: np.ndarray, derivative: np.ndarray, second_derivative: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit direction and its first two time derivatives."""
    x = finite_array(vector, (3,), "vector")
    dx = finite_array(derivative, (3,), "derivative")
    ddx = finite_array(second_derivative, (3,), "second_derivative")
    length = np.linalg.norm(x)
    if length < 1e-12:
        raise ValueError("Cannot differentiate a zero-length direction")
    direction = x / length
    dlength = np.dot(direction, dx)
    ddirection = (dx - direction * dlength) / length
    ddlength = np.dot(ddirection, dx) + np.dot(direction, ddx)
    dddirection = (ddx - direction * ddlength - 2.0 * ddirection * dlength) / length
    return direction, ddirection, dddirection


def desired_attitude_kinematics(
    force_w: np.ndarray,
    force_derivative_w: np.ndarray,
    force_second_derivative_w: np.ndarray,
    yaw_rad: float,
    yaw_rate_rad_s: float,
    yaw_acceleration_rad_s2: float,
    fallback_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct ``R_d``, desired-body ``omega_d`` and ``dot(omega_d)``.

    Body +Z follows force; yaw is a world XY auxiliary heading. At zero force
    attitude is held and rates zeroed, since force has no defined direction.
    At the heading singularity a previous transverse axis (or a least-aligned
    coordinate axis) provides a finite frame. That branch holds its auxiliary
    heading and cannot realize the requested yaw derivatives.
    """
    force = finite_array(force_w, (3,), "force_w")
    dforce = finite_array(force_derivative_w, (3,), "force_derivative_w")
    ddforce = finite_array(force_second_derivative_w, (3,), "force_second_derivative_w")
    yaw = finite_scalar(yaw_rad, "yaw_rad")
    yaw_rate = finite_scalar(yaw_rate_rad_s, "yaw_rate_rad_s")
    yaw_acceleration = finite_scalar(yaw_acceleration_rad_s2, "yaw_acceleration_rad_s2")
    fallback = np.eye(3) if fallback_rotation is None else validate_rotation(fallback_rotation)
    if np.linalg.norm(force) < 1e-10:
        return fallback.copy(), np.zeros(3), np.zeros(3)
    z, dz, ddz = normalized_derivatives(force, dforce, ddforce)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    perpendicular = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    dheading = perpendicular * yaw_rate
    ddheading = perpendicular * yaw_acceleration - heading * yaw_rate**2
    if np.linalg.norm(np.cross(z, heading)) < 1e-8:
        heading = fallback[:, 0].copy()
        if np.linalg.norm(np.cross(z, heading)) < 1e-8:
            heading = np.eye(3)[int(np.argmin(np.abs(z)))]
        dheading = np.zeros(3)
        ddheading = np.zeros(3)
    cross = np.cross(z, heading)
    dcross = np.cross(dz, heading) + np.cross(z, dheading)
    ddcross = np.cross(ddz, heading) + 2.0 * np.cross(dz, dheading) + np.cross(z, ddheading)
    y, dy, ddy = normalized_derivatives(cross, dcross, ddcross)
    x = np.cross(y, z)
    dx = np.cross(dy, z) + np.cross(y, dz)
    ddx = np.cross(ddy, z) + 2.0 * np.cross(dy, dz) + np.cross(y, ddz)
    rotation = np.column_stack((x, y, z))
    derivative = np.column_stack((dx, dy, dz))
    second_derivative = np.column_stack((ddx, ddy, ddz))
    omega = vee(rotation.T @ derivative)
    alpha = vee(derivative.T @ derivative + rotation.T @ second_derivative)
    return rotation, omega, alpha
