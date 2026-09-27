"""Validation and frame helpers shared by environmental models."""

from __future__ import annotations

from numbers import Real

import numpy as np

from isaac_drone.types import finite_array


def finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not a boolean/string")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def numeric_array(value, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    """Do not silently coerce configuration strings or booleans to numbers."""
    def numeric(item):
        if isinstance(item, np.ndarray):
            return item.dtype.kind in "fiu"
        if isinstance(item, (list, tuple)):
            return all(numeric(child) for child in item)
        return isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
    if not numeric(value):
        raise ValueError(f"{name} must contain real numbers, not booleans/strings")
    if shape is not None:
        return finite_array(value, shape, name)
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def positive_dt(dt_s: float) -> float:
    value = finite_scalar(dt_s, "dt_s")
    if value <= 0:
        raise ValueError("dt_s must be finite and positive")
    return value


def body_to_world(quaternion_wxyz) -> np.ndarray:
    """Return an active body-to-world rotation; reject invalid orientations."""
    q = finite_array(quaternion_wxyz, (4,), "quaternion_wxyz")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("quaternion_wxyz must be unit length")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def dissipative_matrix(value, name: str) -> np.ndarray:
    """A damping matrix may couple axes; its symmetric part must be PSD."""
    matrix = numeric_array(value, (3, 3), name)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() < -1e-12 * scale:
        raise ValueError(f"{name} must have a positive-semidefinite symmetric part")
    matrix.setflags(write=False)
    return matrix
