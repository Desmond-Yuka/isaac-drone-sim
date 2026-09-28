"""Validation and frame helpers shared by environmental models."""

from __future__ import annotations

import numpy as np

from isaac_drone.core.rotations import quaternion_to_matrix as body_to_world
from isaac_drone.core.validation import finite_array, finite_scalar, numeric_array, positive_dt

__all__ = ["body_to_world", "dissipative_matrix", "finite_array", "finite_scalar", "numeric_array", "positive_dt"]


def dissipative_matrix(value, name: str) -> np.ndarray:
    """A damping matrix may couple axes; its symmetric part must be PSD."""
    matrix = numeric_array(value, (3, 3), name)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() < -1e-12 * scale:
        raise ValueError(f"{name} must have a positive-semidefinite symmetric part")
    matrix.setflags(write=False)
    return matrix
