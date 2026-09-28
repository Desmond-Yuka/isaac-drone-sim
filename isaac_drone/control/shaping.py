"""Command shaping shared by controllers: acceleration limit and thrust-vector tilt cone."""

from __future__ import annotations

import numpy as np


def limited_thrust_vector(
    acceleration_w: np.ndarray,
    mass_kg: float,
    gravity_w: np.ndarray,
    max_tilt_rad: float,
    max_acceleration_m_s2: float | None = None,
) -> tuple[np.ndarray, bool]:
    """World thrust force ``m (a - g)`` for a commanded net acceleration, within the limits.

    The acceleration norm is first scaled to ``max_acceleration_m_s2`` (if set);
    the force is then orthogonally projected onto the positive-thrust cone of
    half-angle ``max_tilt_rad`` about world +Z (including its apex). Returns the
    force and whether any limit changed the command.
    """
    limited = False
    magnitude = np.linalg.norm(acceleration_w)
    if max_acceleration_m_s2 is not None and magnitude > max_acceleration_m_s2:
        acceleration_w = acceleration_w * (max_acceleration_m_s2 / magnitude)
        limited = True
    force = mass_kg * (acceleration_w - gravity_w)
    horizontal = np.linalg.norm(force[:2])
    sine, cosine = np.sin(max_tilt_rad), np.cos(max_tilt_rad)
    if force[2] >= 0 and horizontal * cosine <= force[2] * sine:
        return force, limited
    # Orthogonal projection onto a circular cone, including its apex.
    projection = max(0.0, sine * horizontal + cosine * force[2])
    direction = np.zeros(3)
    if horizontal > 0:
        direction[:2] = sine * force[:2] / horizontal
    direction[2] = cosine
    return projection * direction, True
