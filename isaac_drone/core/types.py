"""Simulator-independent contracts. SI units; world Z-up; quaternion order wxyz.

Positions/linear velocities refer to the vehicle centre of mass (CoM). Body
axes have the orientation of the USD root link. Wrenches and inertias refer to
the CoM, not the link origin. Arrays describe one vehicle, not a training batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .validation import finite_array


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    inertia_com_b: np.ndarray
    com_b: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be finite and positive")
        inertia = finite_array(self.inertia_com_b, (3, 3), "inertia_com_b")
        if not np.allclose(inertia, inertia.T, atol=1e-10, rtol=1e-7):
            raise ValueError("inertia_com_b must be symmetric")
        eigenvalues = np.linalg.eigvalsh(inertia)
        if eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum() + 1e-8:
            raise ValueError("inertia_com_b must be positive definite and physically realizable")
        object.__setattr__(self, "inertia_com_b", inertia)
        object.__setattr__(self, "com_b", finite_array(self.com_b, (3,), "com_b"))


@dataclass(frozen=True)
class VehicleState:
    time_s: float
    position_w: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_w: np.ndarray
    angular_velocity_b: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("time_s must be finite and nonnegative")
        for name in ("position_w", "linear_velocity_w", "angular_velocity_b"):
            object.__setattr__(self, name, finite_array(getattr(self, name), (3,), name))
        quat = finite_array(self.quaternion_wxyz, (4,), "quaternion_wxyz")
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5):
            raise ValueError("quaternion_wxyz must be a unit quaternion")
        object.__setattr__(self, "quaternion_wxyz", quat / np.linalg.norm(quat))


@dataclass(frozen=True)
class TrajectorySetpoint:
    """CoM reference in world axes and heading [rad].

    Position/velocity/acceleration use m, m/s, m/s². Optional paired jerk_w
    and snap_w use m/s³ and m/s⁴; None means unavailable, never assumed zero.
    """

    position_w: np.ndarray
    velocity_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rad: float = 0.0
    yaw_rate_rad_s: float = 0.0
    yaw_acceleration_rad_s2: float = 0.0
    jerk_w: np.ndarray | None = None
    snap_w: np.ndarray | None = None

    def __post_init__(self):
        for name in ("position_w", "velocity_w", "acceleration_w"):
            object.__setattr__(self, name, finite_array(getattr(self, name), (3,), name))
        if (self.jerk_w is None) != (self.snap_w is None):
            raise ValueError("jerk_w and snap_w must be supplied together")
        for name in ("jerk_w", "snap_w"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, finite_array(getattr(self, name), (3,), name))
        for name in ("yaw_rad", "yaw_rate_rad_s", "yaw_acceleration_rad_s2"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class Wrench:
    """Force [N] and moment [N m] about the vehicle CoM, in body axes."""

    force_b: np.ndarray
    torque_b: np.ndarray

    def __post_init__(self):
        for name in ("force_b", "torque_b"):
            object.__setattr__(self, name, finite_array(getattr(self, name), (3,), name))

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate((self.force_b, self.torque_b))

    @classmethod
    def zero(cls) -> Wrench:
        return cls(np.zeros(3), np.zeros(3))

    @classmethod
    def from_vector(cls, vector) -> Wrench:
        values = finite_array(vector, (6,), "wrench")
        return cls(values[:3], values[3:])

    def __add__(self, other: Wrench) -> Wrench:
        if not isinstance(other, Wrench):
            return NotImplemented
        return Wrench(self.force_b + other.force_b, self.torque_b + other.torque_b)
