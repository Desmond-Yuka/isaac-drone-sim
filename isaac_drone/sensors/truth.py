"""What an ideal sensor would see, from the truth at both ends of one physics step.

Frames follow ``isaac_drone.core.types``: world ENU (Z up), body FLU, body-to-world
wxyz quaternions. Positions/velocities of ``VehicleState`` refer to the CoM, so a
sensor mounted at ``lever_arm_b`` (its position relative to the CoM, body axes)
sees the point kinematics derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaac_drone.core.rotations import quaternion_slerp, quaternion_to_matrix
from isaac_drone.core.types import VehicleState
from isaac_drone.core.validation import finite_array

ATTITUDE_PAIRINGS = ("step_start", "midpoint")
DEFAULT_ATTITUDE_PAIRING = "step_start"


@dataclass(frozen=True)
class StepTruth:
    """Truth at the start and end of one completed physics step.

    The specific force is the interval average f_b = R_pairᵀ(Δv/Δt − g_w); the
    attitude it is paired with is the backend's ``specific_force_attitude_pairing``:
    ``step_start`` for Isaac Lab, whose external forces are applied in the body
    axes of the step start, ``midpoint`` for the synthetic RK4 plant, whose
    stages rotate the thrust with the attitude of each stage. The average
    includes ground support, so a vehicle at rest reads +g along body Z.
    """

    before: VehicleState
    after: VehicleState
    gravity_w: np.ndarray
    attitude_pairing: str = DEFAULT_ATTITUDE_PAIRING

    def __post_init__(self):
        object.__setattr__(self, "gravity_w", finite_array(self.gravity_w, (3,), "gravity_w"))
        if self.attitude_pairing not in ATTITUDE_PAIRINGS:
            raise ValueError(f"attitude_pairing must be one of {ATTITUDE_PAIRINGS}; got {self.attitude_pairing!r}")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("StepTruth needs after.time_s > before.time_s")

    @property
    def dt_s(self) -> float:
        return float(self.after.time_s - self.before.time_s)

    @property
    def rotation_after(self) -> np.ndarray:
        return quaternion_to_matrix(self.after.quaternion_wxyz)

    @property
    def rotation_before(self) -> np.ndarray:
        return quaternion_to_matrix(self.before.quaternion_wxyz)

    @property
    def rotation_pair(self) -> np.ndarray:
        """Body-to-world rotation that the interval-average specific force is expressed with."""
        if self.attitude_pairing == "midpoint":
            return quaternion_to_matrix(quaternion_slerp(self.before.quaternion_wxyz, self.after.quaternion_wxyz, 0.5))
        return self.rotation_before

    @property
    def angular_velocity_b(self) -> np.ndarray:
        """Body angular velocity at the step end (what a gyroscope samples)."""
        return self.after.angular_velocity_b.copy()

    def acceleration_com_w(self) -> np.ndarray:
        """Interval-average CoM acceleration Δv/Δt in world axes (gravity and contact included)."""
        return (self.after.linear_velocity_w - self.before.linear_velocity_w) / self.dt_s

    def specific_force_com_b(self) -> np.ndarray:
        """f_b = R_pairᵀ (Δv/Δt − g_w): the non-gravitational acceleration of the CoM in body axes."""
        return self.rotation_pair.T @ (self.acceleration_com_w() - self.gravity_w)

    def angular_acceleration_b(self) -> np.ndarray:
        """Δ(R ω_b)/Δt rotated into the end body axes (as ``telemetry.measurements``)."""
        omega_before_w = self.rotation_before @ self.before.angular_velocity_b
        omega_after_w = self.rotation_after @ self.after.angular_velocity_b
        return self.rotation_after.T @ ((omega_after_w - omega_before_w) / self.dt_s)

    def specific_force_at_b(self, lever_arm_b) -> np.ndarray:
        """Specific force of the point ``lever_arm_b`` (body axes, relative to the CoM).

        a_point = a_com + α × r + ω × (ω × r); gravity cancels in the difference,
        so the lever-arm terms add directly to the CoM specific force.
        """
        r = finite_array(lever_arm_b, (3,), "lever_arm_b")
        omega = self.angular_velocity_b
        return (
            self.specific_force_com_b()
            + np.cross(self.angular_acceleration_b(), r)
            + np.cross(omega, np.cross(omega, r))
        )

    def point_position_w(self, lever_arm_b) -> np.ndarray:
        """World position of a body point at the step end."""
        r = finite_array(lever_arm_b, (3,), "lever_arm_b")
        return self.after.position_w + self.rotation_after @ r

    def point_velocity_w(self, lever_arm_b) -> np.ndarray:
        """World velocity v + R (ω × r) of a body point at the step end."""
        r = finite_array(lever_arm_b, (3,), "lever_arm_b")
        return self.after.linear_velocity_w + self.rotation_after @ np.cross(self.angular_velocity_b, r)
