"""Identifiable lumped drag models with explicit coefficient/frame conventions."""
from __future__ import annotations
import numpy as np
from isaac_drone.types import VehicleState, Wrench
from .math import body_to_world, dissipative_matrix, numeric_array as finite_array, positive_dt
from .wind import WindField


class BodyDrag:
    """Body-axis drag at one measured aerodynamic centre.

    F_b = -D v_rel,b - q * |v_rel,b| * v_rel,b;
    M_b = r_b x F_b - D_omega omega_b.

    D and D_omega are full dissipative matrices. q contains nonnegative per-axis
    aggregate coefficients (including density/area only if included by the
    calibration); no shape, density, lift, rotor wash or ground effect is guessed.
    The point velocity includes omega x r. Its wind is sampled at the same world
    position, so shear and rotation are not silently discarded. r_b is measured
    from CoM. Angular drag uses inertial angular velocity; atmospheric vorticity
    coupling requires a separately calibrated plugin.
    """
    def __init__(self, *, linear_drag_b_kg_s, quadratic_drag_b_kg_m,
                 angular_linear_drag_b_nm_s, center_of_pressure_b_m):
        self.linear = dissipative_matrix(linear_drag_b_kg_s, "linear_drag_b_kg_s")
        self.quadratic = finite_array(quadratic_drag_b_kg_m, (3,), "quadratic_drag_b_kg_m")
        if np.any(self.quadratic < 0):
            raise ValueError("quadratic_drag_b_kg_m must be nonnegative")
        self.angular = dissipative_matrix(angular_linear_drag_b_nm_s, "angular_linear_drag_b_nm_s")
        self.lever_b = finite_array(center_of_pressure_b_m, (3,), "center_of_pressure_b_m")

    def reset(self, seed: int | None = None) -> None:
        """Stateless model; provided for the effect plugin contract."""

    def evaluate(self, state: VehicleState, dt_s: float,
                 wind: WindField | None = None) -> Wrench:
        positive_dt(dt_s)
        rotation = body_to_world(state.quaternion_wxyz)
        point_w = state.position_w + rotation @ self.lever_b
        if wind is None:
            raise ValueError("BodyDrag requires an explicit WindField; still air must be explicitly configured")
        wind_w = finite_array(wind.velocity_w(point_w, state.time_s), (3,), "wind velocity")
        point_velocity_b = rotation.T @ state.linear_velocity_w + np.cross(
            state.angular_velocity_b, self.lever_b)
        relative_b = point_velocity_b - rotation.T @ wind_w
        force = -self.linear @ relative_b - self.quadratic * np.abs(relative_b) * relative_b
        torque = np.cross(self.lever_b, force) - self.angular @ state.angular_velocity_b
        return Wrench(force_b=force, torque_b=torque)
