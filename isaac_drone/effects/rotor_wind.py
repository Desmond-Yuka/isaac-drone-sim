"""Wind acting on the rotors, split by the rotor geometry in body axes.

The wind component along each rotor's thrust axis changes that rotor's thrust.
It is applied after the motor model as an equivalent rotor-speed offset,

    dOmega_i = c / 2 * Omega_hover_i * u_axial_i,

so the thrust and reaction torque of rotor i follow k_f,i (Omega_i + dOmega_i)^2.
c is the relative thrust change per m/s of axial wind at hover (positive for air
moving along the thrust axis, i.e. an updraft through an upright disk); the
linearized momentum/blade-element result is c = kappa / v_hover. The offset does
not depend on the current rotor speed, so the thrust change scales with Omega
rather than with thrust. Rotor speed cannot produce a force in the rotor-disk
plane: that wind component is applied separately as a drag force k * u_inplane
at the CoM, along the wind. Only the wind itself enters, not the vehicle's own
motion through the air, so still air leaves a run unchanged. Both coefficients
are explicit configuration; no typical-vehicle value is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaac_drone.core.types import VehicleState, Wrench
from isaac_drone.effects._math import body_to_world, finite_scalar, positive_dt
from isaac_drone.effects._math import numeric_array as finite_array
from isaac_drone.effects.wind import WindField


@dataclass(frozen=True)
class RotorGeometry:
    """Actual rotor layout of the plant, in body axes.

    ``positions_b`` are the rotor thrust points relative to the root link (the
    backend's ``rotor_positions_b``) and ``com_b`` the vehicle CoM in the same
    frame; ``axes_b`` are the unit thrust axes and ``hover_rps`` each rotor's
    speed when the four rotors share the weight equally.
    """

    positions_b: np.ndarray
    com_b: np.ndarray
    axes_b: np.ndarray
    hover_rps: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "positions_b", finite_array(self.positions_b, (4, 3), "rotor positions_b"))
        object.__setattr__(self, "com_b", finite_array(self.com_b, (3,), "com_b"))
        axes = finite_array(self.axes_b, (4, 3), "rotor axes_b")
        if not np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-6):
            raise ValueError("rotor axes_b must be unit vectors")
        object.__setattr__(self, "axes_b", axes)
        hover = finite_array(self.hover_rps, (4,), "rotor hover_rps")
        if np.any(hover <= 0):
            raise ValueError("rotor hover_rps must be positive")
        object.__setattr__(self, "hover_rps", hover)


class RotorWind:
    """Axial wind -> per-rotor speed offsets; in-plane wind -> drag force at the CoM.

    A coefficient of None leaves that channel off. The axial channel needs the
    plant geometry (``bind_rotor_geometry``), which the runtime supplies at reset.
    The disk plane is normal to the mean rotor axis (body +Z before binding; the
    flight controllers require every rotor axis along body +Z).
    """

    def __init__(self, *, axial_thrust_gain_per_m_s=None, inplane_drag_n_per_m_s=None):
        self.gain = (
            None
            if axial_thrust_gain_per_m_s is None
            else finite_scalar(axial_thrust_gain_per_m_s, "rotor_wind.axial.thrust_gain_per_m_s")
        )
        self.drag = (
            None
            if inplane_drag_n_per_m_s is None
            else finite_scalar(inplane_drag_n_per_m_s, "rotor_wind.inplane_drag.drag_n_per_m_s")
        )
        if self.drag is not None and self.drag < 0:
            raise ValueError("rotor_wind.inplane_drag.drag_n_per_m_s must be nonnegative")
        if self.gain is None and self.drag is None:
            raise ValueError("rotor wind needs at least one channel (axial or in-plane drag)")
        self.geometry = None
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Deterministic model; clears the last step's diagnostics."""
        self._last = {}

    def bind_rotor_geometry(self, geometry: RotorGeometry) -> None:
        if not isinstance(geometry, RotorGeometry):
            raise TypeError("rotor wind needs a RotorGeometry")
        self.geometry = geometry

    def _disk_normal_b(self) -> np.ndarray:
        if self.geometry is None:
            return np.array([0.0, 0.0, 1.0])
        mean = self.geometry.axes_b.mean(axis=0)
        return mean / np.linalg.norm(mean)

    def evaluate(self, state: VehicleState, dt_s: float, wind: WindField | None = None) -> Wrench:
        """In-plane drag at the CoM; zero when that channel is off."""
        positive_dt(dt_s)
        if wind is None:
            raise ValueError("RotorWind requires an explicit WindField; still air must be explicitly configured")
        self._last = {}
        if self.drag is None:
            return Wrench.zero()
        rotation = body_to_world(state.quaternion_wxyz)
        wind_b = rotation.T @ finite_array(wind.velocity_w(state.position_w, state.time_s), (3,), "wind velocity")
        normal = self._disk_normal_b()
        force = self.drag * (wind_b - (normal @ wind_b) * normal)
        self._last["force_wind_drag_b_n"] = force
        return Wrench(force_b=force, torque_b=np.zeros(3))

    def rotor_speed_offsets(self, state: VehicleState, wind: WindField | None) -> np.ndarray | None:
        """Equivalent rotor-speed offsets [rev/s] in rotor order, or None when the axial channel is off."""
        if self.gain is None:
            return None
        if self.geometry is None:
            raise RuntimeError("bind the plant rotor geometry before evaluating the axial rotor wind")
        if wind is None:
            raise ValueError("RotorWind requires an explicit WindField; still air must be explicitly configured")
        rotation = body_to_world(state.quaternion_wxyz)
        points_w = state.position_w + (self.geometry.positions_b - self.geometry.com_b) @ rotation.T
        wind_b = np.array(
            [
                rotation.T @ finite_array(wind.velocity_w(point, state.time_s), (3,), "wind velocity")
                for point in points_w
            ]
        )
        axial = np.einsum("ij,ij->i", wind_b, self.geometry.axes_b)
        offsets = 0.5 * self.gain * self.geometry.hover_rps * axial
        self._last["rotor_axial_wind_m_s"] = axial
        self._last["rotor_speed_offset_rps"] = offsets
        return offsets.copy()

    def diagnostics(self) -> dict:
        """Last step's drag force, axial wind per rotor and speed offsets, for telemetry."""
        return {key: value.copy() for key, value in self._last.items()}
