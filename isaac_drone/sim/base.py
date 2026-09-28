"""Contracts between the control runtime and a physics backend.

A backend owns one vehicle's physics and motor dynamics. A driver advances the
simulator by exactly one physics step between ``MotionControlLoop.prepare_step``
and ``finish_step``. Frames and units follow ``isaac_drone.core.types``.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from isaac_drone.core.types import MassProperties, VehicleState, Wrench


class VehicleBackend(Protocol):
    """A single vehicle's physics bridge; apply exactly once per physics tick.

    ``allocation_matrix_b`` (6x4) maps rotor thrusts [N] to the body wrench about
    the actual CoM. Optional: ``ground_start_position(ground_z_m, clearance_m)``
    for ground launches, ``assert_control_compatible()`` and ``describe()``.
    """

    allocation_matrix_b: np.ndarray

    def reset(self, initial_position_w: np.ndarray | None = None) -> None: ...

    def read_state(self, time_s: float) -> VehicleState: ...

    def mass_properties(self) -> MassProperties: ...

    def motor_parameters(self) -> dict: ...

    def thrust_to_motor_speeds(self, thrusts_n: np.ndarray) -> np.ndarray: ...

    def apply_motor_speeds(self, target_rps: np.ndarray, external_wrench: Wrench) -> None: ...

    def telemetry(self) -> dict: ...


class StateProvider(Protocol):
    """Sensor/estimator boundary; must use the shared CoM/frame contract."""

    def reset(self) -> None: ...

    def read_state(self, time_s: float) -> VehicleState: ...


class SimulationDriver(Protocol):
    """Advances physics for the experiment runner."""

    def step(self) -> None:
        """Advance physics (and backend state buffers) by exactly one dt."""

    def is_running(self) -> bool:
        """False once the application or simulation has been closed/stopped."""

    def is_paused(self) -> bool:
        """True while the user has paused the simulation (no physics may advance)."""

    def idle(self) -> None:
        """Keep the application responsive while paused."""
