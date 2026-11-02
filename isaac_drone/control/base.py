"""Controller contract, the vehicle context controllers are built with, and their registry.

A controller maps (state, setpoint) to either a body wrench about the CoM
(``output = "wrench"``; the runtime allocates it to rotor thrusts within the
current bounds) or directly to per-rotor thrusts (``output = "rotor_thrust"``;
e.g. a learned policy; the runtime only clips them to the bounds). It is
called at the control rate with the control period as ``dt_s``.

Add a controller by writing a parameter dataclass and a factory::

    @CONTROLLERS.register("my_controller", params=MyParams)
    def build(params: MyParams, *, vehicle: VehicleModel, limits: FlightLimits) -> Controller:
        return MyController(params, vehicle, limits)

and select it with ``controller: {kind: my_controller, ...params}`` in YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from isaac_drone.core.params import parse_params
from isaac_drone.core.registry import Registry
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench
from isaac_drone.core.validation import ConfigurationError

from .allocation import AllocationResult

CONTROLLERS = Registry("controller")


@dataclass(frozen=True)
class FlightLimits:
    """Envelope shared by controllers and the trajectory planner (``limits:`` in YAML).

    max_tilt_rad: thrust-axis tilt from world +Z, below pi/2 (upright flight).
    max_yaw_rate_rad_s: yaw-rate feedforward/planning limit.
    max_acceleration_m_s2: commanded net translational acceleration, or None.
    """

    max_tilt_rad: float
    max_yaw_rate_rad_s: float
    max_acceleration_m_s2: float | None

    def __post_init__(self):
        if not 0 < self.max_tilt_rad < np.pi / 2:
            raise ValueError("max_tilt_rad must satisfy 0 < max_tilt_rad < pi/2 for upright flight")
        if self.max_yaw_rate_rad_s <= 0:
            raise ValueError("max_yaw_rate_rad_s must be positive")
        if self.max_acceleration_m_s2 is not None and self.max_acceleration_m_s2 <= 0:
            raise ValueError("max_acceleration_m_s2 must be positive or null")

    @classmethod
    def from_config(cls, section) -> FlightLimits:
        return parse_params(cls, section, "limits")


@dataclass(frozen=True)
class VehicleModel:
    """Actual vehicle quantities resolved by the backend after reset (never hand-copied constants)."""

    mass_properties: MassProperties
    gravity_w: np.ndarray
    allocation_matrix_b: np.ndarray
    thrust_min_n: np.ndarray
    thrust_max_n: np.ndarray
    control_dt_s: float


@dataclass(frozen=True)
class ControllerDiagnostics:
    """Held desired attitude/rate for telemetry, plus JSON-ready controller-specific details."""

    desired_rotation_w: np.ndarray | None = None
    desired_angular_velocity_d: np.ndarray | None = None
    details: dict = field(default_factory=dict)


class Controller(Protocol):
    output: Literal["wrench", "rotor_thrust"]

    def reset(self) -> None: ...

    def compute(self, state: VehicleState, setpoint: TrajectorySetpoint, dt_s: float) -> Wrench | np.ndarray: ...

    def notify_allocation(self, result: AllocationResult) -> None:
        """Achieved thrusts/wrench of the last command (anti-windup); may ignore it."""

    def diagnostics(self) -> ControllerDiagnostics: ...


def build_controller(section, *, vehicle: VehicleModel, limits: FlightLimits) -> Controller:
    controller = CONTROLLERS.build(section, "controller", vehicle=vehicle, limits=limits)
    if getattr(controller, "output", None) not in ("wrench", "rotor_thrust"):
        raise ConfigurationError(f"controller {section['kind']!r} must declare output 'wrench' or 'rotor_thrust'")
    return controller
