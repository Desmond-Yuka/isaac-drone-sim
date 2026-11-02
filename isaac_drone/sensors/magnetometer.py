"""Magnetometer: the configured Earth field rotated into sensor axes, plus motor-current interference.

The Earth field is a constant world-frame vector (ENU, µT). Motor currents add
a body-fixed disturbance proportional to Σ T_i^1.5 of the applied rotor thrusts
(current ∝ Ω³ ∝ T^1.5). The sum passes through the ``magparams`` error chain.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isaac_drone.core.types import MassProperties
from isaac_drone.core.validation import finite_array, finite_scalar

from .mounting import Mounting
from .triaxial import TriaxialModel, TriaxialParams
from .truth import StepTruth

MAGNETOMETER_FIELDS = (
    "enabled",
    "rate_hz",
    "delay_s",
    "orientation_rpy_rad",
    "temperature_c",
    "field_w_ut",
    "motor_interference_b_ut_per_n15",
    "model",
)


class MagnetometerSensor:
    name = "magnetometer"

    def __init__(self, section: Mapping, path: str = "sensors.magnetometer"):
        self.rate_hz = finite_scalar(section["rate_hz"], f"{path}.rate_hz")
        self.mounting = Mounting((0.0, 0.0, 0.0), section["orientation_rpy_rad"])
        self.field_w = finite_array(section["field_w_ut"], (3,), f"{path}.field_w_ut")
        self.interference_b = finite_array(
            section["motor_interference_b_ut_per_n15"], (3,), f"{path}.motor_interference_b_ut_per_n15"
        )
        self.model = TriaxialModel(
            TriaxialParams.from_config(section["model"], f"{path}.model"),
            self.rate_hz,
            finite_scalar(section["temperature_c"], f"{path}.temperature_c"),
        )

    def bind(self, mass: MassProperties, gravity_w) -> None:
        self.mounting.bind(mass)

    def reset(self, seed: np.random.SeedSequence) -> None:
        self.model.reset(np.random.default_rng(seed))

    def motor_interference_b(self, telemetry: Mapping) -> np.ndarray:
        thrust = telemetry.get("applied_thrust_n")
        if thrust is None:
            return np.zeros(3)
        return self.interference_b * float(
            np.sum(np.maximum(finite_array(thrust, (4,), "applied_thrust_n"), 0.0) ** 1.5)
        )

    def measure(self, truth: StepTruth, telemetry: Mapping) -> dict:
        earth_b = truth.rotation_after.T @ self.field_w
        interference_s = self.mounting.to_sensor(self.motor_interference_b(telemetry))
        field_s = self.mounting.to_sensor(earth_b) + interference_s
        return {
            "time_s": float(truth.after.time_s),
            "magnetic_field_ut": self.model.measure(field_s),
            "true_magnetic_field_s_ut": field_s,
            "motor_interference_s_ut": interference_s,
        }

    def describe(self) -> dict:
        return {
            "rate_hz": self.rate_hz,
            "field_w_ut": self.field_w.tolist(),
            "motor_interference_b_ut_per_n15": self.interference_b.tolist(),
            "orientation_rpy_rad": self.mounting.orientation_rpy.tolist(),
            "white_noise_std_ut": self.model.white_std.tolist(),
        }
