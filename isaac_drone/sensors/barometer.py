"""Barometric altimeter after MATLAB ``altimeterSensor``: altitude of the mounting point plus noise.

reading = z_point + ConstantBias + w·√(f_s/s)·NoiseDensity + β(k), with the bias
instability β(k) = DecayFactor·β(k−1) + BiasInstability·w(k) (0: white, 1: random walk).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isaac_drone.core.types import MassProperties
from isaac_drone.core.validation import finite_scalar

from .mounting import Mounting
from .noise import DecayingNoise, white_noise_std
from .truth import StepTruth

BAROMETER_FIELDS = (
    "enabled",
    "rate_hz",
    "delay_s",
    "position_b_m",
    "constant_bias_m",
    "noise_density_m_sqrt_hz",
    "noise_type",
    "bias_instability_m",
    "decay_factor",
)


class BarometerSensor:
    name = "barometer"

    def __init__(self, section: Mapping, path: str = "sensors.barometer"):
        self.rate_hz = finite_scalar(section["rate_hz"], f"{path}.rate_hz")
        self.mounting = Mounting(section["position_b_m"])
        self.constant_bias = finite_scalar(section["constant_bias_m"], f"{path}.constant_bias_m")
        self.white_std = float(
            white_noise_std(section["noise_density_m_sqrt_hz"], self.rate_hz, section["noise_type"], size=1)[0]
        )
        self.bias_instability = DecayingNoise(section["bias_instability_m"], section["decay_factor"], size=1)
        self._rng = None

    def bind(self, mass: MassProperties, gravity_w) -> None:
        self.mounting.bind(mass)

    def reset(self, seed: np.random.SeedSequence) -> None:
        self._rng = np.random.default_rng(seed)
        self.bias_instability.reset()

    def measure(self, truth: StepTruth, telemetry: Mapping) -> dict:
        if self.mounting.lever_arm_b is None:
            raise RuntimeError("bind the barometer to the vehicle mass properties before sampling")
        altitude = float(truth.point_position_w(self.mounting.lever_arm_b)[2])
        noise = self.white_std * float(self._rng.standard_normal()) + float(self.bias_instability.step(self._rng)[0])
        return {
            "time_s": float(truth.after.time_s),
            "altitude_m": altitude + self.constant_bias + noise,
            "true_altitude_m": altitude,
        }

    def describe(self) -> dict:
        return {
            "rate_hz": self.rate_hz,
            "mounting": self.mounting.describe(),
            "constant_bias_m": self.constant_bias,
            "white_noise_std_m": self.white_std,
            "bias_instability_m": float(self.bias_instability.std[0]),
            "decay_factor": self.bias_instability.decay,
            "bias_instability_stationary_std_m": float(self.bias_instability.stationary_std[0]),
        }
