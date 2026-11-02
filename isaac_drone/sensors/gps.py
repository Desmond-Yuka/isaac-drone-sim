"""GNSS receiver after MATLAB ``gpsSensor``, reporting local ENU position and velocity of the antenna.

Position error per axis: e(k) = DecayFactor·e(k−1) + σ·w(k) with σ the horizontal
(x, y) or vertical (z) accuracy; velocity error: white noise with the velocity
accuracy. Two parameter profiles stand in for the Pixhawk 6C receivers: ``rtk``
(ZED-F9P fixed solution) and ``standard`` (single-point u-blox M8 class); any
numeric field given explicitly overrides its profile value. The receiver's
latency is the suite's delay queue, so a reading carries its own ``time_s``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isaac_drone.core.types import MassProperties
from isaac_drone.core.validation import finite_scalar

from .mounting import Mounting
from .noise import DecayingNoise
from .truth import StepTruth

GPS_NUMERIC_FIELDS = ("horizontal_accuracy_m", "vertical_accuracy_m", "velocity_accuracy_m_s", "decay_factor")
GPS_FIELDS = ("enabled", "rate_hz", "delay_s", "position_b_m", "profile", *GPS_NUMERIC_FIELDS)
# σ is the driving noise of the MATLAB recursion; with a = decay_factor the stationary error is σ/√(1−a²)
# and the correlation time −Δt/ln a. At 10 Hz: rtk 0.023 m horizontal, τ ≈ 0.95 s; standard 2.5 m, τ ≈ 10 s.
GPS_PROFILES = {
    "rtk": {
        "horizontal_accuracy_m": 0.01,
        "vertical_accuracy_m": 0.02,
        "velocity_accuracy_m_s": 0.05,
        "decay_factor": 0.9,
    },
    "standard": {
        "horizontal_accuracy_m": 0.35,
        "vertical_accuracy_m": 0.7,
        "velocity_accuracy_m_s": 0.1,
        "decay_factor": 0.99,
    },
}


def resolve_gps_parameters(section: Mapping, path: str = "sensors.gps") -> dict:
    """Profile values with explicit (non-null) numeric fields overriding them."""
    profile = section["profile"]
    if profile not in GPS_PROFILES:
        raise ValueError(f"{path}.profile must be one of {sorted(GPS_PROFILES)}; got {profile!r}")
    values = dict(GPS_PROFILES[profile])
    for field in GPS_NUMERIC_FIELDS:
        if section.get(field) is not None:
            values[field] = finite_scalar(section[field], f"{path}.{field}")
    for field in GPS_NUMERIC_FIELDS[:3]:
        if values[field] < 0:
            raise ValueError(f"{path}.{field} must be nonnegative")
    if not 0.0 <= values["decay_factor"] <= 1.0:
        raise ValueError(f"{path}.decay_factor must lie in [0, 1]")
    return values


class GpsSensor:
    name = "gps"

    def __init__(self, section: Mapping, path: str = "sensors.gps"):
        self.rate_hz = finite_scalar(section["rate_hz"], f"{path}.rate_hz")
        self.mounting = Mounting(section["position_b_m"])
        self.profile = section["profile"]
        self.parameters = resolve_gps_parameters(section, path)
        horizontal, vertical = self.parameters["horizontal_accuracy_m"], self.parameters["vertical_accuracy_m"]
        self.position_error = DecayingNoise([horizontal, horizontal, vertical], self.parameters["decay_factor"])
        self.velocity_std = self.parameters["velocity_accuracy_m_s"]
        self._rng = None

    def bind(self, mass: MassProperties, gravity_w) -> None:
        self.mounting.bind(mass)

    def reset(self, seed: np.random.SeedSequence) -> None:
        self._rng = np.random.default_rng(seed)
        self.position_error.reset()

    def measure(self, truth: StepTruth, telemetry: Mapping) -> dict:
        if self.mounting.lever_arm_b is None:
            raise RuntimeError("bind the GPS antenna to the vehicle mass properties before sampling")
        position = truth.point_position_w(self.mounting.lever_arm_b)
        velocity = truth.point_velocity_w(self.mounting.lever_arm_b)
        return {
            "time_s": float(truth.after.time_s),
            "position_w_m": position + self.position_error.step(self._rng),
            "velocity_w_m_s": velocity + self.velocity_std * self._rng.standard_normal(3),
            "true_position_w_m": position,
            "true_velocity_w_m_s": velocity,
        }

    def describe(self) -> dict:
        return {
            "rate_hz": self.rate_hz,
            "mounting": self.mounting.describe(),
            "profile": self.profile,
            **self.parameters,
            "position_error_stationary_std_m": self.position_error.stationary_std.tolist(),
        }
