"""Inertial measurement unit: accelerometer and gyroscope at one mounting point.

The accelerometer reads the specific force of its mounting point (lever-arm
terms included) in sensor axes; a vehicle at rest reads +g along body Z. The
gyroscope reads the body angular velocity at the step end. Both pass through
the MATLAB ``imuSensor`` error chain (``triaxial.py``). Beyond MATLAB, an
optional vibration term adds white noise whose standard deviation grows with
the total rotor thrust relative to the vehicle weight, standing in for the
rotor vibration a rigid-body simulator does not produce.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from isaac_drone.core.types import MassProperties
from isaac_drone.core.validation import finite_array, finite_scalar

from .mounting import Mounting
from .triaxial import TriaxialModel, TriaxialParams
from .truth import StepTruth

IMU_FIELDS = (
    "enabled",
    "rate_hz",
    "delay_s",
    "position_b_m",
    "orientation_rpy_rad",
    "temperature_c",
    "accelerometer",
    "gyroscope",
    "vibration",
)
VIBRATION_FIELDS = ("enabled", "accelerometer_std_m_s2", "gyroscope_std_rad_s", "scale_with_thrust")


class ImuSensor:
    name = "imu"

    def __init__(self, section: Mapping, path: str = "sensors.imu"):
        self.rate_hz = finite_scalar(section["rate_hz"], f"{path}.rate_hz")
        self.mounting = Mounting(section["position_b_m"], section["orientation_rpy_rad"])
        temperature = finite_scalar(section["temperature_c"], f"{path}.temperature_c")
        self.accelerometer = TriaxialModel(
            TriaxialParams.from_config(section["accelerometer"], f"{path}.accelerometer"), self.rate_hz, temperature
        )
        self.gyroscope = TriaxialModel(
            TriaxialParams.from_config(section["gyroscope"], f"{path}.gyroscope", gyroscope=True),
            self.rate_hz,
            temperature,
        )
        vibration = section["vibration"]
        if not isinstance(vibration, Mapping) or set(vibration) != set(VIBRATION_FIELDS):
            raise ValueError(f"{path}.vibration must have exactly the fields {VIBRATION_FIELDS}")
        self.vibration_enabled = bool(vibration["enabled"])
        self.vibration_accelerometer = finite_array(
            np.broadcast_to(np.asarray(vibration["accelerometer_std_m_s2"], dtype=float), (3,)),
            (3,),
            f"{path}.vibration.accelerometer_std_m_s2",
        )
        self.vibration_gyroscope = finite_array(
            np.broadcast_to(np.asarray(vibration["gyroscope_std_rad_s"], dtype=float), (3,)),
            (3,),
            f"{path}.vibration.gyroscope_std_rad_s",
        )
        if np.any(self.vibration_accelerometer < 0) or np.any(self.vibration_gyroscope < 0):
            raise ValueError(f"{path}.vibration standard deviations must be nonnegative")
        self.vibration_scales_with_thrust = bool(vibration["scale_with_thrust"])
        self.weight_n = None
        self._rng = None

    def bind(self, mass: MassProperties, gravity_w) -> None:
        self.mounting.bind(mass)
        self.weight_n = mass.mass_kg * float(np.linalg.norm(finite_array(gravity_w, (3,), "gravity_w")))

    def reset(self, seed: np.random.SeedSequence) -> None:
        accelerometer, gyroscope, vibration = seed.spawn(3)
        self.accelerometer.reset(np.random.default_rng(accelerometer))
        self.gyroscope.reset(np.random.default_rng(gyroscope))
        self._rng = np.random.default_rng(vibration)

    def _vibration_scale(self, telemetry: Mapping) -> float:
        if not self.vibration_scales_with_thrust or self.weight_n is None:
            return 1.0
        thrust = telemetry.get("applied_thrust_n")
        if thrust is None:
            return 1.0
        return float(np.sum(np.maximum(finite_array(thrust, (4,), "applied_thrust_n"), 0.0)) / self.weight_n)

    def measure(self, truth: StepTruth, telemetry: Mapping) -> dict:
        if self.mounting.lever_arm_b is None:
            raise RuntimeError("bind the IMU to the vehicle mass properties before sampling")
        specific_force_s = self.mounting.to_sensor(truth.specific_force_at_b(self.mounting.lever_arm_b))
        angular_velocity_s = self.mounting.to_sensor(truth.angular_velocity_b)
        vibration_accelerometer = vibration_gyroscope = None
        if self.vibration_enabled:
            scale = self._vibration_scale(telemetry)
            vibration_accelerometer = scale * self.vibration_accelerometer
            vibration_gyroscope = scale * self.vibration_gyroscope
        accelerometer = self.accelerometer.measure(specific_force_s, extra_noise_std=vibration_accelerometer)
        gyroscope = self.gyroscope.measure(
            angular_velocity_s, specific_force_s=specific_force_s, extra_noise_std=vibration_gyroscope
        )
        return {
            "time_s": float(truth.after.time_s),
            "accelerometer_m_s2": accelerometer,
            "gyroscope_rad_s": gyroscope,
            "true_specific_force_s_m_s2": specific_force_s,
            "true_angular_velocity_s_rad_s": angular_velocity_s,
        }

    def describe(self) -> dict:
        return {
            "rate_hz": self.rate_hz,
            "mounting": self.mounting.describe(),
            "vibration": {
                "enabled": self.vibration_enabled,
                "accelerometer_std_m_s2": self.vibration_accelerometer.tolist(),
                "gyroscope_std_rad_s": self.vibration_gyroscope.tolist(),
                "scale_with_thrust": self.vibration_scales_with_thrust,
            },
            "accelerometer_white_noise_std_m_s2": self.accelerometer.white_std.tolist(),
            "gyroscope_white_noise_std_rad_s": self.gyroscope.white_std.tolist(),
        }
