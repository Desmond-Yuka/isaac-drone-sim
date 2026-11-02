"""The MATLAB ``imuSensor`` error chain for one three-axis sensor (accelerometer, gyroscope, magnetometer).

Parameters mirror ``accelparams`` / ``gyroparams`` / ``magparams``; the processing
order is the one of the imuSensor Algorithms section:

1. ideal reading in sensor axes ``a``;
2. bulk model ``b = (M/100) a + ConstantBias`` with the misalignment matrix ``M`` in percent;
3. bias instability ``β₁`` (Gauss--Markov, see ``noise.py``);
4. white noise ``β₂ = w √(f_s/s) NoiseDensity``;
5. random walk ``β₃(k) = β₃(k−1) + w RandomWalk/√(f_s/s)``;
6. temperature drift ``Δe = (T − 25) TemperatureBias``;
7. gyroscope only: acceleration sensitivity ``Δa = AccelerationBias · f`` with the
   specific force in sensor axes;
8. scale ``d = (b + β₁ + β₂ + β₃ + Δe + Δa) · (1 + (T − 25)/100 · TemperatureScaleFactor)``;
9. saturation at ±MeasurementRange, then quantization ``Resolution · round(d / Resolution)``.

Beyond MATLAB, an optional extra white noise (IMU vibration) is added with the
other noise terms before saturation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from isaac_drone.core.validation import finite_array, finite_scalar

from .noise import NOISE_TYPES, GaussMarkov, RandomWalk, random_walk_step_std, white_noise_std

NOMINAL_TEMPERATURE_C = 25.0
TRIAXIAL_FIELDS = (
    "measurement_range",
    "resolution",
    "constant_bias",
    "axes_misalignment",
    "noise_density",
    "bias_instability",
    "bias_instability_time_constant_s",
    "random_walk",
    "noise_type",
    "temperature_bias",
    "temperature_scale_factor",
)
GYRO_FIELDS = (*TRIAXIAL_FIELDS, "acceleration_bias")


def misalignment_matrix(value) -> np.ndarray:
    """``AxesMisalignment`` in percent as a 3x3 matrix.

    A scalar ``s`` sets every off-diagonal element to ``s`` (diagonal 100); a
    3-vector ``[a, b, c]`` sets m21 = m31 = a, m12 = m32 = b, m13 = m23 = c; a
    3x3 matrix is used as given. ``diag([100, 100, 100])`` is no misalignment.
    """
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        matrix = np.full((3, 3), float(array))
        np.fill_diagonal(matrix, 100.0)
    elif array.shape == (3,):
        a, b, c = array
        matrix = np.array([[100.0, b, c], [a, 100.0, c], [a, b, 100.0]])
    elif array.shape == (3, 3):
        matrix = array.copy()
    else:
        raise ValueError("axes_misalignment must be a scalar, a 3-vector or a 3x3 matrix (percent)")
    return finite_array(matrix, (3, 3), "axes_misalignment")


def _vector3(value, name, *, nonnegative=False) -> np.ndarray:
    array = finite_array(np.broadcast_to(np.asarray(value, dtype=float), (3,)), (3,), name)
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


@dataclass(frozen=True)
class TriaxialParams:
    """Validated accelparams/gyroparams/magparams values; ``acceleration_bias`` is zero except for a gyroscope."""

    measurement_range: float
    resolution: float
    constant_bias: np.ndarray
    misalignment: np.ndarray
    noise_density: np.ndarray
    bias_instability: np.ndarray
    bias_instability_time_constant_s: float
    random_walk: np.ndarray
    noise_type: str
    temperature_bias: np.ndarray
    temperature_scale_factor: np.ndarray
    acceleration_bias: np.ndarray

    @classmethod
    def from_config(cls, section: Mapping, path: str, *, gyroscope: bool = False) -> TriaxialParams:
        if not isinstance(section, Mapping):
            raise ValueError(f"{path} must be a mapping")
        fields = GYRO_FIELDS if gyroscope else TRIAXIAL_FIELDS
        unknown, missing = set(section) - set(fields), set(fields) - set(section)
        if unknown or missing:
            raise ValueError(f"{path}: unknown={sorted(unknown)}, missing={sorted(missing)}")
        measurement_range = section["measurement_range"]
        measurement_range = np.inf if measurement_range is None else finite_scalar(measurement_range, "range")
        if measurement_range <= 0:
            raise ValueError(f"{path}.measurement_range must be positive or null (unlimited)")
        resolution = finite_scalar(section["resolution"], f"{path}.resolution")
        if resolution < 0:
            raise ValueError(f"{path}.resolution must be nonnegative (0 disables quantization)")
        tau = finite_scalar(section["bias_instability_time_constant_s"], f"{path}.bias_instability_time_constant_s")
        if tau <= 0:
            raise ValueError(f"{path}.bias_instability_time_constant_s must be positive")
        if section["noise_type"] not in NOISE_TYPES:
            raise ValueError(f"{path}.noise_type must be one of {NOISE_TYPES}")
        scale = _vector3(section["temperature_scale_factor"], f"{path}.temperature_scale_factor")
        if np.any(scale < 0) or np.any(scale > 100):
            raise ValueError(f"{path}.temperature_scale_factor must lie in [0, 100] %/°C")
        return cls(
            float(measurement_range),
            float(resolution),
            _vector3(section["constant_bias"], f"{path}.constant_bias"),
            misalignment_matrix(section["axes_misalignment"]),
            _vector3(section["noise_density"], f"{path}.noise_density", nonnegative=True),
            _vector3(section["bias_instability"], f"{path}.bias_instability", nonnegative=True),
            float(tau),
            _vector3(section["random_walk"], f"{path}.random_walk", nonnegative=True),
            str(section["noise_type"]),
            _vector3(section["temperature_bias"], f"{path}.temperature_bias"),
            scale,
            _vector3(section["acceleration_bias"], f"{path}.acceleration_bias") if gyroscope else np.zeros(3),
        )


class TriaxialModel:
    """Apply the error chain to ideal sensor-axis readings at a fixed sample rate and temperature."""

    def __init__(self, params: TriaxialParams, rate_hz: float, temperature_c: float = NOMINAL_TEMPERATURE_C):
        self.params = params
        self.rate_hz = finite_scalar(rate_hz, "rate_hz")
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.dt_s = 1.0 / self.rate_hz
        delta_t = finite_scalar(temperature_c, "temperature_c") - NOMINAL_TEMPERATURE_C
        self.white_std = white_noise_std(params.noise_density, self.rate_hz, params.noise_type)
        self.bias_instability = GaussMarkov(params.bias_instability, params.bias_instability_time_constant_s)
        self.random_walk = RandomWalk(random_walk_step_std(params.random_walk, self.rate_hz, params.noise_type))
        self.temperature_drift = delta_t * params.temperature_bias
        self.scale_factor = 1.0 + delta_t / 100.0 * params.temperature_scale_factor
        self.gain = params.misalignment / 100.0
        self._rng = None

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.bias_instability.reset(rng)
        self.random_walk.reset()

    def measure(self, true_s, *, specific_force_s=None, extra_noise_std=None) -> np.ndarray:
        """One reading from the ideal sensor-axis value ``true_s``.

        ``specific_force_s`` feeds a gyroscope's acceleration sensitivity;
        ``extra_noise_std`` adds a further white noise term (vibration).
        """
        if self._rng is None:
            raise RuntimeError("reset the sensor model with a random generator first")
        ideal = finite_array(true_s, (3,), "true_s")
        p = self.params
        value = self.gain @ ideal + p.constant_bias
        value = value + self.bias_instability.step(self.dt_s, self._rng)
        value = value + self.white_std * self._rng.standard_normal(3)
        value = value + self.random_walk.step(self._rng)
        if extra_noise_std is not None:
            value = value + _vector3(extra_noise_std, "extra_noise_std", nonnegative=True) * self._rng.standard_normal(
                3
            )
        value = value + self.temperature_drift
        if np.any(p.acceleration_bias):
            if specific_force_s is None:
                raise ValueError("a gyroscope with acceleration_bias needs the specific force in sensor axes")
            value = value + p.acceleration_bias * finite_array(specific_force_s, (3,), "specific_force_s")
        value = value * self.scale_factor
        value = np.clip(value, -p.measurement_range, p.measurement_range)
        if p.resolution > 0:
            value = p.resolution * np.round(value / p.resolution)
        return value
