"""Discrete noise processes shared by the sensor models.

The scalings follow the Algorithms section of the MATLAB ``imuSensor`` documentation
with ``s = 2`` for a ``double-sided`` and ``s = 1`` for a ``single-sided`` noise
density (the ``NoiseType`` of accelparams/gyroparams/magparams):

- white noise, per sample:      β₂ = w · √(f_s / s) · NoiseDensity
- random walk, per sample:      β₃(k) = β₃(k−1) + w(k) · RandomWalk / √(f_s / s)

A datasheet density (RMS = density · √bandwidth with the Nyquist bandwidth f_s/2)
corresponds to the MATLAB default ``double-sided``: at 200 Hz one sample has
σ = 10 × density. Bias instability is a first-order Gauss--Markov process with
the handbook value as stationary standard deviation and an explicit correlation
time (exact discretization, as ``effects/stochastic.py``), instead of MATLAB's
1/f filter; its statistics, not its samples, are what the model reproduces.
``gpsSensor`` and ``altimeterSensor`` use the decay recursion x(k) = a·x(k−1) + σ·w(k):
a = 0 is white noise, a = 1 a random walk, the stationary deviation is σ/√(1−a²)
and the time constant −Δt/ln a.
"""

from __future__ import annotations

import numpy as np

from isaac_drone.core.validation import finite_array, finite_scalar

NOISE_TYPES = ("double-sided", "single-sided")


def noise_type_scale(noise_type: str) -> float:
    """``s``: 2 for a double-sided density, 1 for a single-sided one."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}; got {noise_type!r}")
    return 2.0 if noise_type == "double-sided" else 1.0


def white_noise_std(noise_density, rate_hz: float, noise_type: str, size: int = 3) -> np.ndarray:
    """Per-sample standard deviation ``NoiseDensity · √(f_s / s)`` of a density given per √Hz."""
    density = _nonnegative(noise_density, size, "noise_density")
    return density * np.sqrt(_rate(rate_hz) / noise_type_scale(noise_type))


def random_walk_step_std(random_walk, rate_hz: float, noise_type: str, size: int = 3) -> np.ndarray:
    """Per-sample increment ``RandomWalk / √(f_s / s)`` of a random-walk coefficient given in unit·√Hz."""
    coefficient = _nonnegative(random_walk, size, "random_walk")
    return coefficient / np.sqrt(_rate(rate_hz) / noise_type_scale(noise_type))


def _rate(rate_hz) -> float:
    rate = finite_scalar(rate_hz, "rate_hz")
    if rate <= 0:
        raise ValueError("rate_hz must be positive")
    return rate


def _nonnegative(value, size, name) -> np.ndarray:
    array = finite_array(np.broadcast_to(np.asarray(value, dtype=float), (size,)), (size,), name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


class GaussMarkov:
    """First-order Gauss--Markov process with stationary deviation ``std`` and correlation time ``tau``.

    ``reset`` draws the state from the stationary distribution; ``step`` applies
    the exact discretization x ← e^{−Δt/τ} x + std·√(1 − e^{−2Δt/τ}) w.
    """

    def __init__(self, std, time_constant_s: float, size: int = 3):
        self.std = _nonnegative(std, size, "std")
        self.tau = finite_scalar(time_constant_s, "time_constant_s")
        if self.tau <= 0:
            raise ValueError("time_constant_s must be positive")
        self.state = np.zeros(size)

    def reset(self, rng: np.random.Generator) -> None:
        self.state = self.std * rng.standard_normal(self.std.shape)

    def step(self, dt_s: float, rng: np.random.Generator) -> np.ndarray:
        dt = finite_scalar(dt_s, "dt_s")
        if dt <= 0:
            raise ValueError("dt_s must be positive")
        decay = np.exp(-dt / self.tau)
        self.state = decay * self.state + self.std * np.sqrt(-np.expm1(-2.0 * dt / self.tau)) * rng.standard_normal(
            self.std.shape
        )
        return self.state.copy()


class DecayingNoise:
    """MATLAB ``gpsSensor``/``altimeterSensor`` recursion x(k) = a·x(k−1) + σ·w(k), starting from zero."""

    def __init__(self, std, decay_factor: float, size: int = 3):
        self.std = _nonnegative(std, size, "std")
        self.decay = finite_scalar(decay_factor, "decay_factor")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay_factor must lie in [0, 1]")
        self.state = np.zeros(size)

    def reset(self) -> None:
        self.state = np.zeros(self.std.shape)

    def step(self, rng: np.random.Generator) -> np.ndarray:
        self.state = self.decay * self.state + self.std * rng.standard_normal(self.std.shape)
        return self.state.copy()

    @property
    def stationary_std(self) -> np.ndarray:
        """σ/√(1−a²); infinite for a random walk (a = 1)."""
        if self.decay >= 1.0:
            return np.where(self.std > 0, np.inf, 0.0)
        return self.std / np.sqrt(1.0 - self.decay**2)


class RandomWalk:
    """Integrated white noise x(k) = x(k−1) + q·w(k), starting from zero."""

    def __init__(self, step_std, size: int = 3):
        self.step_std = _nonnegative(step_std, size, "step_std")
        self.state = np.zeros(size)

    def reset(self) -> None:
        self.state = np.zeros(self.step_std.shape)

    def step(self, rng: np.random.Generator) -> np.ndarray:
        self.state = self.state + self.step_std * rng.standard_normal(self.step_std.shape)
        return self.state.copy()
