"""World-frame wind fields; no ARL-specific atmosphere is assumed.

The built-in field is explicitly parameterized. A gradient is a local affine
approximation, gusts are smooth finite pulses, and optional turbulence is an
Ornstein--Uhlenbeck process, not a CFD model. Stateful fields are evaluated once
per physics step by EffectsPipeline; the same sampled field is used at every
aerodynamic point during that step.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np
from .math import finite_scalar, numeric_array as finite_array, positive_dt


class WindField(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def advance(self, time_s: float, dt_s: float) -> None: ...
    def velocity_w(self, position_w: np.ndarray, time_s: float) -> np.ndarray: ...


@dataclass(frozen=True)
class Gust:
    start_time_s: float
    duration_s: float
    delta_velocity_w_m_s: np.ndarray

    def __post_init__(self):
        finite_scalar(self.start_time_s, "gust start_time_s")
        if self.start_time_s < 0:
            raise ValueError("gust start_time_s must be finite and nonnegative")
        positive_dt(self.duration_s)
        object.__setattr__(self, "delta_velocity_w_m_s", finite_array(
            self.delta_velocity_w_m_s, (3,), "gust delta_velocity_w_m_s"))

    def velocity_w(self, time_s: float) -> np.ndarray:
        phase = (time_s - self.start_time_s) / self.duration_s
        if phase <= 0 or phase >= 1:
            return np.zeros(3)
        return 0.5 * (1 - np.cos(2 * np.pi * phase)) * self.delta_velocity_w_m_s


class UniformGustWind:
    """Affine mean wind + half-cosine gust pulses + optional correlated noise.

    OU noise starts at zero deviation on reset, with explicit stationary standard
    deviation and time constant per world axis. The first advance samples that
    process at the first step; this is a specified initialization, not a draw
    from its stationary distribution. No turbulence is generated when disabled.
    """
    def __init__(self, velocity_w_m_s, *, gusts: Sequence[Gust] = (),
                 spatial_gradient_per_s=None, reference_position_w_m=None,
                 turbulence_time_constant_s=None, turbulence_std_w_m_s=None):
        self.mean_w = finite_array(velocity_w_m_s, (3,), "velocity_w_m_s")
        if (spatial_gradient_per_s is None) != (reference_position_w_m is None):
            raise ValueError("wind gradient and reference position must be supplied together")
        self.gradient = None if spatial_gradient_per_s is None else finite_array(
            spatial_gradient_per_s, (3, 3), "spatial_gradient_per_s")
        self.reference_w = None if reference_position_w_m is None else finite_array(
            reference_position_w_m, (3,), "reference_position_w_m")
        self.gusts = tuple(gusts)
        if not all(isinstance(gust, Gust) for gust in self.gusts):
            raise TypeError("gusts must contain Gust instances")
        if (turbulence_time_constant_s is None) != (turbulence_std_w_m_s is None):
            raise ValueError("OU time constants and standard deviations must be supplied together")
        self.tau_s = None
        self.std_w = None
        if turbulence_time_constant_s is not None:
            self.tau_s = finite_array(turbulence_time_constant_s, (3,), "turbulence time_constant_s")
            self.std_w = finite_array(turbulence_std_w_m_s, (3,), "turbulence stationary_std_w_m_s")
            if np.any(self.tau_s <= 0) or np.any(self.std_w < 0):
                raise ValueError("OU time constants must be positive and deviations nonnegative")
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._noise_w = np.zeros(3)
        self._last_time_s = None

    def advance(self, time_s: float, dt_s: float) -> None:
        dt_s = positive_dt(dt_s)
        finite_scalar(time_s, "wind time_s")
        if time_s < 0:
            raise ValueError("wind time_s must be finite and nonnegative")
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("wind must advance once per increasing physics time; reset before replay")
        if self.tau_s is not None:
            decay = np.exp(-dt_s / self.tau_s)
            sigma = self.std_w * np.sqrt(-np.expm1(-2 * dt_s / self.tau_s))
            self._noise_w = decay * self._noise_w + sigma * self._rng.normal(size=3)
        self._last_time_s = float(time_s)

    def velocity_w(self, position_w: np.ndarray, time_s: float) -> np.ndarray:
        position = finite_array(position_w, (3,), "wind position_w")
        finite_scalar(time_s, "wind time_s")
        if time_s < 0:
            raise ValueError("wind time_s must be finite and nonnegative")
        velocity = self.mean_w + self._noise_w
        if self.gradient is not None:
            velocity = velocity + self.gradient @ (position - self.reference_w)
        for gust in self.gusts:
            velocity = velocity + gust.velocity_w(time_s)
        return velocity
