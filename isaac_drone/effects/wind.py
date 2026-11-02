"""World-frame wind fields; no ARL-specific atmosphere is assumed.

The built-in field is explicitly parameterized. A gradient is a local affine
approximation, gusts are finite pulses (smooth 1-cosine or a sudden step), and
optional turbulence is an Ornstein--Uhlenbeck process, not a CFD model: either
one independent process per world axis, or one scalar process along a fixed
world direction (a single axis is the direction [1, 0, 0], [0, 1, 0] or
[0, 0, 1]). The OU state is drawn from its stationary distribution at reset and
advanced with the exact discretization every physics step, so turbulence is at
full strength from the start. An optional [start, end) window gates the whole
field: still air outside it, the configured wind at once inside it. Stateful
fields are evaluated once per physics step by EffectsPipeline; the same sampled
field is used at every aerodynamic point during that step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from isaac_drone.effects._math import finite_scalar, positive_dt
from isaac_drone.effects._math import numeric_array as finite_array

GUST_SHAPES = ("one_minus_cosine", "step")


class WindField(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def advance(self, time_s: float, dt_s: float) -> None: ...
    def velocity_w(self, position_w: np.ndarray, time_s: float) -> np.ndarray: ...


@dataclass(frozen=True)
class Gust:
    """A gust pulse added to the wind in [start, start + duration).

    ``one_minus_cosine`` rises smoothly from zero and returns to zero. ``step``
    appears at full strength at the start and disappears at the end; a step with
    ``duration_s`` None lasts until the end of the run.
    """

    start_time_s: float
    duration_s: float | None
    delta_velocity_w_m_s: np.ndarray
    shape: str = "one_minus_cosine"

    def __post_init__(self):
        finite_scalar(self.start_time_s, "gust start_time_s")
        if self.start_time_s < 0:
            raise ValueError("gust start_time_s must be finite and nonnegative")
        if self.shape not in GUST_SHAPES:
            raise ValueError(f"gust shape must be one of {', '.join(GUST_SHAPES)}")
        if self.duration_s is None:
            if self.shape != "step":
                raise ValueError("only a step gust may last to the end of the run (duration_s None)")
        else:
            positive_dt(self.duration_s)
        object.__setattr__(
            self, "delta_velocity_w_m_s", finite_array(self.delta_velocity_w_m_s, (3,), "gust delta_velocity_w_m_s")
        )

    def velocity_w(self, time_s: float) -> np.ndarray:
        elapsed = time_s - self.start_time_s
        if elapsed < 0 or (self.duration_s is not None and elapsed >= self.duration_s):
            return np.zeros(3)
        if self.shape == "step":
            return self.delta_velocity_w_m_s.copy()
        phase = elapsed / self.duration_s
        return 0.5 * (1 - np.cos(2 * np.pi * phase)) * self.delta_velocity_w_m_s


class UniformGustWind:
    """Affine mean wind + gust pulses + optional correlated noise, inside a time window.

    Turbulence is either one OU process per world axis (time constants and
    stationary standard deviations given per axis; a zero deviation leaves that
    axis free of noise) or, with ``turbulence_direction_w``, one scalar OU process
    along that unit direction (scalar time constant and deviation). Outside
    [start_time_s, end_time_s) the field is still air; the OU state keeps
    evolving there, so the turbulence is stationary when the window opens.
    """

    def __init__(
        self,
        velocity_w_m_s,
        *,
        gusts: Sequence[Gust] = (),
        spatial_gradient_per_s=None,
        reference_position_w_m=None,
        turbulence_time_constant_s=None,
        turbulence_std_w_m_s=None,
        turbulence_direction_w=None,
        start_time_s=0.0,
        end_time_s=None,
    ):
        self.mean_w = finite_array(velocity_w_m_s, (3,), "velocity_w_m_s")
        if (spatial_gradient_per_s is None) != (reference_position_w_m is None):
            raise ValueError("wind gradient and reference position must be supplied together")
        self.gradient = (
            None
            if spatial_gradient_per_s is None
            else finite_array(spatial_gradient_per_s, (3, 3), "spatial_gradient_per_s")
        )
        self.reference_w = (
            None
            if reference_position_w_m is None
            else finite_array(reference_position_w_m, (3,), "reference_position_w_m")
        )
        self.gusts = tuple(gusts)
        if not all(isinstance(gust, Gust) for gust in self.gusts):
            raise TypeError("gusts must contain Gust instances")
        if (turbulence_time_constant_s is None) != (turbulence_std_w_m_s is None):
            raise ValueError("OU time constants and standard deviations must be supplied together")
        self.tau_s = None
        self.std_w = None
        self.direction_w = None
        if turbulence_time_constant_s is not None:
            if turbulence_direction_w is None:
                self.tau_s = finite_array(turbulence_time_constant_s, (3,), "turbulence time_constant_s")
                self.std_w = finite_array(turbulence_std_w_m_s, (3,), "turbulence stationary_std_w_m_s")
            else:
                direction = finite_array(turbulence_direction_w, (3,), "turbulence direction_w")
                length = float(np.linalg.norm(direction))
                if length == 0:
                    raise ValueError("turbulence direction_w must be a nonzero vector")
                self.direction_w = direction / length
                self.tau_s = np.array([finite_scalar(turbulence_time_constant_s, "turbulence time_constant_s")])
                self.std_w = np.array([finite_scalar(turbulence_std_w_m_s, "turbulence stationary_std_w_m_s")])
            if np.any(self.tau_s <= 0) or np.any(self.std_w < 0):
                raise ValueError("OU time constants must be positive and deviations nonnegative")
        elif turbulence_direction_w is not None:
            raise ValueError("turbulence direction_w requires an OU time constant and standard deviation")
        self.start_s = finite_scalar(start_time_s, "wind start_time_s")
        if self.start_s < 0:
            raise ValueError("wind start_time_s must be nonnegative")
        self.end_s = None if end_time_s is None else finite_scalar(end_time_s, "wind end_time_s")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("wind end_time_s must exceed start_time_s, or be None")
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        # Stationary initialization: full-strength turbulence from the first step, no build-up from zero.
        self._noise = None if self.std_w is None else self.std_w * self._rng.normal(size=self.std_w.shape)
        self._last_time_s = None

    def advance(self, time_s: float, dt_s: float) -> None:
        dt_s = positive_dt(dt_s)
        finite_scalar(time_s, "wind time_s")
        if time_s < 0:
            raise ValueError("wind time_s must be finite and nonnegative")
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("wind must advance once per increasing physics time; reset before replay")
        if self._noise is not None:
            decay = np.exp(-dt_s / self.tau_s)
            sigma = self.std_w * np.sqrt(-np.expm1(-2 * dt_s / self.tau_s))
            self._noise = decay * self._noise + sigma * self._rng.normal(size=self._noise.shape)
        self._last_time_s = float(time_s)

    def active(self, time_s: float) -> bool:
        """True inside the [start_time_s, end_time_s) window."""
        return time_s >= self.start_s and (self.end_s is None or time_s < self.end_s)

    def turbulence_w(self) -> np.ndarray:
        """Current OU turbulence velocity, ignoring the window."""
        if self._noise is None:
            return np.zeros(3)
        if self.direction_w is not None:
            return self._noise[0] * self.direction_w
        return self._noise.copy()

    def components_w(self, position_w: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """(mean + gradient + gusts, OU turbulence) at a world point; both zero outside the window."""
        position = finite_array(position_w, (3,), "wind position_w")
        finite_scalar(time_s, "wind time_s")
        if time_s < 0:
            raise ValueError("wind time_s must be finite and nonnegative")
        if not self.active(time_s):
            return np.zeros(3), np.zeros(3)
        deterministic = self.mean_w.copy()
        if self.gradient is not None:
            deterministic = deterministic + self.gradient @ (position - self.reference_w)
        for gust in self.gusts:
            deterministic = deterministic + gust.velocity_w(time_s)
        return deterministic, self.turbulence_w()

    def velocity_w(self, position_w: np.ndarray, time_s: float) -> np.ndarray:
        deterministic, turbulence = self.components_w(position_w, time_s)
        return deterministic + turbulence
