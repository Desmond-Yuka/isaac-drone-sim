"""Rest-to-rest progress time laws sigma(t) in [0, 1] for one mission phase.

A law returns ``[sigma, sigma', sigma'', sigma''', sigma'''']`` at a time
relative to its phase start. Geometry (takeoff line, helix) is applied by the
trajectory; the law only decides how fast the fixed path is traversed.

* ``HermiteTimeLaw``: the ninth-degree C4 polynomial over a given duration.
* ``CruiseTimeLaw``: accelerate / constant-rate cruise / decelerate. The rate
  ramps follow the same ninth-degree polynomial, so the rate is C4 and the
  position C5, and all four position derivatives vanish at both ends.
"""

from __future__ import annotations

import math

import numpy as np


def _progress_derivatives(u: float, duration_s: float) -> np.ndarray:
    """Return S and d^kS/dt^k (k=1..4), exactly zero outside transition.

    Symmetry S(1-u)=1-S(u) avoids cancellation in the ninth-degree position
    polynomial near the endpoint. Factored derivative forms retain their
    endpoint zeros without subtracting large polynomial terms.
    """
    if u <= 0:
        return np.zeros(5)
    if u >= 1:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    q = min(u, 1.0 - u)
    position = q**5 * (126.0 + q * (-420.0 + q * (540.0 + q * (-315.0 + 70.0 * q))))
    if u > 0.5:
        position = 1.0 - position
    v = 1.0 - u
    first = 630.0 * u**4 * v**4
    second = 2520.0 * u**3 * v**3 * (1.0 - 2.0 * u)
    third = 2520.0 * u**2 * v**2 * (3.0 - 14.0 * u + 14.0 * u * u)
    fourth = 15120.0 * u * v * (1.0 - 2.0 * u) * (1.0 - 7.0 * u + 7.0 * u * u)
    inverse = 1.0 / duration_s
    return np.array([position, first * inverse, second * inverse**2, third * inverse**3, fourth * inverse**4])


def _progress_integral(u: float) -> float:
    """I(u) = integral of S over [0, u]; I(1) = 1/2 and I(u) = u - 1/2 + I(1-u)."""
    if u <= 0:
        return 0.0
    if u >= 1:
        return u - 0.5
    q = min(u, 1.0 - u)
    value = q**6 * (21.0 + q * (-60.0 + q * (67.5 + q * (-35.0 + 7.0 * q))))
    return u - 0.5 + value if u > 0.5 else value


def _positive(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class HermiteTimeLaw:
    """sigma(t) = S(t/T): the fixed-duration ninth-degree rest-to-rest law."""

    kind = "hermite"

    def __init__(self, duration_s: float):
        self.duration_s = _positive(duration_s, "phase duration")

    @property
    def peak_rate(self) -> float:
        """Exact maximum of sigma' [1/s], reached at the midpoint."""
        return 315.0 / (128.0 * self.duration_s)

    @property
    def segments(self) -> tuple[tuple[str, float, float], ...]:
        return (("transition", 0.0, self.duration_s),)

    def derivatives(self, time_s: float) -> np.ndarray:
        return _progress_derivatives(time_s / self.duration_s, self.duration_s)

    def describe(self) -> dict:
        return {"kind": self.kind, "duration_s": self.duration_s, "peak_rate_per_s": self.peak_rate}


class CruiseTimeLaw:
    """Accelerate over T_a, cruise at rate r, decelerate over T_d.

    sigma' = r S(t/T_a) while accelerating, r while cruising and
    r (1 - S(t'/T_d)) while decelerating, where S is the ninth-degree progress
    polynomial. The cruise time follows from total progress one:
    r (T_a/2 + T_c + T_d/2) = 1, so T_c >= 0 is required.
    """

    kind = "accelerate_cruise_decelerate"

    def __init__(self, cruise_rate: float, accelerate_s: float, decelerate_s: float):
        self.cruise_rate = _positive(cruise_rate, "cruise rate")
        self.accelerate_s = _positive(accelerate_s, "acceleration duration")
        self.decelerate_s = _positive(decelerate_s, "deceleration duration")
        cruise = 1.0 / self.cruise_rate - 0.5 * (self.accelerate_s + self.decelerate_s)
        # Rounding of a planned exact fit may leave a tiny negative cruise time.
        if cruise < -1e-9 * (1.0 / self.cruise_rate):
            raise ValueError("Ramps are too long to reach the cruise rate within total progress one")
        self.cruise_s = max(0.0, cruise)
        self.duration_s = self.accelerate_s + self.cruise_s + self.decelerate_s
        if not math.isfinite(self.duration_s):
            raise ValueError("Cruise time law duration is not finite")

    @staticmethod
    def cruise_duration(cruise_rate: float, accelerate_s: float, decelerate_s: float) -> float:
        """Cruise time implied by the three parameters; negative means unreachable."""
        return 1.0 / cruise_rate - 0.5 * (accelerate_s + decelerate_s)

    @property
    def peak_rate(self) -> float:
        return self.cruise_rate

    @property
    def segments(self) -> tuple[tuple[str, float, float], ...]:
        start_decelerate = self.accelerate_s + self.cruise_s
        result = [("accelerate", 0.0, self.accelerate_s)]
        if self.cruise_s > 0:
            result.append(("cruise", self.accelerate_s, start_decelerate))
        result.append(("decelerate", start_decelerate, self.duration_s))
        return tuple(result)

    @staticmethod
    def accelerate(cruise_rate: float, accelerate_s: float, time_s: float) -> np.ndarray:
        """Ramp from rest; depends only on (r, T_a), not on the rest of the law."""
        u = time_s / accelerate_s
        ramp = _progress_derivatives(u, accelerate_s)
        return np.r_[cruise_rate * accelerate_s * _progress_integral(min(u, 1.0)), cruise_rate * ramp[:4]]

    @staticmethod
    def decelerate(cruise_rate: float, decelerate_s: float, time_to_end_s: float) -> np.ndarray:
        """Ramp to rest at progress one, indexed by the time remaining to the end."""
        u = 1.0 - time_to_end_s / decelerate_s
        ramp = _progress_derivatives(u, decelerate_s)
        remaining = cruise_rate * decelerate_s * _progress_integral(max(1.0 - u, 0.0))
        return np.r_[1.0 - remaining, cruise_rate * (1.0 - ramp[0]), -cruise_rate * ramp[1:4]]

    def derivatives(self, time_s: float) -> np.ndarray:
        if time_s <= 0:
            return np.zeros(5)
        if time_s >= self.duration_s:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        if time_s < self.accelerate_s:
            return self.accelerate(self.cruise_rate, self.accelerate_s, time_s)
        if time_s <= self.accelerate_s + self.cruise_s:
            return np.array(
                [
                    self.cruise_rate * (0.5 * self.accelerate_s + time_s - self.accelerate_s),
                    self.cruise_rate,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        return self.decelerate(self.cruise_rate, self.decelerate_s, self.duration_s - time_s)

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "duration_s": self.duration_s,
            "peak_rate_per_s": self.cruise_rate,
            "accelerate_s": self.accelerate_s,
            "cruise_s": self.cruise_s,
            "decelerate_s": self.decelerate_s,
        }
