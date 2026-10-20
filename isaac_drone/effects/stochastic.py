"""Mean plus Ornstein--Uhlenbeck random wrench: force in world axes at the CoM, torque in body axes.

This is the d1/d2 (force/torque) disturbance pair of Kidambi's quadrotor model
(IEEE Access 2025 eq. 1; arXiv 2609.12188 eqs. 6a/6b) with a stationary OU
process instead of sinusoids. The OU state is sampled from its stationary
distribution at reset and advanced with the exact discretization.
"""

from __future__ import annotations

import numpy as np

from isaac_drone.core.types import VehicleState, Wrench
from isaac_drone.effects._math import body_to_world, finite_scalar, positive_dt
from isaac_drone.effects._math import numeric_array as finite_array
from isaac_drone.effects.wind import WindField


class OUWrench:
    def __init__(
        self,
        *,
        force_mean_w_n,
        force_std_w_n,
        force_time_constant_s,
        torque_mean_b_nm,
        torque_std_b_nm,
        torque_time_constant_s,
        start_time_s=0.0,
        end_time_s=None,
    ):
        self.force_mean = finite_array(force_mean_w_n, (3,), "force_mean_w_n")
        self.force_std = finite_array(force_std_w_n, (3,), "force_std_w_n")
        self.torque_mean = finite_array(torque_mean_b_nm, (3,), "torque_mean_b_nm")
        self.torque_std = finite_array(torque_std_b_nm, (3,), "torque_std_b_nm")
        self.force_tau = finite_scalar(force_time_constant_s, "force_time_constant_s")
        self.torque_tau = finite_scalar(torque_time_constant_s, "torque_time_constant_s")
        if np.any(self.force_std < 0) or np.any(self.torque_std < 0):
            raise ValueError("OU standard deviations must be nonnegative")
        if self.force_tau <= 0 or self.torque_tau <= 0:
            raise ValueError("OU time constants must be positive")
        self.start = finite_scalar(start_time_s, "start_time_s")
        if self.start < 0:
            raise ValueError("start_time_s must be nonnegative")
        self.end = None if end_time_s is None else finite_scalar(end_time_s, "end_time_s")
        if self.end is not None and self.end <= self.start:
            raise ValueError("end_time_s must exceed start_time_s, or be None")
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._force = self.force_std * self._rng.normal(size=3)
        self._torque = self.torque_std * self._rng.normal(size=3)
        self._last = {}

    def _advance(self, value, std, tau, dt_s):
        decay = np.exp(-dt_s / tau)
        return decay * value + std * np.sqrt(-np.expm1(-2 * dt_s / tau)) * self._rng.normal(size=3)

    def evaluate(self, state: VehicleState, dt_s: float, wind: WindField | None = None) -> Wrench:
        dt_s = positive_dt(dt_s)
        force_w = self.force_mean + self._force
        torque_b = self.torque_mean + self._torque
        self._force = self._advance(self._force, self.force_std, self.force_tau, dt_s)
        self._torque = self._advance(self._torque, self.torque_std, self.torque_tau, dt_s)
        if state.time_s < self.start or (self.end is not None and state.time_s >= self.end):
            self._last = {key: np.zeros(3) for key in _DIAGNOSTICS}
            return Wrench.zero()
        self._last = dict(zip(_DIAGNOSTICS, (force_w, self.force_mean.copy(), torque_b, self.torque_mean.copy())))
        return Wrench(force_b=body_to_world(state.quaternion_wxyz).T @ force_w, torque_b=torque_b)

    def diagnostics(self) -> dict:
        """Last step's applied force (world) and torque (body) with their means; zero outside the window."""
        return {key: value.copy() for key, value in self._last.items()}


_DIAGNOSTICS = ("force_ou_w_n", "force_ou_mean_w_n", "torque_ou_b_nm", "torque_ou_mean_b_nm")
