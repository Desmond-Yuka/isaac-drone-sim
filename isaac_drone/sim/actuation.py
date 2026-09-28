"""Rotor-speed actuation using Isaac Lab's native first-order motor integrators.

This is a coefficient-based propulsion model, not CFD or a calibrated electric
motor model. Positive running commands retain native per-motor minimum thrust;
zero commands explicitly request spin-down to stopped rotors.
"""
from __future__ import annotations

import numpy as np

from isaac_drone.core.validation import finite_array


def numpy_value(value):
    tensor = getattr(value, "torch", None)
    if tensor is not None:
        value = tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def native_value(value, like):
    if isinstance(like, np.ndarray):
        return np.asarray(value, dtype=like.dtype)
    import torch
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def thrust_to_motor_speeds(thrust_n, sampled_kf, thrust_min_n, thrust_max_n):
    """Convert nonnegative rotor thrust requests to limited RPS magnitudes.

    Zero stays zero (motor off); positive requests retain each motor's original
    operating lower bound. This conversion does not advance motor dynamics.
    """
    kf = np.asarray(sampled_kf, dtype=np.float64)
    if kf.ndim != 1:
        raise ValueError("sampled_kf must be a one-dimensional motor array")
    shape = kf.shape
    kf = finite_array(kf, shape, "sampled_kf")
    thrust = finite_array(thrust_n, shape, "rotor thrust request")
    lower = finite_array(thrust_min_n, shape, "minimum running thrust")
    upper = finite_array(thrust_max_n, shape, "maximum thrust")
    if np.any(kf <= 0) or np.any(lower < 0) or np.any(upper <= 0) or np.any(lower > upper):
        raise ValueError("Invalid positive motor coefficient or thrust limits")
    if np.any(thrust < 0):
        raise ValueError("Rotor thrust requests must be nonnegative")
    limited = np.where(thrust == 0, 0.0, np.clip(thrust, lower, upper))
    return finite_array(np.sqrt(limited/kf), shape, "target motor speed rps")


class NativeRpsActuator:
    """Store RPS directly and reuse the native actuator's integration functions."""

    def __init__(self, native, initial_rps):
        self.native = native
        self.rps = native_value(initial_rps, native.curr_thrust)
        self.raw_target_rps = self.rps * 1.0
        self.target_rps = self.rps * 1.0
        self.reset(initial_rps)

    def parameters(self):
        shape = numpy_value(self.rps).shape
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError("Native RPS actuator requires one environment")
        kf = finite_array(numpy_value(self.native.thrust_const), shape, "sampled thrust_const")
        bounds = finite_array(numpy_value(self.native.cfg.thrust_range), (2,), "native thrust_range")
        if np.any(kf <= 0) or bounds[0] < 0 or bounds[1] <= 0 or bounds[0] > bounds[1]:
            raise ValueError("Native thrust coefficient/limits are invalid")
        return kf, np.sqrt(bounds[0]/kf), np.sqrt(bounds[1]/kf)

    def _synchronize_thrust(self):
        # During spin-up/spin-down, thrust can lie below the minimum running
        # command. Do not invent a minimum force inconsistent with kf*n**2.
        self.native.curr_thrust[:] = self.native.thrust_const * self.rps**2
        self.native.computed_thrust = self.native.curr_thrust
        self.native.applied_thrust = self.native.curr_thrust * 1.0

    def reset(self, initial_rps):
        kf, _, maximum = self.parameters()
        initial = finite_array(numpy_value(initial_rps), kf.shape, "initial motor rps")
        if np.any(initial < 0) or np.any(initial > maximum):
            raise ValueError("Initial motor RPS must be within zero and the native maximum")
        self.rps = native_value(initial, self.native.curr_thrust)
        self.raw_target_rps = self.rps * 1.0
        self.target_rps = self.rps * 1.0
        self._synchronize_thrust()

    def step(self, target_rps):
        kf, minimum, maximum = self.parameters()
        target = finite_array(numpy_value(target_rps), kf.shape, "target motor rps")
        current = finite_array(numpy_value(self.rps), kf.shape, "actual motor rps")
        if np.any(target < 0):
            raise ValueError("Motor RPS commands are nonnegative magnitudes")
        limited = np.where(target == 0, 0.0, np.clip(target, minimum, maximum))
        tau_inc = finite_array(numpy_value(self.native.tau_inc_s), kf.shape, "native tau_inc_s")
        tau_dec = finite_array(numpy_value(self.native.tau_dec_s), kf.shape, "native tau_dec_s")
        if np.any(tau_inc <= 0) or np.any(tau_dec <= 0):
            raise ValueError("Motor time constants must be positive")
        dt = float(self.native.cfg.dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Native motor dt must be positive")
        tau = native_value(np.where(limited < current, tau_dec, tau_inc), self.rps)
        self.raw_target_rps = native_value(target, self.rps)
        self.target_rps = native_value(limited, self.rps)
        mixing = self.native.mixing_factor_function(tau)
        error = self.target_rps - self.rps
        if self.native.cfg.integration_scheme == "rk4":
            increment = self.native.rk4_integration(error, mixing)
        elif self.native.cfg.integration_scheme == "euler":
            increment = self.native.motor_model_rate(error, mixing) * dt
        else:
            raise ValueError("Unsupported native motor integration scheme")
        proposed = finite_array(numpy_value(self.rps + increment), kf.shape, "integrated motor rps")
        # Preserve physical nonnegative speed and the original maximum output.
        # Min-running-speed applies to positive targets, never to an off rotor.
        self.rps = native_value(np.clip(proposed, 0.0, maximum), self.rps)
        self._synchronize_thrust()
        return self.rps
