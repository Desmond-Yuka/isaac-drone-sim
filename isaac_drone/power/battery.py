"""Calibrated, lumped isothermal equivalent-circuit battery model.

Positive current discharges the pack, negative current charges it. The OCV
lookup describes the whole pack (not one cell); capacity is pack Ah, and every
resistance/capacitance is also pack-level. Current is held constant over each
integration interval. RC dynamics and terminal energy integration are exact for
that input and piecewise-linear OCV. This is a declared numerical input contract,
not an assertion that a real ESC draws constant current.

This model does not identify chemistry, cells in series/parallel, hysteresis,
ageing, self-discharge, thermal dynamics, motor efficiency, ESC load or a BMS.
Those need measured extensions. Its coefficients are valid only over the stated
calibration temperature range. Missing temperature remains unknown; no ambient
or room temperature is silently inserted. Energy counters start at reset and
are not estimates of physically remaining energy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np
from isaac_drone.aero.math import finite_scalar, numeric_array, positive_dt


class BatteryLimitError(ValueError):
    """A step or actuation exceeds an identified battery operating limit."""


class ModelValidityError(ValueError):
    """Inputs lie outside the declared calibration/model domain."""


def _number(value, name: str, *, positive=False, nonnegative=False) -> float:
    value = finite_scalar(value, name)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else " and nonnegative" if nonnegative else ""))
    return value


def validate_ocv_table(value) -> np.ndarray:
    table = numeric_array(value, None, "ocv_soc_voltage_table")
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
        raise ValueError("ocv_soc_voltage_table must be a finite N x 2 table, N >= 2")
    if table[0, 0] != 0 or table[-1, 0] != 1 or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError("OCV SOC coordinates must strictly increase and cover exactly [0, 1]")
    if np.any(table[:, 1] <= 0) or np.any(np.diff(table[:, 1]) < 0):
        raise ValueError("OCV voltages must be positive and nondecreasing with SOC")
    table.setflags(write=False)
    return table


@dataclass(frozen=True)
class RCBranch:
    resistance_ohm: float
    capacitance_f: float
    initial_polarization_v: float

    def __post_init__(self):
        for name in ("resistance_ohm", "capacitance_f"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        if not np.isfinite(self.resistance_ohm * self.capacitance_f) or self.resistance_ohm * self.capacitance_f <= 0:
            raise ValueError("RC time constant must be finite and positive")
        object.__setattr__(self, "initial_polarization_v", _number(
            self.initial_polarization_v, "initial_polarization_v"))


@dataclass(frozen=True)
class BatteryState:
    elapsed_s: float
    soc: float
    open_circuit_voltage_v: float
    terminal_voltage_v: float | None
    current_a: float | None
    polarization_voltage_v: tuple[float, ...]
    energy_delivered_j: float
    ohmic_heat_j: float
    charge_throughput_c: float
    temperature_c: float | None
    flags: tuple[str, ...]


class PowerModel(Protocol):
    @property
    def state(self) -> BatteryState: ...
    def reset(self) -> BatteryState: ...
    def observe(self, current_a: float, *, temperature_c: float | None = None) -> BatteryState: ...
    def step(self, current_a: float, dt_s: float, *, temperature_c: float | None = None) -> BatteryState: ...


class EquivalentCircuitBattery:
    def __init__(self, *, capacity_ah: float, initial_soc: float,
                 ocv_soc_voltage_table, series_resistance_ohm: float,
                 rc_branches: Sequence[RCBranch], charge_coulombic_efficiency: float,
                 discharge_coulombic_efficiency: float, cutoff_voltage_v: float,
                 maximum_voltage_v: float, max_discharge_current_a: float,
                 max_charge_current_a: float, calibration_temperature_c: float,
                 valid_temperature_range_c, initial_temperature_c: float | None,
                 calibration_id: str):
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError("calibration_id must identify the source of battery parameters")
        self.calibration_id = calibration_id
        self.capacity_ah = _number(capacity_ah, "capacity_ah", positive=True)
        if not np.isfinite(self.capacity_ah * 3600):
            raise ValueError("capacity in coulombs must remain finite")
        self.initial_soc = _number(initial_soc, "initial_soc")
        if not 0 <= self.initial_soc <= 1:
            raise ValueError("initial_soc must lie in [0, 1]")
        self.r0 = _number(series_resistance_ohm, "series_resistance_ohm", nonnegative=True)
        self.ocv_table = validate_ocv_table(ocv_soc_voltage_table)
        self.rc_branches = tuple(rc_branches)
        if not all(isinstance(branch, RCBranch) for branch in self.rc_branches):
            raise TypeError("rc_branches must contain RCBranch instances")
        self.charge_efficiency = _number(charge_coulombic_efficiency, "charge_coulombic_efficiency", positive=True)
        self.discharge_efficiency = _number(discharge_coulombic_efficiency, "discharge_coulombic_efficiency", positive=True)
        if self.charge_efficiency > 1 or self.discharge_efficiency > 1:
            raise ValueError("coulombic efficiencies must lie in (0, 1]")
        self.cutoff_v = _number(cutoff_voltage_v, "cutoff_voltage_v", positive=True)
        self.maximum_v = _number(maximum_voltage_v, "maximum_voltage_v", positive=True)
        if self.maximum_v <= self.cutoff_v:
            raise ValueError("maximum_voltage_v must exceed cutoff_voltage_v")
        self.max_discharge_a = _number(max_discharge_current_a, "max_discharge_current_a", positive=True)
        self.max_charge_a = _number(max_charge_current_a, "max_charge_current_a", nonnegative=True)
        self.calibration_temperature_c = _number(calibration_temperature_c, "calibration_temperature_c")
        temp_range = numeric_array(valid_temperature_range_c, (2,), "valid_temperature_range_c")
        if temp_range.shape != (2,) or not np.isfinite(temp_range).all() or temp_range[0] > temp_range[1]:
            raise ValueError("valid_temperature_range_c must be finite [lower, upper]")
        if not temp_range[0] <= self.calibration_temperature_c <= temp_range[1]:
            raise ValueError("calibration temperature must be inside the declared valid range")
        self.temperature_range = tuple(float(x) for x in temp_range)
        self.initial_temperature_c = self._temperature(initial_temperature_c)
        self.reset()

    def ocv(self, soc: float) -> float:
        soc = _number(soc, "soc")
        if not 0 <= soc <= 1:
            raise ModelValidityError("OCV lookup cannot extrapolate outside SOC [0, 1]")
        return float(np.interp(soc, self.ocv_table[:, 0], self.ocv_table[:, 1]))

    def _temperature(self, temperature_c):
        if temperature_c is None:
            return None
        value = _number(temperature_c, "temperature_c")
        if not self.temperature_range[0] <= value <= self.temperature_range[1]:
            raise ModelValidityError("temperature is outside the battery calibration range")
        return value

    def _current(self, current_a):
        current = _number(current_a, "current_a")
        if current > self.max_discharge_a or current < -self.max_charge_a:
            raise BatteryLimitError("current exceeds the calibrated charge/discharge limit")
        return current

    def _flags(self, soc: float, terminal_v: float | None, temperature_c: float | None) -> tuple[str, ...]:
        flags = []
        if terminal_v is None:
            flags.append("terminal_voltage_unobserved")
        else:
            if terminal_v <= self.cutoff_v:
                flags.append("undervoltage")
            if terminal_v >= self.maximum_v:
                flags.append("overvoltage")
        if soc <= 0:
            flags.append("empty")
        if soc >= 1:
            flags.append("full")
        if temperature_c is None:
            flags.append("temperature_unobserved")
        return tuple(flags)

    @property
    def state(self) -> BatteryState:
        return self._state

    def reset(self) -> BatteryState:
        soc = self.initial_soc
        self._state = BatteryState(0.0, soc, self.ocv(soc), None, None,
            tuple(branch.initial_polarization_v for branch in self.rc_branches),
            0.0, 0.0, 0.0, self.initial_temperature_c,
            self._flags(soc, None, self.initial_temperature_c))
        return self._state

    def observe(self, current_a: float, *, temperature_c: float | None = None) -> BatteryState:
        """Set an instantaneous operating point without integrating SOC or time.

        Temperature is a fresh observation. Omitting it marks it unknown, rather
        than silently persisting a stale reading or running a thermal model.
        """
        current = self._current(current_a)
        temperature = self._temperature(temperature_c)
        old = self.state
        voltage = old.open_circuit_voltage_v - current * self.r0 - sum(old.polarization_voltage_v)
        self._state = BatteryState(old.elapsed_s, old.soc, old.open_circuit_voltage_v, voltage,
            current, old.polarization_voltage_v, old.energy_delivered_j, old.ohmic_heat_j,
            old.charge_throughput_c, temperature, self._flags(old.soc, voltage, temperature))
        return self._state

    def _ocv_integral(self, soc_start: float, soc_end: float, dt_s: float) -> float:
        if soc_start == soc_end:
            return self.ocv(soc_start) * dt_s
        low, high = sorted((soc_start, soc_end))
        knots = self.ocv_table[:, 0]
        socs = np.concatenate(([low], knots[(knots > low) & (knots < high)], [high]))
        volts = np.interp(socs, self.ocv_table[:, 0], self.ocv_table[:, 1])
        integral_soc = np.sum(0.5 * (volts[1:] + volts[:-1]) * np.diff(socs))
        return float(integral_soc / (high - low) * dt_s)

    def step(self, current_a: float, dt_s: float, *, temperature_c: float | None = None) -> BatteryState:
        """Advance an entire constant-current interval, atomically.

        SOC/current/temperature violations reject the interval without mutating
        state. Endpoint under/overvoltage is reported without clipping physical
        voltage; coupling code must stop or apply a calibrated protection policy.
        """
        dt = positive_dt(dt_s)
        current = self._current(current_a)
        temperature = self._temperature(temperature_c)
        old = self.state
        effective_current = current / self.discharge_efficiency if current >= 0 else current * self.charge_efficiency
        soc = old.soc - effective_current * dt / (self.capacity_ah * 3600)
        if not np.isfinite(soc) or soc < -1e-12 or soc > 1 + 1e-12:
            raise BatteryLimitError("step would leave SOC [0, 1]; reduce the interval or stop the load")
        # Only absorb roundoff at the exact endpoint, never physical overshoot.
        soc = min(1.0, max(0.0, soc))
        polarization = []
        polarization_integral = 0.0
        heat = current * current * self.r0 * dt
        for branch, initial_v in zip(self.rc_branches, old.polarization_voltage_v):
            tau = branch.resistance_ohm * branch.capacitance_f
            one_minus_decay = -np.expm1(-dt / tau)
            steady_v = current * branch.resistance_ohm
            change_v = steady_v - initial_v
            end_v = initial_v + change_v * one_minus_decay
            polarization.append(float(end_v))
            x = dt / tau
            if x < 1e-3:
                # Stable integrals of (1-exp(-t/tau)) and its square;
                # avoid cancellation for fast physics ticks / slow RC branches.
                a = tau * x * x * (0.5 + x * (-1/6 + x * (1/24 + x * (-1/120 + x/720))))
                b = tau * x * x * x * (1/3 + x * (-1/4 + x * (7/60 + x * (-1/24 + x * 31/2520))))
            else:
                a = dt - tau * one_minus_decay
                b = dt - 2 * tau * one_minus_decay + tau * (-np.expm1(-2 * x)) / 2
            polarization_integral += initial_v * dt + change_v * a
            branch_heat = (initial_v * initial_v * dt
                + 2 * initial_v * change_v * a + change_v * change_v * b) / branch.resistance_ohm
            heat += max(0.0, float(branch_heat))
        ocv = self.ocv(soc)
        terminal_v = ocv - current * self.r0 - sum(polarization)
        voltage_integral = self._ocv_integral(old.soc, soc, dt) - current * self.r0 * dt - polarization_integral
        energy = old.energy_delivered_j + current * voltage_integral
        heat_total = old.ohmic_heat_j + heat
        elapsed = old.elapsed_s + dt
        throughput = old.charge_throughput_c + abs(current) * dt
        if not np.isfinite([terminal_v, energy, heat_total, elapsed, throughput, *polarization]).all():
            raise ModelValidityError("battery arithmetic overflow; state has not been changed")
        self._state = BatteryState(elapsed, soc, ocv, float(terminal_v), current,
            tuple(polarization), float(energy), float(heat_total), throughput,
            temperature, self._flags(soc, terminal_v, temperature))
        return self._state
