"""Explicit electrical-load and voltage-to-actuator coupling contracts.

No current is inferred from thrust times airspeed. Electrical power requires a
measured current input or an explicitly supplied calibrated load model. Likewise,
no rotor-speed or thrust scaling with battery voltage is assumed: an injected
actuator-envelope calibration provides the allowable thrust for each motor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from isaac_drone.core.types import VehicleState
from isaac_drone.core.validation import finite_scalar, numeric_array

from .battery import (
    BatteryLimitError,
    BatteryState,
    EquivalentCircuitBattery,
    ModelValidityError,
    PowerModel,
    RCBranch,
    validate_ocv_table,
)


@dataclass(frozen=True)
class ThrustBounds:
    lower_n: np.ndarray
    upper_n: np.ndarray

    def __post_init__(self):
        lower = numeric_array(self.lower_n, None, "lower_n")
        upper = numeric_array(self.upper_n, None, "upper_n")
        if lower.ndim != 1 or len(lower) == 0 or upper.shape != lower.shape:
            raise ValueError("thrust bounds must be matching, nonempty one-dimensional arrays")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower < 0) or np.any(upper < lower):
            raise ValueError("thrust bounds must be finite with 0 <= lower_n <= upper_n")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower_n", lower)
        object.__setattr__(self, "upper_n", upper)


class LoadModel(Protocol):
    """Measured/calibrated pack current, including ESC/motor/accessory losses.

    The implementation owns any nonlinear electrical operating-point solve.
    battery_state is historical state, not a guaranteed contemporaneous bus
    voltage. Returning thrust * airspeed / voltage is not a supported default.
    """

    def reset(self) -> None: ...
    def current_a(self, vehicle_state: VehicleState, telemetry: Mapping, battery_state: BatteryState) -> float: ...


class ActuatorEnvelope(Protocol):
    """Calibrated per-rotor available thrust under the electrical state.

    The contract produces bounds, not a physical motor/ESC voltage response.
    A backend that independently holds a prescribed thrust cannot thereby claim
    it simulates that response. Such behavior needs a calibrated actuator plugin.
    """

    def reset(self) -> None: ...
    def thrust_bounds_n(
        self, battery_state: BatteryState, vehicle_state: VehicleState, telemetry: Mapping
    ) -> ThrustBounds: ...


class PowerSystem:
    def __init__(
        self,
        battery: PowerModel,
        *,
        current_source: str,
        actuator_envelope: ActuatorEnvelope,
        load_model: LoadModel | None = None,
    ):
        if current_source not in ("external_measurement", "load_model"):
            raise ValueError("current_source must be 'external_measurement' or 'load_model'")
        if current_source == "load_model" and load_model is None:
            raise ValueError("enabled power requires an injected calibrated LoadModel")
        if current_source == "external_measurement" and load_model is not None:
            raise ValueError("external_measurement and LoadModel are mutually exclusive current sources")
        if actuator_envelope is None or not callable(getattr(actuator_envelope, "thrust_bounds_n", None)):
            raise ValueError("enabled power requires a calibrated ActuatorEnvelope; no voltage scaling is assumed")
        if not callable(getattr(actuator_envelope, "reset", None)):
            raise TypeError("ActuatorEnvelope must implement reset")
        if load_model is not None and not all(
            callable(getattr(load_model, method, None)) for method in ("current_a", "reset")
        ):
            raise TypeError("LoadModel must implement current_a and reset")
        self.battery = battery
        self.current_source = current_source
        self.actuator_envelope = actuator_envelope
        self.load_model = load_model

    @property
    def state(self) -> BatteryState:
        return self.battery.state

    def reset(self) -> BatteryState:
        state = self.battery.reset()
        self.actuator_envelope.reset()
        if self.load_model is not None:
            self.load_model.reset()
        return state

    def initialize(self, current_a: float, *, temperature_c: float | None = None) -> BatteryState:
        """Supply an identified initial current/temperature operating point."""
        return self.battery.observe(current_a, temperature_c=temperature_c)

    def update(
        self,
        vehicle_state: VehicleState,
        telemetry: Mapping,
        dt_s: float,
        *,
        measured_current_a: float | None = None,
        temperature_c: float | None = None,
    ) -> BatteryState:
        if not isinstance(telemetry, Mapping):
            raise TypeError("telemetry must be a mapping of actual actuator/load observations")
        if self.current_source == "external_measurement":
            if measured_current_a is None:
                raise ValueError("external_measurement requires measured_current_a at every physics step")
            current = measured_current_a
        else:
            if measured_current_a is not None:
                raise ValueError("cannot mix measured_current_a with configured load_model")
            current = self.load_model.current_a(vehicle_state, telemetry, self.state)
        return self.battery.step(current, dt_s, temperature_c=temperature_c)

    def thrust_bounds_n(self, vehicle_state: VehicleState, telemetry: Mapping) -> ThrustBounds:
        """Fail closed for unobserved operating state and active electrical limits."""
        if not isinstance(telemetry, Mapping):
            raise TypeError("telemetry must be a mapping")
        unknown = {"terminal_voltage_unobserved", "temperature_unobserved"}.intersection(self.state.flags)
        if unknown:
            raise ModelValidityError("power operating state is not identified: " + ", ".join(sorted(unknown)))
        unsafe = {"empty", "undervoltage", "overvoltage"}.intersection(self.state.flags)
        if unsafe:
            raise BatteryLimitError("battery cannot authorize thrust: " + ", ".join(sorted(unsafe)))
        bounds = self.actuator_envelope.thrust_bounds_n(self.state, vehicle_state, telemetry)
        if not isinstance(bounds, ThrustBounds):
            raise TypeError("ActuatorEnvelope must return ThrustBounds")
        return bounds


_BATTERY_FIELDS = (
    "capacity_ah",
    "initial_soc",
    "ocv_soc_voltage_table",
    "series_resistance_ohm",
    "charge_coulombic_efficiency",
    "discharge_coulombic_efficiency",
    "cutoff_voltage_v",
    "maximum_voltage_v",
    "max_discharge_current_a",
    "max_charge_current_a",
    "calibration_temperature_c",
    "valid_temperature_range_c",
    "calibration_id",
)


def _validate_optional_parameters(config: Mapping) -> None:
    if config.get("model") is not None and config["model"] != "equivalent_circuit":
        raise ValueError("unsupported power.model")
    if config.get("current_source") is not None and config["current_source"] not in (
        "external_measurement",
        "load_model",
    ):
        raise ValueError("power.current_source must be external_measurement or load_model")
    if config.get("calibration_id") is not None and (
        not isinstance(config["calibration_id"], str) or not config["calibration_id"].strip()
    ):
        raise ValueError("calibration_id must be a nonempty string or None")
    positive = {
        "capacity_ah",
        "charge_coulombic_efficiency",
        "discharge_coulombic_efficiency",
        "cutoff_voltage_v",
        "maximum_voltage_v",
        "max_discharge_current_a",
    }
    nonnegative = {"series_resistance_ohm", "max_charge_current_a"}
    scalars = positive | nonnegative | {"initial_soc", "calibration_temperature_c", "initial_temperature_c"}
    for key in scalars:
        if config.get(key) is not None:
            value = finite_scalar(config[key], key)
            if (key in positive and value <= 0) or (key in nonnegative and value < 0):
                raise ValueError(f"{key} is outside its allowed range")
            if (
                key in ("initial_soc", "charge_coulombic_efficiency", "discharge_coulombic_efficiency")
                and not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must lie in [0, 1]")
    if config.get("ocv_soc_voltage_table") is not None:
        validate_ocv_table(config["ocv_soc_voltage_table"])
    if config.get("valid_temperature_range_c") is not None:
        bounds = numeric_array(config["valid_temperature_range_c"], (2,), "valid_temperature_range_c")
        if bounds[0] > bounds[1]:
            raise ValueError("temperature range must be ordered")
        for key in ("calibration_temperature_c", "initial_temperature_c"):
            if config.get(key) is not None and not bounds[0] <= config[key] <= bounds[1]:
                raise ValueError(f"{key} must lie within valid_temperature_range_c")
    if (
        config.get("cutoff_voltage_v") is not None
        and config.get("maximum_voltage_v") is not None
        and config["cutoff_voltage_v"] >= config["maximum_voltage_v"]
    ):
        raise ValueError("maximum_voltage_v must exceed cutoff_voltage_v")
    if config.get("rc_branches") is not None:
        if not isinstance(config["rc_branches"], (list, tuple)):
            raise ValueError("rc_branches must be a list or None")
        allowed = {"resistance_ohm", "capacitance_f", "initial_polarization_v"}
        for branch in config["rc_branches"]:
            if not isinstance(branch, Mapping):
                raise ValueError("each RC branch must be a mapping")
            if set(branch) - allowed:
                raise ValueError("RC branch has unsupported keys")
            for key in allowed:
                if branch.get(key) is not None:
                    value = finite_scalar(branch[key], key)
                    if key != "initial_polarization_v" and value <= 0:
                        raise ValueError(f"RC {key} must be positive")


def build_battery(config: Mapping) -> EquivalentCircuitBattery | None:
    """Build an independently testable battery; a disabled model returns None."""
    if not isinstance(config, Mapping):
        raise ValueError("power must be a mapping")
    allowed = set(_BATTERY_FIELDS) | {"enabled", "model", "current_source", "rc_branches", "initial_temperature_c"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("power has unsupported keys: " + ", ".join(sorted(map(str, unknown))))
    if type(config.get("enabled")) is not bool:
        raise ValueError("power.enabled must be an explicit boolean")
    _validate_optional_parameters(config)
    if not config["enabled"]:
        return None
    if config.get("model") != "equivalent_circuit":
        raise ValueError("unsupported power.model; inject a PowerModel for other calibrated models")
    missing = [key for key in _BATTERY_FIELDS if config.get(key) is None]
    if missing:
        raise ValueError("enabled power requires identified parameters: " + ", ".join(missing))
    if "initial_temperature_c" not in config:
        raise ValueError("initial_temperature_c must be explicit; None denotes unobserved temperature")
    branches = config.get("rc_branches")
    if not isinstance(branches, (list, tuple)):
        raise ValueError("power.rc_branches must be an explicit list; [] declares no RC branches")
    parsed_branches = []
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValueError("each RC branch must be a mapping")
        keys = ("resistance_ohm", "capacitance_f", "initial_polarization_v")
        if set(branch) - set(keys):
            raise ValueError("RC branch has unsupported keys")
        if any(branch.get(key) is None for key in keys):
            raise ValueError("every RC branch requires resistance, capacitance and initial polarization")
        parsed_branches.append(RCBranch(**{key: branch[key] for key in keys}))
    return EquivalentCircuitBattery(
        **{key: config[key] for key in _BATTERY_FIELDS},
        initial_temperature_c=config["initial_temperature_c"],
        rc_branches=parsed_branches,
    )


def validate_power_config(config: Mapping) -> None:
    """Offline schema/parameter validation without requiring runtime plugins."""
    battery = build_battery(config)
    if battery is not None and config.get("current_source") not in ("external_measurement", "load_model"):
        raise ValueError("enabled power.current_source must be external_measurement or load_model")


def build_power(
    config: Mapping, *, load_model: LoadModel | None = None, actuator_envelope: ActuatorEnvelope | None = None
) -> PowerSystem | None:
    validate_power_config(config)
    battery = build_battery(config)
    if battery is None:
        return None
    return PowerSystem(
        battery, current_source=config["current_source"], load_model=load_model, actuator_envelope=actuator_envelope
    )
