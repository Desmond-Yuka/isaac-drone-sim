"""Composable external CoM wrenches and environmental model construction.

Physics gravity, contacts and rotor forces belong to the backend, and must not
be duplicated here. All returned wrenches use body axes about the current CoM.
A disabled model is absent, not a declaration of physically negligible effects.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from isaac_drone.core.types import VehicleState, Wrench
from isaac_drone.effects._math import body_to_world, dissipative_matrix, finite_scalar, positive_dt
from isaac_drone.effects._math import numeric_array as finite_array
from isaac_drone.effects.drag import BodyDrag
from isaac_drone.effects.wind import Gust, UniformGustWind, WindField


class EffectModel(Protocol):
    """A plugin may be stateful, and is reset for each repeatable simulation."""
    def reset(self, seed: int | None = None) -> None: ...
    def evaluate(self, state: VehicleState, dt_s: float,
                 wind: WindField | None = None) -> Wrench: ...


class ConstantWrench:
    """Sampled constant force/couple in the half-open interval [start, end).

    A body application point is a CoM-relative offset; a world application point
    is an absolute world position. Neither means an uncorrected USD link origin.
    With neither point supplied the force acts at CoM. The free torque is in the
    same frame as force and is added to the moment of the force exactly once.
    Infinite end time is expressed as None, not an IEEE infinity.
    """
    def __init__(self, *, force_n, torque_nm, frame: str,
                 start_time_s: float, end_time_s: float | None,
                 application_point_b_m=None, application_point_w_m=None):
        if frame not in ("body", "world"):
            raise ValueError("constant wrench frame must be 'body' or 'world'")
        finite_scalar(start_time_s, "start_time_s")
        if start_time_s < 0:
            raise ValueError("start_time_s must be finite and nonnegative")
        if end_time_s is not None:
            finite_scalar(end_time_s, "end_time_s")
        if end_time_s is not None and end_time_s <= start_time_s:
            raise ValueError("end_time_s must exceed start_time_s, or be None")
        if application_point_b_m is not None and application_point_w_m is not None:
            raise ValueError("specify at most one body or world application point")
        self.force = finite_array(force_n, (3,), "force_n")
        self.torque = finite_array(torque_nm, (3,), "torque_nm")
        self.frame = frame
        self.start = float(start_time_s)
        self.end = end_time_s
        self.point_b = None if application_point_b_m is None else finite_array(
            application_point_b_m, (3,), "application_point_b_m")
        self.point_w = None if application_point_w_m is None else finite_array(
            application_point_w_m, (3,), "application_point_w_m")

    def reset(self, seed: int | None = None) -> None:
        """Stateless model."""

    def evaluate(self, state: VehicleState, dt_s: float,
                 wind: WindField | None = None) -> Wrench:
        positive_dt(dt_s)
        if state.time_s < self.start or (self.end is not None and state.time_s >= self.end):
            return Wrench.zero()
        rotation = body_to_world(state.quaternion_wxyz)
        force = self.force if self.frame == "body" else rotation.T @ self.force
        torque = self.torque if self.frame == "body" else rotation.T @ self.torque
        if self.point_b is not None:
            torque = torque + np.cross(self.point_b, force)
        elif self.point_w is not None:
            torque = torque + np.cross(rotation.T @ (self.point_w - state.position_w), force)
        return Wrench(force_b=force, torque_b=torque)


class EffectsPipeline:
    def __init__(self, effects: Sequence[EffectModel] = (), wind: WindField | None = None):
        self.effects = tuple(effects)
        self.wind = wind
        self._last_time_s = None

    def reset(self, seed: int | None = None) -> None:
        # Spawn independent deterministic streams; adding a plugin cannot cause
        # every existing plugin to share identical random draws.
        streams = np.random.SeedSequence(seed).spawn(len(self.effects) + 1)
        if self.wind is not None:
            self.wind.reset(int(streams[0].generate_state(1)[0]))
        for effect, stream in zip(self.effects, streams[1:]):
            effect.reset(int(stream.generate_state(1)[0]))
        self._last_time_s = None

    def evaluate(self, state: VehicleState, dt_s: float) -> Wrench:
        positive_dt(dt_s)
        if self._last_time_s is not None and state.time_s <= self._last_time_s:
            raise ValueError("effects must be evaluated once per increasing physics time; reset first")
        if self.wind is not None:
            self.wind.advance(state.time_s, dt_s)
        wrench = Wrench.zero()
        for effect in self.effects:
            contribution = effect.evaluate(state, dt_s, self.wind)
            if not isinstance(contribution, Wrench):
                raise TypeError("effect plugins must return Wrench")
            wrench = wrench + contribution
        self._last_time_s = state.time_s
        return wrench


def _section(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(config: Mapping, allowed: set[str], name: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"{name} has unsupported keys: {', '.join(sorted(map(str, unknown)))}")


def _enabled(config: Mapping, name: str) -> bool:
    if "enabled" not in config or type(config["enabled"]) is not bool:
        raise ValueError(f"{name}.enabled must be an explicit boolean")
    return config["enabled"]


def _required(config: Mapping, key: str, name: str):
    if key not in config or config[key] is None:
        raise ValueError(f"enabled {name} requires explicit {key}; it has not been identified")
    return config[key]


def _optional_array(config: Mapping, key: str, shape: tuple[int, ...]) -> None:
    if config.get(key) is not None:
        finite_array(config[key], shape, key)


def _optional_time(config: Mapping, key: str, *, positive=False) -> None:
    if config.get(key) is not None:
        value = finite_scalar(config[key], key)
        if value < 0 or (positive and value == 0):
            raise ValueError(f"{key} must be {'positive' if positive else 'nonnegative'}")


def _validate_optional_wind(config: Mapping) -> None:
    # Even disabled settings remain structurally checked, while None retains its
    # meaning of unidentified. This catches dormant typos before later enabling.
    if config.get("model") is not None and config["model"] != "uniform_gusts":
        raise ValueError("unsupported wind model")
    for key in ("velocity_w_m_s", "reference_position_w_m"):
        _optional_array(config, key, (3,))
    _optional_array(config, "spatial_gradient_per_s", (3, 3))
    if config.get("gusts") is not None:
        if not isinstance(config["gusts"], (list, tuple)):
            raise ValueError("wind.gusts must be a list")
        for item in config["gusts"]:
            item = _section(item, "gust")
            _keys(item, {"start_time_s", "duration_s", "delta_velocity_w_m_s"}, "gust")
            _optional_time(item, "start_time_s")
            _optional_time(item, "duration_s", positive=True)
            _optional_array(item, "delta_velocity_w_m_s", (3,))
    if config.get("turbulence") is not None:
        turbulence = _section(config["turbulence"], "turbulence")
        _keys(turbulence, {"enabled", "time_constant_s", "stationary_std_w_m_s"}, "turbulence")
        _enabled(turbulence, "turbulence")
        for key in ("time_constant_s", "stationary_std_w_m_s"):
            _optional_array(turbulence, key, (3,))
            if turbulence.get(key) is not None:
                values = np.asarray(turbulence[key])
                if np.any(values < 0) or (key == "time_constant_s" and np.any(values == 0)):
                    raise ValueError(f"turbulence.{key} is outside its allowed range")


def _validate_optional_disturbance(item: Mapping) -> None:
    if item.get("model") is not None and item["model"] != "constant_wrench":
        raise ValueError("unsupported disturbance model")
    if item.get("frame") is not None and item["frame"] not in ("body", "world"):
        raise ValueError("disturbance frame must be body or world")
    for key in ("force_n", "torque_nm", "application_point_b_m", "application_point_w_m"):
        _optional_array(item, key, (3,))
    for key in ("start_time_s", "end_time_s"):
        _optional_time(item, key)
    if item.get("start_time_s") is not None and item.get("end_time_s") is not None:
        if item["end_time_s"] <= item["start_time_s"]:
            raise ValueError("end_time_s must exceed start_time_s")
    if item.get("application_point_b_m") is not None and item.get("application_point_w_m") is not None:
        raise ValueError("specify at most one application point frame")


def build_effects(config: Mapping) -> EffectsPipeline:
    """Build from the YAML effects section; missing enabled parameters fail closed."""
    config = _section(config, "effects")
    _keys(config, {"wind", "aerodynamic", "disturbances"}, "effects")
    wind_config = _section(_required(config, "wind", "effects"), "wind")
    aero_config = _section(_required(config, "aerodynamic", "effects"), "aerodynamic")
    _keys(wind_config, {"enabled", "model", "velocity_w_m_s", "spatial_gradient_per_s",
                        "reference_position_w_m", "gusts", "turbulence"}, "wind")
    _keys(aero_config, {"enabled", "linear_drag_b_kg_s", "quadratic_drag_b_kg_m",
                        "angular_linear_drag_b_nm_s", "center_of_pressure_b_m"}, "aerodynamic")
    _validate_optional_wind(wind_config)
    for key in ("linear_drag_b_kg_s", "angular_linear_drag_b_nm_s"):
        if aero_config.get(key) is not None:
            dissipative_matrix(aero_config[key], key)
    for key in ("quadratic_drag_b_kg_m", "center_of_pressure_b_m"):
        _optional_array(aero_config, key, (3,))
    if aero_config.get("quadratic_drag_b_kg_m") is not None and np.any(np.asarray(aero_config["quadratic_drag_b_kg_m"]) < 0):
        raise ValueError("quadratic_drag_b_kg_m must be nonnegative")
    wind = None
    if _enabled(wind_config, "wind"):
        if _required(wind_config, "model", "wind") != "uniform_gusts":
            raise ValueError("unsupported wind model; inject a WindField plugin for other models")
        gust_configs = _required(wind_config, "gusts", "wind")
        if not isinstance(gust_configs, (list, tuple)):
            raise ValueError("wind.gusts must be a list")
        gusts = []
        for item in gust_configs:
            item = _section(item, "gust")
            _keys(item, {"start_time_s", "duration_s", "delta_velocity_w_m_s"}, "gust")
            gusts.append(Gust(*(_required(item, key, "gust") for key in (
                "start_time_s", "duration_s", "delta_velocity_w_m_s"))))
        turbulence = _section(_required(wind_config, "turbulence", "wind"), "turbulence")
        _keys(turbulence, {"enabled", "time_constant_s", "stationary_std_w_m_s"}, "turbulence")
        tau = std = None
        if _enabled(turbulence, "turbulence"):
            tau = _required(turbulence, "time_constant_s", "turbulence")
            std = _required(turbulence, "stationary_std_w_m_s", "turbulence")
        wind = UniformGustWind(_required(wind_config, "velocity_w_m_s", "wind"), gusts=gusts,
            spatial_gradient_per_s=wind_config.get("spatial_gradient_per_s"),
            reference_position_w_m=wind_config.get("reference_position_w_m"),
            turbulence_time_constant_s=tau, turbulence_std_w_m_s=std)
    effects = []
    if _enabled(aero_config, "aerodynamic"):
        if wind is None:
            raise ValueError("enabled aerodynamics requires an explicit wind field, including explicit still air")
        keys = ("linear_drag_b_kg_s", "quadratic_drag_b_kg_m",
                "angular_linear_drag_b_nm_s", "center_of_pressure_b_m")
        effects.append(BodyDrag(**{key: _required(aero_config, key, "aerodynamic") for key in keys}))
    disturbances = _required(config, "disturbances", "effects")
    if not isinstance(disturbances, (list, tuple)):
        raise ValueError("effects.disturbances must be a list")
    for item in disturbances:
        item = _section(item, "disturbance")
        _keys(item, {"enabled", "model", "force_n", "torque_nm", "frame", "start_time_s",
                     "end_time_s", "application_point_b_m", "application_point_w_m"}, "disturbance")
        _validate_optional_disturbance(item)
        if not _enabled(item, "disturbance"):
            continue
        if _required(item, "model", "disturbance") != "constant_wrench":
            raise ValueError("unsupported disturbance model; inject an EffectModel plugin")
        if "end_time_s" not in item:
            raise ValueError("constant_wrench requires explicit end_time_s (None means no end)")
        if "application_point_b_m" not in item and "application_point_w_m" not in item:
            raise ValueError("constant_wrench requires an explicit application point (None means CoM)")
        keys = ("force_n", "torque_nm", "frame", "start_time_s")
        effects.append(ConstantWrench(**{key: _required(item, key, "disturbance") for key in keys},
            end_time_s=item["end_time_s"], application_point_b_m=item.get("application_point_b_m"),
            application_point_w_m=item.get("application_point_w_m")))
    return EffectsPipeline(effects, wind)
