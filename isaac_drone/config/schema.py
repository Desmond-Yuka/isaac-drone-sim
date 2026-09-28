"""Experiment configuration schema (``schema_version: 2``) without importing the simulator.

Sections: simulation, recording (optional), vehicle, limits, controller,
allocation, trajectory, completion, effects, power, scene, logging. Plugin
sections (``controller``, ``trajectory``) are validated by the parameter
dataclass registered for their ``kind``. Native Isaac Lab fields are validated
against the installed classes by the backend builder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from isaac_drone.control import CONTROLLERS, FlightLimits
from isaac_drone.core.params import parse_params
from isaac_drone.core.validation import ConfigurationError, finite_array
from isaac_drone.trajectories import TRAJECTORIES, CompletionParams

from .loader import apply_override, load_yaml_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "arl_robot_1.yaml"
HELIX_CONFIG = CONFIG_DIR / "helix.yaml"
SCHEMA_VERSION = 2
DEFAULT_RECORDING = {
    "enabled": False,
    "fps": 60,
    "width": 1280,
    "height": 720,
    "cameras": ["overview", "follow", "top"],
}
ROTOR_ORDER = ["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"]
_SECTIONS = {
    "schema_version",
    "simulation",
    "vehicle",
    "limits",
    "controller",
    "allocation",
    "trajectory",
    "completion",
    "effects",
    "power",
    "scene",
    "logging",
}


def _keys(data, allowed, required, path):
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    unknown, missing = set(data) - set(allowed), set(required) - set(data)
    if unknown or missing:
        raise ConfigurationError(f"{path}: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _number(value, path, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{path} must be a finite number")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ConfigurationError(f"{path} is outside its allowed range")


def _integer(value, path, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{path} must be an integer >= {minimum}")


def _boolean(value, path):
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} must be true or false")


def _array(value, shape, path):
    try:
        return finite_array(value, shape, path)
    except (ValueError, TypeError) as error:
        raise ConfigurationError(str(error)) from error


def _finite_tree(value, path="config"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"{path}: keys must be strings")
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError(f"{path}: NaN and infinity are not allowed")


def load_config(path: str | Path = DEFAULT_CONFIG, overrides=()) -> dict:
    """Load a YAML experiment (resolving ``extends``), apply ``key.path=value`` overrides, validate."""
    config = load_yaml_tree(path)
    for assignment in overrides:
        apply_override(config, assignment)
    config.setdefault("recording", deepcopy(DEFAULT_RECORDING))
    validate_config(config)
    return deepcopy(config)


def validate_config(config: dict) -> None:
    _keys(config, _SECTIONS | {"recording"}, _SECTIONS, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Only schema_version: {SCHEMA_VERSION} is supported; see docs/configuration.md for the layout"
        )
    _finite_tree(config)
    _validate_simulation(config["simulation"])
    _validate_recording(config.get("recording", DEFAULT_RECORDING), config["simulation"]["dt"])
    _validate_scene(config["scene"])
    _validate_vehicle(config["vehicle"], config["scene"])
    FlightLimits.from_config(config["limits"])
    CONTROLLERS.parse(config["controller"], "controller")
    _validate_allocation(config["allocation"])
    TRAJECTORIES.parse(config["trajectory"], "trajectory")
    if config["completion"] is not None:
        parse_params(CompletionParams, config["completion"], "completion")
    _validate_logging(config["logging"])
    from isaac_drone.effects import build_effects
    from isaac_drone.power import validate_power_config

    build_effects(config["effects"])
    validate_power_config(config["power"])


def _validate_simulation(sim):
    fields = {
        "dt",
        "control_decimation",
        "render_interval",
        "device",
        "gravity",
        "seed",
        "duration_s",
        "native_overrides",
    }
    _keys(sim, fields, fields, "simulation")
    _number(sim["dt"], "simulation.dt", positive=True)
    _number(sim["duration_s"], "simulation.duration_s", positive=True)
    for field in ("control_decimation", "render_interval"):
        _integer(sim[field], f"simulation.{field}", 1)
    _integer(sim["seed"], "simulation.seed", 0)
    gravity = _array(sim["gravity"], (3,), "simulation.gravity")
    if not np.allclose(gravity[:2], 0) or gravity[2] >= 0:
        raise ConfigurationError("This Z-up controller requires gravity along world -Z")
    if not isinstance(sim["device"], str) or not sim["device"]:
        raise ConfigurationError("simulation.device must be a nonempty string")
    if not isinstance(sim["native_overrides"], dict):
        raise ConfigurationError("simulation.native_overrides must be a mapping")
    if {"dt", "gravity", "device", "render_interval", "use_newton_actuators"} & sim["native_overrides"].keys():
        raise ConfigurationError("native overrides cannot shadow timing/gravity/device or native actuator execution")


def _validate_recording(recording, dt):
    fields = {"enabled", "fps", "width", "height", "cameras"}
    _keys(recording, fields, fields, "recording")
    _boolean(recording["enabled"], "recording.enabled")
    for field in ("fps", "width", "height"):
        _integer(recording[field], f"recording.{field}", 1)
    for field in ("width", "height"):
        if recording[field] % 2:
            raise ConfigurationError(f"recording.{field} must be even for H.264 video")
    if recording["enabled"] and recording["fps"] > 1.0 / dt:
        raise ConfigurationError("recording.fps cannot exceed the simulation physics frequency (1 / simulation.dt)")
    cameras = recording["cameras"]
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(not isinstance(name, str) or name not in {"overview", "follow", "top"} for name in cameras)
        or len(cameras) != len(set(cameras))
    ):
        raise ConfigurationError("recording.cameras must be a nonempty list of unique overview/follow/top names")


def _validate_vehicle(vehicle, scene):
    fields = {
        "name",
        "asset_path",
        "prim_path",
        "geometry_tolerance_m",
        "allocation_matrix",
        "rotor_directions",
        "rotor_direction_source",
        "thrusters",
        "initial_state",
        "launch",
        "native_overrides",
        "usd_overrides",
    }
    _keys(vehicle, fields, fields, "vehicle")
    if vehicle["name"] != "arl_robot_1":
        raise ConfigurationError("This configuration supports ARL-Robot-1 only")
    for key in ("asset_path", "prim_path"):
        if not isinstance(vehicle[key], str) or not vehicle[key]:
            raise ConfigurationError(f"vehicle.{key} must be a nonempty string")
    if not vehicle["prim_path"].startswith("/World/") or any(c in vehicle["prim_path"] for c in "{}*[]"):
        raise ConfigurationError("vehicle.prim_path must name one concrete prim below /World")
    _number(vehicle["geometry_tolerance_m"], "vehicle.geometry_tolerance_m", positive=True)
    _array(vehicle["allocation_matrix"], (6, 4), "vehicle.allocation_matrix")
    directions = _array(vehicle["rotor_directions"], (4,), "vehicle.rotor_directions")
    if not np.isin(directions, (-1, 1)).all():
        raise ConfigurationError("rotor_directions must be four -1/+1 signs")
    if vehicle["rotor_direction_source"] not in ("simulation_diagonal_pairs", "measured", "unverified_upstream"):
        raise ConfigurationError("rotor_direction_source must document the rotation-direction provenance")
    if not isinstance(vehicle["native_overrides"], dict):
        raise ConfigurationError("vehicle.native_overrides must be a mapping")
    reserved = {"init_state", "actuators", "allocation_matrix", "rotor_directions", "prim_path", "class_type"}
    if reserved & vehicle["native_overrides"].keys():
        raise ConfigurationError("vehicle.native_overrides shadows an authoritative vehicle field")
    spawn = vehicle["native_overrides"].get("spawn", {})
    if not isinstance(spawn, dict) or {"usd_path", "func"} & spawn.keys():
        raise ConfigurationError("spawn overrides cannot shadow usd_path/func")
    if not isinstance(vehicle["usd_overrides"], list):
        raise ConfigurationError("vehicle.usd_overrides must be a list")
    for override in vehicle["usd_overrides"]:
        _keys(override, {"prim_path", "attribute", "value"}, {"prim_path", "attribute", "value"}, "usd_overrides[]")
    _validate_thrusters(vehicle["thrusters"])
    initial = vehicle["initial_state"]
    fields = {"pos", "quaternion_wxyz", "lin_vel", "ang_vel", "rps"}
    _keys(initial, fields, fields, "vehicle.initial_state")
    for field in ("pos", "lin_vel", "ang_vel"):
        _array(initial[field], (3,), f"vehicle.initial_state.{field}")
    quat = _array(initial["quaternion_wxyz"], (4,), "initial quaternion")
    if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-8):
        raise ConfigurationError("initial quaternion_wxyz must be normalized")
    _keys(initial["rps"], ROTOR_ORDER, ROTOR_ORDER, "vehicle.initial_state.rps")
    for name, rps in initial["rps"].items():
        _number(rps, f"initial_state.rps.{name}", nonnegative=True)
    launch = vehicle["launch"]
    fields = {"from_ground", "ground_z_m", "clearance_m", "spin_up_s"}
    _keys(launch, fields, fields, "vehicle.launch")
    _boolean(launch["from_ground"], "vehicle.launch.from_ground")
    _number(launch["ground_z_m"], "vehicle.launch.ground_z_m")
    _number(launch["clearance_m"], "vehicle.launch.clearance_m", nonnegative=True)
    _number(launch["spin_up_s"], "vehicle.launch.spin_up_s", nonnegative=True)
    if launch["from_ground"]:
        if not scene["ground"]["enabled"]:
            raise ConfigurationError("Ground launch requires the configured physical ground plane")
        if any(initial["rps"].values()) or np.any(initial["lin_vel"]) or np.any(initial["ang_vel"]):
            raise ConfigurationError("Ground launch requires initially stopped motors and zero initial velocities")
        if launch["spin_up_s"] <= 0:
            raise ConfigurationError("Ground launch requires a positive vehicle.launch.spin_up_s for motor spin-up")


def _validate_thrusters(thrusters):
    fields = {
        "thruster_names_expr",
        "thrust_range",
        "thrust_const_range",
        "tau_inc_range",
        "tau_dec_range",
        "torque_to_thrust_ratio",
        "max_thrust_rate",
        "use_discrete_approximation",
        "integration_scheme",
    }
    _keys(thrusters, fields, fields, "vehicle.thrusters")
    if thrusters["thruster_names_expr"] != ROTOR_ORDER:
        raise ConfigurationError("ARL action order must be back_left, back_right, front_left, front_right")
    for field in ("thrust_range", "thrust_const_range", "tau_inc_range", "tau_dec_range"):
        bounds = _array(thrusters[field], (2,), f"vehicle.thrusters.{field}")
        if bounds[1] < bounds[0] or bounds[0] < 0 or (field != "thrust_range" and bounds[0] == 0):
            raise ConfigurationError(f"{field} must have ordered positive endpoints (minimum thrust may be zero)")
    if thrusters["thrust_range"][1] <= 0:
        raise ConfigurationError("maximum thrust must be positive")
    _number(thrusters["torque_to_thrust_ratio"], "torque_to_thrust_ratio", nonnegative=True)
    _number(thrusters["max_thrust_rate"], "max_thrust_rate", positive=True)
    _boolean(thrusters["use_discrete_approximation"], "use_discrete_approximation")
    if thrusters["integration_scheme"] not in ("rk4", "euler"):
        raise ConfigurationError("integration_scheme must be rk4 or euler")


def _validate_allocation(alloc):
    _keys(alloc, {"weights", "regularization"}, {"weights", "regularization"}, "allocation")
    if np.any(_array(alloc["weights"], (6,), "allocation.weights") <= 0):
        raise ConfigurationError("allocation.weights must be positive")
    _number(alloc["regularization"], "allocation.regularization", nonnegative=True)


def _validate_scene(scene):
    _keys(scene, {"ground", "light"}, {"ground", "light"}, "scene")
    for field in ("ground", "light"):
        allowed = (
            {"enabled", "prim_path", "native_overrides"} if field == "ground" else {"enabled", "prim_path", "intensity"}
        )
        _keys(scene[field], allowed, allowed, f"scene.{field}")
        _boolean(scene[field]["enabled"], f"scene.{field}.enabled")
        if not isinstance(scene[field]["prim_path"], str) or not scene[field]["prim_path"].startswith("/World/"):
            raise ConfigurationError(f"scene.{field}.prim_path must be below /World")
    if not isinstance(scene["ground"]["native_overrides"], dict):
        raise ConfigurationError("scene.ground.native_overrides must be a mapping")
    _number(scene["light"]["intensity"], "scene.light.intensity", nonnegative=True)


def _validate_logging(log):
    fields = {"directory", "every_n_steps", "flush_every_n_records"}
    _keys(log, fields, fields, "logging")
    if not isinstance(log["directory"], str) or not log["directory"]:
        raise ConfigurationError("logging.directory must be nonempty")
    for field in ("every_n_steps", "flush_every_n_records"):
        _integer(log[field], f"logging.{field}", 1)


def resolve_asset(config: dict, repo_root: Path = REPO_ROOT) -> Path:
    """Require a local USD; never silently select a different network asset."""
    path = Path(config["vehicle"]["asset_path"]).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigurationError(f"Local ARL USD missing: {path}; run scripts/download_assets.sh explicitly")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ConfigurationError(f"{path} is a Git LFS pointer; run git lfs pull")
    return path


def assert_flight_ready(config: dict, matrix: np.ndarray) -> None:
    """Require explicit direction provenance and four independent control channels."""
    if config["vehicle"]["rotor_direction_source"] == "unverified_upstream":
        raise ConfigurationError("Specify measured or explicitly chosen simulation rotor directions before flight")
    allocation = _array(matrix, (6, 4), "runtime allocation matrix")
    if not np.allclose(allocation[:3], np.tile([[0.0], [0.0], [1.0]], (1, 4)), atol=1e-7, rtol=0):
        raise ConfigurationError(
            "The upright controllers require all thrust axes parallel to body +Z; "
            "tilted rotors need a different controller"
        )
    if np.linalg.matrix_rank(allocation[[2, 3, 4, 5]]) != 4:
        raise ConfigurationError("Actual rotor geometry/directions have rank < 4 for collective/roll/pitch/yaw")
