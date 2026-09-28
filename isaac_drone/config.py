"""Strict YAML loading without importing the simulator. Native fields are
validated against the installed Isaac Lab classes by the backend builder.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
import math

import numpy as np
import yaml

from .types import finite_array

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arl_robot_1.yaml"
HELIX_CONFIG = REPO_ROOT / "configs" / "arl_robot_1_helix.yaml"


class ConfigurationError(ValueError):
    """Invalid or incomplete explicitly selected configuration."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("YAML keys must be strings")
        if key in result:
            raise ConfigurationError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


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


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        try:
            config = yaml.load(stream, Loader=_UniqueLoader)
        except yaml.YAMLError as error:
            raise ConfigurationError(str(error)) from error
    validate_config(config)
    return deepcopy(config)


def validate_config(config: dict) -> None:
    top = {"schema_version", "simulation", "vehicle", "control", "allocation", "effects", "power",
           "scene", "trajectory", "logging"}
    _keys(config, top, top, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigurationError("Only schema_version: 1 is supported")
    _finite_tree(config)
    sim = config["simulation"]
    fields = {"dt", "control_decimation", "render_interval", "device", "gravity", "seed", "duration_s", "native_overrides"}
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
    vehicle = config["vehicle"]
    fields = {"name", "asset_path", "prim_path", "geometry_tolerance_m", "allocation_matrix", "rotor_directions",
              "rotor_direction_source", "thrusters", "initial_state", "launch", "native_overrides", "usd_overrides"}
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
    thrusters = vehicle["thrusters"]
    fields = {"thruster_names_expr", "thrust_range", "thrust_const_range", "tau_inc_range", "tau_dec_range",
              "torque_to_thrust_ratio", "max_thrust_rate", "use_discrete_approximation", "integration_scheme"}
    _keys(thrusters, fields, fields, "vehicle.thrusters")
    names = thrusters["thruster_names_expr"]
    expected = ["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"]
    if names != expected:
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
    initial = vehicle["initial_state"]
    fields = {"pos", "quaternion_wxyz", "lin_vel", "ang_vel", "rps"}
    _keys(initial, fields, fields, "vehicle.initial_state")
    for field in ("pos", "lin_vel", "ang_vel"):
        _array(initial[field], (3,), f"vehicle.initial_state.{field}")
    quat = _array(initial["quaternion_wxyz"], (4,), "initial quaternion")
    if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-8):
        raise ConfigurationError("initial quaternion_wxyz must be normalized")
    _keys(initial["rps"], expected, expected, "vehicle.initial_state.rps")
    for name, rps in initial["rps"].items():
        _number(rps, f"initial_state.rps.{name}", nonnegative=True)
    launch = vehicle["launch"]
    _keys(launch, {"from_ground", "ground_z_m", "clearance_m"}, {"from_ground", "ground_z_m", "clearance_m"}, "vehicle.launch")
    _boolean(launch["from_ground"], "vehicle.launch.from_ground")
    _number(launch["ground_z_m"], "vehicle.launch.ground_z_m")
    _number(launch["clearance_m"], "vehicle.launch.clearance_m", nonnegative=True)
    if launch["from_ground"]:
        if not config["scene"]["ground"]["enabled"]:
            raise ConfigurationError("Ground launch requires the configured physical ground plane")
        if any(initial["rps"].values()) or np.any(initial["lin_vel"]) or np.any(initial["ang_vel"]):
            raise ConfigurationError("Ground launch requires initially stopped motors and zero initial velocities")
    control = config["control"]
    vectors = {"position_kp", "velocity_kd", "position_ki", "integral_limit_m_s", "attitude_kp", "angular_rate_kd"}
    fields = vectors | {"max_tilt_rad", "max_yaw_rate_rad_s", "max_acceleration_m_s2"}
    _keys(control, fields, fields, "control")
    for field in vectors:
        if np.any(_array(control[field], (3,), f"control.{field}") < 0):
            raise ConfigurationError(f"control.{field} must be nonnegative")
    _number(control["max_tilt_rad"], "max_tilt_rad", positive=True)
    if control["max_tilt_rad"] >= math.pi / 2:
        raise ConfigurationError("max_tilt_rad must be below pi/2 for this upright controller")
    _number(control["max_yaw_rate_rad_s"], "max_yaw_rate_rad_s", positive=True)
    if control["max_acceleration_m_s2"] is not None:
        _number(control["max_acceleration_m_s2"], "max_acceleration_m_s2", positive=True)
    alloc = config["allocation"]
    _keys(alloc, {"weights", "regularization"}, {"weights", "regularization"}, "allocation")
    if np.any(_array(alloc["weights"], (6,), "allocation.weights") <= 0):
        raise ConfigurationError("allocation.weights must be positive")
    _number(alloc["regularization"], "allocation.regularization", nonnegative=True)
    trajectory = config["trajectory"]
    _keys(trajectory, {"kind", "hold", "spiral", "completion"}, {"kind", "hold", "spiral", "completion"}, "trajectory")
    if trajectory["kind"] not in ("hold", "spiral", "helix"):
        raise ConfigurationError("trajectory.kind must be hold, helix, or legacy alias spiral")
    _keys(trajectory["hold"], {"position_w_m", "yaw_rad"}, {"position_w_m", "yaw_rad"}, "trajectory.hold")
    if trajectory["hold"]["position_w_m"] is not None:
        _array(trajectory["hold"]["position_w_m"], (3,), "trajectory.hold.position_w_m")
    if trajectory["hold"]["yaw_rad"] is not None:
        _number(trajectory["hold"]["yaw_rad"], "trajectory.hold.yaw_rad")
    from .trajectories.spiral import validate_spiral_config
    from .mission import validate_completion_config
    validate_spiral_config(trajectory["spiral"])
    validate_completion_config(trajectory["completion"])
    if trajectory["kind"] in ("spiral", "helix"):
        if not launch["from_ground"]:
            raise ConfigurationError("Helix mission requires ground launch; use configs/arl_robot_1_helix.yaml")
        plan = trajectory["spiral"]
        if plan["start_delay_s"] <= 0:
            raise ConfigurationError("Ground helix requires positive start_delay_s for motor spin-up")
        # A null (minimum-time) duration is known only after planning; runtime reset checks it then.
        if plan["takeoff_duration_s"] is not None and plan["spiral_duration_s"] is not None:
            mission_time = plan["start_delay_s"]+plan["takeoff_duration_s"]+plan["spiral_duration_s"]
            if sim["duration_s"] < mission_time+trajectory["completion"]["dwell_time_s"]+2*sim["dt"]:
                raise ConfigurationError("simulation.duration_s must allow the complete helix and measured hover dwell")
    scene = config["scene"]
    _keys(scene, {"ground", "light"}, {"ground", "light"}, "scene")
    for field in ("ground", "light"):
        allowed = {"enabled", "prim_path", "native_overrides"} if field == "ground" else {"enabled", "prim_path", "intensity"}
        _keys(scene[field], allowed, allowed, f"scene.{field}")
        _boolean(scene[field]["enabled"], f"scene.{field}.enabled")
        if not isinstance(scene[field]["prim_path"], str) or not scene[field]["prim_path"].startswith("/World/"):
            raise ConfigurationError(f"scene.{field}.prim_path must be below /World")
    if not isinstance(scene["ground"]["native_overrides"], dict):
        raise ConfigurationError("scene.ground.native_overrides must be a mapping")
    _number(scene["light"]["intensity"], "scene.light.intensity", nonnegative=True)
    log = config["logging"]
    _keys(log, {"directory", "every_n_steps", "flush_every_n_records"}, {"directory", "every_n_steps", "flush_every_n_records"}, "logging")
    if not isinstance(log["directory"], str) or not log["directory"]:
        raise ConfigurationError("logging.directory must be nonempty")
    for field in ("every_n_steps", "flush_every_n_records"):
        _integer(log[field], f"logging.{field}", 1)
    from .disturbances import build_effects
    from .power import validate_power_config
    build_effects(config["effects"])
    validate_power_config(config["power"])


def resolve_asset(config: dict, repo_root: Path = REPO_ROOT) -> Path:
    """Require a local USD; never silently select a different network asset."""
    path = Path(config["vehicle"]["asset_path"]).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigurationError(f"Local ARL USD missing: {path}; run download_assets.sh explicitly")
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
        raise ConfigurationError("The current geometric controller requires all thrust axes parallel to body +Z; choose a different controller for tilted rotors")
    if np.linalg.matrix_rank(allocation[[2, 3, 4, 5], :]) != 4:
        raise ConfigurationError("Actual rotor geometry/directions have rank < 4 for collective/roll/pitch/yaw")
