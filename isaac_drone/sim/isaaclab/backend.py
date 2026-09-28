"""Single-vehicle Isaac Lab 3.0 / PhysX bridge.

Isaac Lab uses xyzw; the application's contracts use wxyz. Native motor
integration functions drive direct rotor-speed states. Individual propeller
forces and reaction torques are submitted per physics tick. Runtime allocation
uses resolved link geometry and the whole-vehicle
CoM. The upstream allocation remains available as a diagnostic reference.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from isaac_drone.core.types import MassProperties, VehicleState, Wrench
from isaac_drone.core.validation import finite_array
from isaac_drone.sim.actuation import NativeRpsActuator, thrust_to_motor_speeds

# Legacy schema slots remain writable even when the native default is None.
_NULLABLE_CONFIG_TYPES = {
    "mass_props": "MassPropertiesCfg",
    "collision_props": "CollisionPropertiesCfg",
    "rigid_props": "RigidBodyPropertiesCfg",
    "articulation_props": "ArticulationRootPropertiesCfg",
    "physics_material": "RigidBodyMaterialCfg",
    "visual_material": "PreviewSurfaceCfg",
    "joint_drive_props": "JointDrivePropertiesCfg",
    "fixed_tendons_props": "FixedTendonPropertiesCfg",
    "spatial_tendons_props": "SpatialTendonPropertiesCfg",
    "physics": "PhysxCfg",
}
_DATA_MAPPINGS = {"variants", "visual_material_bindings", "rps", "joint_pos", "joint_vel"}


def _native_cfg(type_name):
    """Resolve native schema types from known modules, never arbitrary imports."""
    import importlib

    if not isinstance(type_name, str) or not type_name.endswith("Cfg") or not type_name.isidentifier():
        raise ValueError(f"Invalid native config type: {type_name}")
    for module_name in (
        "isaaclab.sim",
        "isaaclab_physx.physics",
        "isaaclab_physx.sim.schemas",
        "isaaclab_physx.sim.spawners.materials",
    ):
        module = importlib.import_module(module_name)
        cls = getattr(module, type_name, None)
        if isinstance(cls, type):
            return cls()
    raise ValueError(f"Unknown native config type: {type_name}")


def _fragment(value, path):
    """Decode explicit {_type: NativeCfg, ...} schema fragments."""
    if not isinstance(value, Mapping) or "_type" not in value:
        raise ValueError(f"Native fragment at {path} requires _type")
    result = _native_cfg(value["_type"])
    override_native(result, {key: val for key, val in value.items() if key != "_type"}, path)
    return result


def override_native(target, values: Mapping, path: str = "") -> None:
    """Apply existing fields recursively; null leaves the native value intact."""
    for key, value in values.items():
        name = f"{path}.{key}" if path else key
        if key.startswith("_") or (key not in target if isinstance(target, dict) else not hasattr(target, key)):
            raise ValueError(f"Unknown native field: {name}")
        if value is None:
            continue
        current = target[key] if isinstance(target, dict) else getattr(target, key)
        replacement = None
        if key in _DATA_MAPPINGS and isinstance(value, Mapping):
            replacement = copy.deepcopy(value)
        elif isinstance(value, Mapping) and "_type" in value:
            replacement = _fragment(value, name)
        elif key in _NULLABLE_CONFIG_TYPES and isinstance(value, list):
            replacement = [_fragment(item, name) for item in value]
        elif (
            key in _NULLABLE_CONFIG_TYPES
            and isinstance(value, Mapping)
            and any(str(k).startswith("/") or k == "" for k in value)
        ):
            replacement = {
                pattern: [_fragment(item, f"{name}.{pattern}") for item in fragments]
                for pattern, fragments in value.items()
            }
        if replacement is not None:
            if isinstance(target, dict):
                target[key] = replacement
            else:
                setattr(target, key, replacement)
        elif isinstance(value, Mapping):
            if current is None:
                type_name = _NULLABLE_CONFIG_TYPES.get(key)
                if type_name is None:
                    raise ValueError(f"Unset native config field {name} needs an explicit _type")
                current = _native_cfg(type_name)
                if isinstance(target, dict):
                    target[key] = current
                else:
                    setattr(target, key, current)
            override_native(current, value, name)
        else:
            value = tuple(value) if isinstance(current, tuple) and isinstance(value, list) else copy.deepcopy(value)
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)


def _reserved(overrides, names, scope):
    conflicts = {key for key in overrides if overrides[key] is not None} & set(names)
    if conflicts:
        raise ValueError(f"{scope} has dedicated fields; remove native overrides for {sorted(conflicts)}")


def make_simulation_cfg(config):
    """Build native configuration lazily, preserving all unmodified defaults."""
    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    values = config["simulation"]
    cfg = SimulationCfg(physics=PhysxCfg())
    overrides = values.get("native_overrides", {})
    _reserved(overrides, ("dt", "device", "gravity", "render_interval", "use_newton_actuators"), "simulation")
    override_native(cfg, overrides)
    cfg.dt = float(values["dt"])
    if not np.isfinite(cfg.dt) or cfg.dt <= 0:
        raise ValueError("simulation.dt must be finite and positive")
    cfg.device = values["device"]
    cfg.gravity = tuple(finite_array(values["gravity"], (3,), "gravity"))
    cfg.render_interval = int(values["render_interval"])
    cfg.use_newton_actuators = False  # ThrusterCfg has no Newton-native execution path.
    return cfg


def make_robot_cfg(config, asset_path):
    """Copy the native ARL model; initial positions retain native root-link semantics.

    Initial linear velocity refers to root-body CoM in world coordinates;
    initial angular velocity is also world-frame, matching native Isaac Lab.
    """
    from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

    values = config["vehicle"]
    if values.get("name", "arl_robot_1") != "arl_robot_1":
        raise ValueError("This backend supports arl_robot_1")
    cfg = copy.deepcopy(ARL_ROBOT_1_CFG)
    overrides = values.get("native_overrides", {})
    _reserved(
        overrides,
        ("actuators", "init_state", "allocation_matrix", "rotor_directions", "prim_path", "class_type"),
        "vehicle",
    )
    if (overrides.get("spawn") or {}).get("usd_path") is not None:
        raise ValueError("Use vehicle.asset_path instead of native_overrides.spawn.usd_path")
    override_native(cfg, overrides)
    cfg.spawn.usd_path = str(asset_path)
    cfg.prim_path = values["prim_path"]
    initial = copy.deepcopy(values.get("initial_state", {}))
    if "quaternion_wxyz" in initial:
        q = finite_array(initial.pop("quaternion_wxyz"), (4,), "initial quaternion")
        if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6):
            raise ValueError("initial quaternion must be normalized")
        initial["rot"] = tuple(q[[1, 2, 3, 0]])
    override_native(cfg.init_state, initial, "initial_state")
    thruster_values = values.get("thrusters", {})
    _reserved(thruster_values, ("dt", "class_type"), "thrusters")
    expected_names = list(ARL_ROBOT_1_CFG.actuators["thrusters"].thruster_names_expr)
    if (
        thruster_values.get("thruster_names_expr") is not None
        and list(thruster_values["thruster_names_expr"]) != expected_names
    ):
        raise ValueError("ARL thruster_names_expr defines the action order and cannot be changed")
    override_native(cfg.actuators["thrusters"], thruster_values, "thrusters")
    cfg.actuators["thrusters"].dt = float(config["simulation"]["dt"])
    if "allocation_matrix" in values:
        cfg.allocation_matrix = finite_array(values["allocation_matrix"], (6, 4), "allocation_matrix").tolist()
    if "rotor_directions" in values:
        directions = finite_array(values["rotor_directions"], (4,), "rotor_directions")
        if not np.isin(directions, [-1, 1]).all():
            raise ValueError("rotor_directions must contain only -1 or +1")
        cfg.rotor_directions = directions.astype(int).tolist()
    return cfg


def apply_usd_overrides(robot_cfg, overrides):
    """Author existing mass properties on the active stage, before sim.reset().

    Overrides use root-relative prim paths. principalAxes is a USD quaternion
    represented as [w, x, y, z]. The stage's source asset files are not saved.
    """
    if not overrides:
        return
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, Sdf

    stage = get_current_stage()
    root = Sdf.Path(robot_cfg.prim_path)
    prepared = []
    for item in overrides:
        if set(item) != {"prim_path", "attribute", "value"}:
            raise ValueError("USD override requires prim_path, attribute, value")
        relative = Sdf.Path(item["prim_path"])
        if relative.IsAbsolutePath() or ".." in item["prim_path"].split("/"):
            raise ValueError("USD override prim_path must remain under the robot")
        path = root.AppendPath(relative)
        if not path.HasPrefix(root) or not path.IsPrimPath():
            raise ValueError("USD override must identify a robot prim")
        prim = stage.GetPrimAtPath(path)
        attr = prim.GetAttribute(item["attribute"]) if prim else None
        if not attr:
            raise ValueError(f"Unknown USD attribute: {path}.{item['attribute']}")
        name, value = item["attribute"], item["value"]
        if name == "physics:mass":
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("USD mass must be positive")
        elif name in ("physics:centerOfMass", "physics:diagonalInertia"):
            vector = finite_array(value, (3,), name)
            if name.endswith("diagonalInertia") and (
                np.min(vector) <= 0 or np.max(vector) > np.sum(vector) - np.max(vector) + 1e-9
            ):
                raise ValueError("USD principal inertia must be physically realizable")
            value = Gf.Vec3f(*vector)
        elif name == "physics:principalAxes":
            q = finite_array(value, (4,), name)
            if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6):
                raise ValueError("USD principalAxes must be a normalized wxyz quaternion")
            value = Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:]))
        else:
            raise ValueError(f"Unsupported USD mass-property override: {name}")
        prepared.append((attr, value))
    for attr, value in prepared:
        if not attr.Set(value):
            raise ValueError(f"Failed to set {attr.GetPath()}")


def _numpy(value):
    tensor = getattr(value, "torch", None)
    if tensor is not None:
        value = tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _rotation_xyzw(quaternion):
    q = finite_array(quaternion, (4,), "native quaternion")
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("Native quaternion has zero norm")
    q /= norm
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _native(values, like):
    """Convert at the bridge boundary, keeping module import simulator-free."""
    if isinstance(like, np.ndarray):
        return np.asarray(values, dtype=like.dtype)
    import torch

    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def _collision_corners_b(stage, root_path, base_name="base_link"):
    """Recompute collision shape extents, retaining scale and instance proxies.

    Bounds exclude visuals and ignore stale authored `extent` values. Rotated
    non-box shapes use conservative bounding corners; this estimates geometric
    clearance, not cooked-shape contact/rest-offset settling.
    """
    from itertools import product

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError("Cannot find spawned robot for ground placement")
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    bases = [p for p in prims if p.GetName() == base_name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bases) != 1:
        raise ValueError("Ground placement requires one unambiguous root rigid body")
    time = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(time)
    base_world = cache.GetLocalToWorldTransform(bases[0])
    base_origin = base_world.ExtractTranslation()
    base_rotation_inverse = base_world.ExtractRotation().GetInverse()
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters) or meters <= 0:
        raise ValueError("Invalid stage length units")
    corners, paths = [], []
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(time):
            continue
        boundable = UsdGeom.Boundable(prim)
        if not boundable:
            raise ValueError(f"Unsupported collision geometry: {prim.GetPath()}")
        # ComputeExtent() and BBoxCache trust the asset's authored extent. ARL's
        # cube has a stale extent, so explicitly recompute from shape parameters.
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, time)
        if extent is None or len(extent) != 2:
            raise ValueError(f"Cannot recompute collision extent: {prim.GetPath()}")
        limits = finite_array(np.asarray(extent), (2, 3), "collision extent")
        if np.any(limits[0] > limits[1]):
            raise ValueError("Invalid collision bounds")
        transform = cache.GetLocalToWorldTransform(prim)
        for point in product(*zip(limits[0], limits[1])):
            world_offset = transform.Transform(Gf.Vec3d(*point)) - base_origin
            # Remove only root rotation/translation, never root scale: physical
            # dimensions must survive native spawn.scale overrides.
            corners.append(np.asarray(base_rotation_inverse.TransformDir(world_offset)) * meters)
        paths.append(str(prim.GetPath()))
    if not corners:
        raise ValueError("No enabled collision geometry for ground placement")
    return finite_array(corners, (len(corners), 3), "collision corners"), paths


def _ground_plane_height_m(stage, root_path):
    """Verify the actual static collision plane, independently of visual mesh Z."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError("Configured ground prim does not exist")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    planes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "Plane" or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if (
                ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get()
            ):
                raise ValueError("Ground launch requires a static collision plane")
            ancestor = ancestor.GetParent()
        axis_attr = prim.GetAttribute("axis")
        axis = axis_attr.Get() if axis_attr else "Z"
        if axis not in ("X", "Y", "Z"):
            raise ValueError("Ground plane has an invalid normal axis")
        transform = cache.GetLocalToWorldTransform(prim)
        tangents = [np.eye(3)[i] for i in range(3) if i != "XYZ".index(axis)]
        first, second = [np.asarray(transform.TransformDir(Gf.Vec3d(*v))) for v in tangents]
        normal = np.cross(first, second)
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length <= 1e-12 or np.linalg.norm(normal[:2]) / length > 1e-6:
            raise ValueError("Ground launch requires a horizontal physical plane")
        height = float(transform.Transform(Gf.Vec3d(0.0))[2]) * meters
        if not np.isfinite(height):
            raise ValueError("Physical ground height is nonfinite")
        planes.append((height, str(prim.GetPath())))
    if len(planes) != 1:
        raise ValueError("Ground launch requires exactly one enabled static physics Plane under scene.ground")
    return planes[0]


class ARLBackend:
    """A rigid assembly of fixed links, with four native ARL motor models."""

    def __init__(self, sim, robot, config):
        self.sim, self.robot, self.config = sim, robot, copy.deepcopy(config)
        # Multirotor replaces the parent's data container after initialization.
        # Reinstall a requested public body ordering on that new container.
        if getattr(robot.cfg, "body_ordering", None) is not None:
            robot._resolve_and_install_ordering_maps()
        if sim is not None and not np.isclose(sim.get_physics_dt(), config["simulation"]["dt"]):
            raise ValueError("Actual simulation dt differs from configured native motor dt")
        if robot.num_instances != 1:
            raise ValueError("ARLBackend requires exactly one vehicle")
        if robot.num_joints != 0:
            raise ValueError("Movable internal joints are unsupported by the rigid-vehicle controller")
        self.thruster_names = list(robot.cfg.actuators["thrusters"].thruster_names_expr)
        native_names = list(robot.thruster_names)
        if len(self.thruster_names) != 4 or set(native_names) != set(self.thruster_names) or len(native_names) != 4:
            raise ValueError("Exactly four named ARL thrusters are required")
        self._canonical_to_native = np.array([native_names.index(n) for n in self.thruster_names])
        self._native_to_canonical = np.argsort(self._canonical_to_native)
        self._root_body_id = list(robot.body_names).index("base_link")
        self._last_tick = None
        self._last_command = np.zeros(4)
        self._last_external = Wrench.zero()
        self._last_total = Wrench.zero()
        self._has_submitted_wrench = False
        self._initialize_speed_actuators()
        self._extract_geometry()
        self._body_physics = self._read_body_physics() if sim is not None else None
        for actuator in robot.actuators.values():
            if not np.isclose(actuator.cfg.dt, config["simulation"]["dt"]):
                raise ValueError("Native motor dt must equal the physics timestep")

    def _extract_geometry(self):
        data = self.robot.data
        root_pos = _numpy(data.root_link_pos_w)[0]
        root_R = _rotation_xyzw(_numpy(data.root_link_quat_w)[0])
        positions = _numpy(data.body_link_pos_w)[0]
        rotations = np.array([_rotation_xyzw(q) for q in _numpy(data.body_link_quat_w)[0]])
        self._link_pos_b = (positions - root_pos) @ root_R
        self._link_rot_b = np.einsum("ij,bjk->bik", root_R.T, rotations)
        if not np.allclose(self._link_pos_b[self._root_body_id], 0, atol=1e-5) or not np.allclose(
            self._link_rot_b[self._root_body_id], np.eye(3), atol=1e-5
        ):
            raise ValueError("base_link is not the articulation root frame")
        masses = _numpy(data.body_mass)[0]
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("Resolved dynamic body masses must be finite and positive")
        body_com = _numpy(data.body_com_pos_b)[0]
        coms_b = self._link_pos_b + np.einsum("bij,bj->bi", self._link_rot_b, body_com)
        com = np.average(coms_b, weights=masses, axis=0)
        # PhysX ArticulationView.get_inertias() is COM-local, despite the core
        # BaseArticulationData docstring saying world-frame. The PhysX data
        # implementation forwards that API unchanged (apart from body order).
        # https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html
        native_inertias = _numpy(data.body_inertia)[0].reshape(-1, 3, 3)
        com_rot = np.array([_rotation_xyzw(q) for q in _numpy(data.body_com_quat_b)[0]])
        inertia = np.zeros((3, 3))
        for m, local_I, link_R, mass_R, r in zip(masses, native_inertias, self._link_rot_b, com_rot, coms_b - com):
            R = link_R @ mass_R
            inertia += R @ local_I @ R.T + m * ((r @ r) * np.eye(3) - np.outer(r, r))
        self._mass = MassProperties(float(masses.sum()), (inertia + inertia.T) / 2, com)
        self._root_com_b = coms_b[self._root_body_id]
        rotor_ids = [list(self.robot.body_names).index(name) for name in self.thruster_names]
        self._rotor_body_ids = rotor_ids
        self.rotor_positions_b = self._link_pos_b[rotor_ids].copy()
        axis = finite_array(self.robot.cfg.thruster_force_direction, (3,), "thruster_force_direction")
        if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6):
            raise ValueError("Native thruster_force_direction must be a unit vector")
        self._rotor_axis_local = axis
        axes = self._link_rot_b[rotor_ids] @ axis
        directions = finite_array(self.robot.cfg.rotor_directions, (4,), "rotor_directions")
        native_ratios = np.empty(4)
        for _name, actuator, indices in self._actuator_groups():
            native_ratios[indices] = float(actuator.cfg.torque_to_thrust_ratio)
        ratios = native_ratios[self._canonical_to_native]
        if not np.isfinite(ratios).all() or np.any(ratios < 0):
            raise ValueError("Native reaction-torque coefficients must be finite and nonnegative")
        self._rotor_torque_ratios = ratios
        moments = np.cross(self.rotor_positions_b - com, axes) + ratios[:, None] * directions[:, None] * axes
        self.allocation_matrix_b = np.concatenate((axes.T, moments.T), axis=0)
        reference = finite_array(self.robot.cfg.allocation_matrix, (6, 4), "reference allocation")
        # Reference is authored about the root origin, so shift it to the same CoM.
        reference_com = reference.copy()
        reference_com[3:] -= np.cross(np.broadcast_to(com, (4, 3)), reference[:3].T).T
        self._allocation_difference = self.allocation_matrix_b - reference_com
        self._reference_allocation = reference
        self._allocation_rank = int(np.linalg.matrix_rank(self.allocation_matrix_b[[2, 3, 4, 5]]))
        self._rotor_directions = directions
        self._geometry_matches_reference = bool(
            np.allclose(
                self._allocation_difference, 0, atol=self.config["vehicle"].get("geometry_tolerance_m", 1e-4), rtol=0
            )
        )

    def _read_body_physics(self):
        """Read effective USD flags after the native spawner has applied overrides."""
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Usd, UsdPhysics

        stage = get_current_stage()
        root = stage.GetPrimAtPath(self.robot.cfg.prim_path)
        if not root:
            raise ValueError("Cannot audit spawned robot USD prim")
        matches = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in self.robot.body_names:
                if prim.GetName() in matches:
                    raise ValueError(f"Ambiguous USD rigid-body name: {prim.GetName()}")
                rigid = UsdPhysics.RigidBodyAPI(prim)
                gravity_attr = prim.GetAttribute("physxRigidBody:disableGravity")
                disable_gravity = gravity_attr.Get() if gravity_attr else False
                matches[prim.GetName()] = {
                    "path": str(prim.GetPath()),
                    "rigid_body_enabled": bool(rigid.GetRigidBodyEnabledAttr().Get()),
                    "kinematic_enabled": bool(rigid.GetKinematicEnabledAttr().Get()),
                    "disable_gravity": bool(disable_gravity),
                }
        if set(matches) != set(self.robot.body_names):
            raise ValueError("Could not match every PhysX body to a spawned USD rigid-body prim")
        return matches

    def assert_control_compatible(self):
        """Check actual spawned flags before using gravity-compensating flight control."""
        if getattr(self.robot, "is_fixed_base", False):
            raise ValueError("Flight control requires a free base; the actual articulation is fixed")
        view = getattr(self.robot, "root_view", None)
        meta = getattr(view, "shared_metatype", None)
        if bool(getattr(meta, "fixed_base", False)):
            raise ValueError("PhysX reports a fixed-base articulation")
        flags = self._read_body_physics() if self.sim is not None else self._body_physics
        if flags is None:
            raise ValueError("Actual spawned body physics flags have not been audited")
        self._body_physics = flags
        for name, body in flags.items():
            if not body["rigid_body_enabled"] or body["kinematic_enabled"] or body["disable_gravity"]:
                raise ValueError(f"Flight control requires a dynamic gravity-enabled body: {name}: {body}")
        if self._allocation_rank < 4:
            raise ValueError("Flight allocation is rank deficient")

    def mass_properties(self):
        return self._mass

    def read_state(self, time_s):
        data = self.robot.data
        q = _numpy(data.root_link_quat_w)[0]
        R = _rotation_xyzw(q)
        omega_w = _numpy(data.root_link_ang_vel_w)[0]
        offset_w = R @ self._mass.com_b
        return VehicleState(
            time_s,
            _numpy(data.root_link_pos_w)[0] + offset_w,
            q[[3, 0, 1, 2]],
            _numpy(data.root_link_lin_vel_w)[0] + np.cross(omega_w, offset_w),
            R.T @ omega_w,
        )

    def _propeller_wrench_arrays(self, thrusts_n, external_wrench):
        """Per-propeller local forces at rotor origins; base gets only disturbance.

        The physics engine transmits rotor forces through the fixed joints.
        No controller-generated force or torque is applied to the vehicle base.
        """
        thrusts = finite_array(thrusts_n, (4,), "actual rotor thrust")
        force = np.zeros((1, self.robot.num_bodies, 3), dtype=np.float32)
        torque = np.zeros_like(force)
        positions = np.zeros_like(force)
        rotor_ids = self.robot.map_body_ids_to_backend(self._rotor_body_ids)
        for motor, body_id in enumerate(rotor_ids):
            force[0, body_id] = thrusts[motor] * self._rotor_axis_local
            torque[0, body_id] = (
                self._rotor_directions[motor]
                * self._rotor_torque_ratios[motor]
                * thrusts[motor]
                * self._rotor_axis_local
            )
            # Explicit local (0,0,0) is the rotor thrust point, not rotor COM.
            # PhysX includes the force's moment arm about that body's COM.
        root_id = self.robot.map_body_ids_to_backend([self._root_body_id])[0]
        if root_id in rotor_ids:
            raise ValueError("A propeller cannot share the vehicle root body")
        force[0, root_id] = external_wrench.force_b
        torque[0, root_id] = external_wrench.torque_b + np.cross(
            self._mass.com_b - self._root_com_b, external_wrench.force_b
        )
        positions[0, root_id] = self._root_com_b
        return force, torque, positions

    def _submit_body_wrenches(self, force, torque, positions):
        """Submit rotor-local forces and external disturbance in one PhysX call."""
        import torch
        import warp as wp

        tensors = [torch.as_tensor(array, device=self.robot.device) for array in (force, torque, positions)]
        self.robot.root_view.apply_forces_and_torques_at_position(
            force_data=wp.from_torch(tensors[0].reshape(-1), dtype=wp.float32),
            torque_data=wp.from_torch(tensors[1].reshape(-1), dtype=wp.float32),
            position_data=wp.from_torch(tensors[2].reshape(-1), dtype=wp.float32),
            indices=self.robot._ALL_INDICES,
            is_global=False,
        )

    def _actuator_groups(self):
        """Resolve a complete non-overlapping mapping of native motor groups."""
        count = len(self.robot.thruster_names)
        occupied = np.zeros(count, dtype=bool)
        groups = []
        for name, actuator in self.robot.actuators.items():
            selection = actuator.thruster_indices
            if isinstance(selection, slice):
                indices = np.arange(count)[selection]
            else:
                raw = _numpy(selection)
                if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
                    raise ValueError(f"Actuator {name} has invalid thruster indices")
                indices = raw.astype(int)
            if (
                not len(indices)
                or np.any(indices < 0)
                or np.any(indices >= count)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(f"Actuator {name} has invalid or duplicate thruster indices")
            if occupied[indices].any():
                raise ValueError(f"Actuator {name} overlaps another thruster group")
            occupied[indices] = True
            groups.append((name, actuator, indices))
        if not occupied.all():
            raise ValueError("Native actuator groups do not cover every thruster")
        return groups

    def _initialize_speed_actuators(self, use_defaults=False):
        self._speed_actuators = {}
        for name, actuator, indices in self._actuator_groups():
            if use_defaults:
                initial = self.robot.data.default_thruster_rps[:, indices]
            else:
                kf = finite_array(_numpy(actuator.thrust_const), (1, len(indices)), f"{name}.thrust_const")
                if np.any(kf <= 0):
                    raise ValueError("Sampled motor coefficients must be positive")
                initial = np.sqrt(np.maximum(_numpy(actuator.curr_thrust) / kf, 0))
            self._speed_actuators[name] = NativeRpsActuator(actuator, initial)

    def motor_parameters(self):
        """Return canonical sampled per-motor parameters for control allocation."""
        count = len(self.thruster_names)
        kf, minimum, maximum, current = (np.empty(count) for _ in range(4))
        for name, actuator, indices in self._actuator_groups():
            adapter = self._speed_actuators[name]
            group_kf, _, _ = adapter.parameters()
            kf[indices] = group_kf[0]
            minimum[indices], maximum[indices] = actuator.cfg.thrust_range
            current[indices] = finite_array(_numpy(adapter.rps), (1, len(indices)), "actual motor rps")[0]
        order = self._canonical_to_native
        return {
            "thruster_names": list(self.thruster_names),
            "sampled_kf": kf[order],
            "thrust_min_n": minimum[order],
            "thrust_max_n": maximum[order],
            "rps_min": np.sqrt(minimum[order] / kf[order]),
            "rps_max": np.sqrt(maximum[order] / kf[order]),
            "current_rps": current[order],
        }

    def thrust_to_motor_speeds(self, thrust_n):
        parameters = self.motor_parameters()
        return thrust_to_motor_speeds(
            thrust_n, parameters["sampled_kf"], parameters["thrust_min_n"], parameters["thrust_max_n"]
        )

    def apply_motor_speeds(self, target_rps, external_wrench):
        """Advance native RPS dynamics once and apply forces to the four prop bodies."""
        command = finite_array(target_rps, (4,), "target motor rps")
        if np.any(command < 0):
            raise ValueError("Motor speed commands must be nonnegative magnitudes")
        if self._allocation_rank < 4:
            raise RuntimeError(
                "Rotor geometry/directions cannot independently control collective thrust and three moments"
            )
        tick = getattr(self.robot.data, "_sim_timestamp", None)
        if tick is not None and tick == self._last_tick:
            raise RuntimeError("apply_motor_speeds() called twice without advancing physics and robot.update(dt)")
        for name in ("_permanent_wrench_composer", "_instantaneous_wrench_composer"):
            if getattr(getattr(self.robot, name, None), "active", False):
                raise RuntimeError("Pass all external forces through external_wrench, not native composers")
        native_command = command[self._native_to_canonical][None, :]
        for name, actuator, indices in self._actuator_groups():
            adapter = self._speed_actuators[name]
            adapter.step(native_command[:, indices])
            target_thrust = actuator.thrust_const * adapter.target_rps**2
            self.robot.data.thrust_target[:, indices] = target_thrust
            self.robot.data.computed_thrust[:, indices] = actuator.curr_thrust
            self.robot.data.applied_thrust[:, indices] = actuator.applied_thrust
        applied = _numpy(self.robot.data.applied_thrust)[0][self._canonical_to_native]
        arrays = self._propeller_wrench_arrays(applied, external_wrench)
        self._submit_body_wrenches(*arrays)
        self._last_tick = tick
        self._last_command = _numpy(self.robot.data.thrust_target)[0][self._canonical_to_native].copy()
        self._last_external = external_wrench
        # Aggregate only for diagnostics: this wrench is never sent to root.
        self._last_total = Wrench.from_vector(self.allocation_matrix_b @ applied) + external_wrench
        self._has_submitted_wrench = True

    def ground_start_position(self, ground_z_m=0.0, clearance_m=0.001, *, stage=None):
        """Compute a root-link reset position above actual collision bounds.

        Uses the configured reset attitude/XY so the returned position matches
        reset(initial_position_w=...). This method never writes state or USD.
        """
        ground, clearance = float(ground_z_m), float(clearance_m)
        if not np.isfinite(ground) or not np.isfinite(clearance) or clearance < 0:
            raise ValueError("Ground height must be finite and clearance nonnegative")
        if stage is None:
            from isaaclab.sim.utils.stage import get_current_stage

            stage = get_current_stage()
        ground_path = self.config.get("scene", {}).get("ground", {}).get("prim_path")
        physical_ground_path = None
        if ground_path:
            actual_ground, physical_ground_path = _ground_plane_height_m(stage, ground_path)
            if not np.isclose(actual_ground, ground, rtol=0.0, atol=1e-6):
                raise ValueError(f"Physical ground plane height {actual_ground} differs from configured {ground}")
            ground = actual_ground
        corners, paths = _collision_corners_b(stage, self.robot.cfg.prim_path)
        initial = self.config["vehicle"]["initial_state"]
        q = finite_array(initial["quaternion_wxyz"], (4,), "initial quaternion")
        rotation = _rotation_xyzw(q[[1, 2, 3, 0]])
        lowest = float(np.min((corners @ rotation.T)[:, 2]))
        position = finite_array(initial["pos"], (3,), "initial root-link position").copy()
        position[2] = ground + clearance - lowest
        self._ground_start_audit = {
            "source": "enabled_usd_collision_shape_extents_recomputed_with_plugins_including_instance_proxies",
            "collision_prim_paths": paths,
            "minimum_collision_z_relative_to_root_m": lowest,
            "ground_z_m": ground,
            "physical_ground_prim_path": physical_ground_path,
            "clearance_m": clearance,
            "root_position_w_m": position.tolist(),
            "vehicle_com_position_w_m": (position + rotation @ self._mass.com_b).tolist(),
            "limitation": "conservative_geometry_bounds; does_not_predict_cooked_contact_or_rest_offset_settling",
        }
        return position

    def reset(self, initial_position_w=None):
        # Upstream constructs actuators before filling default RPS, taking an
        # advanced-index copy of zeros. Rebind that reset input to the parsed RPS.
        for actuator in self.robot.actuators.values():
            defaults = self.robot.data.default_thruster_rps[:, actuator.thruster_indices]
            actuator._init_thruster_rps = defaults.clone() if hasattr(defaults, "clone") else defaults.copy()
        # Avoid the native None path: Multirotor.reset then uses Warp
        # _ALL_INDICES to index torch thrust_target. An explicit list follows
        # its supported torch-index conversion while still resetting motors.
        self.robot.reset(env_ids=[0])
        initial = self.config["vehicle"]["initial_state"]
        q = finite_array(initial["quaternion_wxyz"], (4,), "initial quaternion")
        position = finite_array(
            initial["pos"] if initial_position_w is None else initial_position_w, (3,), "initial root-link position"
        )
        omega_w = finite_array(initial.get("ang_vel", [0, 0, 0]), (3,), "initial world angular velocity")
        velocity_root_com = finite_array(initial.get("lin_vel", [0, 0, 0]), (3,), "initial root-body CoM velocity")
        like = self.robot.data.thrust_target
        self.robot.write_root_pose_to_sim_index(root_pose=_native(np.r_[position, q[[1, 2, 3, 0]]][None, :], like))
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=_native(np.r_[velocity_root_com, omega_w][None, :], like)
        )
        # Zero-RPS initialization remains zero; the running thrust minimum is
        # a nonzero target bound, not a force applied to stopped rotors.
        self._initialize_speed_actuators(use_defaults=True)
        for actuator in self.robot.actuators.values():
            ids = actuator.thruster_indices
            actual = actuator.curr_thrust
            self.robot.data.thrust_target[:, ids] = actual
            self.robot.data.computed_thrust[:, ids] = actual
            self.robot.data.applied_thrust[:, ids] = actual
        self.robot.update(0.0)
        self._last_tick = None
        self._last_command = _numpy(self.robot.data.thrust_target)[0][self._canonical_to_native].copy()
        self._last_external, self._last_total = Wrench.zero(), Wrench.zero()
        self._has_submitted_wrench = False

    def _motor_telemetry(self):
        """Report direct native-integrator RPS state, not spinning-joint measurements."""
        parameters = self.motor_parameters()
        native_target, native_raw = np.empty(4), np.empty(4)
        groups = {}
        for name, actuator, indices in self._actuator_groups():
            adapter = self._speed_actuators[name]
            native_target[indices] = _numpy(adapter.target_rps)[0]
            native_raw[indices] = _numpy(adapter.raw_target_rps)[0]
            groups[name] = {
                key: finite_array(_numpy(getattr(actuator, key)), (1, len(indices)), f"{name}.{key}")[0].tolist()
                for key in ("tau_inc_s", "tau_dec_s", "thrust_const", "curr_thrust")
            }
            groups[name]["thruster_indices"] = indices.tolist()
            groups[name]["thruster_names"] = [self.robot.thruster_names[index] for index in indices]
        order = self._canonical_to_native
        speeds = parameters["current_rps"]
        kf = parameters["sampled_kf"]
        target = finite_array(native_target[order], (4,), "limited motor speed target")
        return {
            "motor_speed_rps": speeds.tolist(),
            "motor_speed_rpm": finite_array(speeds * 60.0, (4,), "motor rpm").tolist(),
            "motor_speed_rad_s": finite_array(speeds * 2 * np.pi, (4,), "motor rad/s").tolist(),
            "motor_speed_source": "direct_rps_state_integrated_with_native_motor_rate_and_rk4_or_euler",
            "motor_speed_is_magnitude": True,
            "motor_speed_is_encoder_measurement": False,
            "motor_state_thrust_n": (kf * speeds**2).tolist(),
            "motor_state_thrust_source": "sampled_kf_times_direct_actual_rps_squared",
            "motor_thrust_coefficient_n_per_rps2": kf.tolist(),
            "native_clipped_thrust_target_n": (kf * target**2).tolist(),
            "requested_motor_speed_rps": native_raw[order].tolist(),
            "commanded_motor_speed_rps": target.tolist(),
            "commanded_motor_speed_source": (
                "rps_command_with_native_running_limits_and_explicit_zero_shutdown_before_motor_lag"
            ),
            "motor_shutdown_semantics": "zero_target_spins_down_with_native_fall_dynamics; "
            "nonzero_target_retains_native_running_minimum",
            "actuators": groups,
        }

    def _native_acceleration_telemetry(self):
        """Mass-weighted solver acceleration at the whole-vehicle CoM.

        Read only after a completed physics step following apply(). The native
        values already include gravity; unlike IMU specific force, no gravity
        bias is added or removed. Reset/pre-integration values are unavailable.
        """
        result = {
            "linear_acceleration_com_w_m_s2": None,
            "linear_acceleration_source": "unavailable_before_completed_physics_step_after_apply",
        }
        timestamp = getattr(self.robot.data, "_sim_timestamp", None)
        if not self._has_submitted_wrench or self._last_tick is None or timestamp is None:
            return result
        if not np.isfinite(timestamp) or timestamp <= self._last_tick:
            return result
        try:
            # Each property getter is evaluated once: PhysX reads are lazy.
            native_acceleration = self.robot.data.body_com_lin_acc_w
            native_mass = self.robot.data.body_mass
        except (AttributeError, NotImplementedError, RuntimeError):
            result["linear_acceleration_source"] = "unavailable_native_body_com_acceleration_interface"
            return result
        try:
            count = self.robot.num_bodies
            acceleration = finite_array(_numpy(native_acceleration), (1, count, 3), "native body COM acceleration")[0]
            masses = finite_array(_numpy(native_mass), (1, count), "resolved body mass")[0]
            if np.any(masses <= 0):
                raise ValueError("Native body masses must be positive")
            weighted = finite_array(np.average(acceleration, axis=0, weights=masses), (3,), "vehicle COM acceleration")
        except (TypeError, ValueError, RuntimeError):
            result["linear_acceleration_source"] = "unavailable_invalid_native_body_acceleration_or_mass"
            return result
        result["linear_acceleration_com_w_m_s2"] = weighted.tolist()
        result["linear_acceleration_source"] = (
            "physx_mass_weighted_body_com_lin_acc_w_world_coordinate_acceleration_including_gravity"
        )
        return result

    def telemetry(self):
        motors = self._motor_telemetry()
        applied = finite_array(
            _numpy(self.robot.data.applied_thrust), (1, len(self.thruster_names)), "native applied thrust"
        )[0][self._canonical_to_native]
        motor_wrench = Wrench.from_vector(self.allocation_matrix_b @ applied)
        return {
            "mass_kg": self._mass.mass_kg,
            "com_b": self._mass.com_b.tolist(),
            "inertia_com_b": self._mass.inertia_com_b.tolist(),
            "thruster_names": self.thruster_names,
            "native_thruster_names": list(self.robot.thruster_names),
            "rotor_positions_b": self.rotor_positions_b.tolist(),
            "allocation_matrix_b": self.allocation_matrix_b.tolist(),
            "reference_allocation_matrix": self._reference_allocation.tolist(),
            "allocation_difference_at_com": self._allocation_difference.tolist(),
            "geometry_matches_reference": self._geometry_matches_reference,
            "collective_roll_pitch_yaw_rank": self._allocation_rank,
            "rotor_directions": self._rotor_directions.tolist(),
            "rotor_direction_source": self.config["vehicle"].get("rotor_direction_source", "native_reference"),
            "mass_property_source": "PhysX resolved public body mass/inertia/COM/pose buffers; principal-axis "
            "rotation and parallel-axis theorem",
            "internal_movable_joint_count": int(self.robot.num_joints),
            "fixed_base": bool(getattr(self.robot, "is_fixed_base", False)),
            "ground_start": copy.deepcopy(getattr(self, "_ground_start_audit", None)),
            "body_physics_flags": copy.deepcopy(self._body_physics),
            "force_submission": "Per-propeller local thrust and reaction torque at rotor origins; base_link "
            "receives only explicit external disturbance",
            "commanded_thrust_n": self._last_command.tolist(),
            "applied_thrust_n": applied.tolist(),
            "applied_thrust_source": "sampled_kf_times_direct_integrated_rps_squared_applied_to_individual_prop_bodies",
            "motor_wrench_b": motor_wrench.vector.tolist(),
            "motor_wrench_source": (
                "diagnostic_sum_of_individual_propeller_forces_and_moments_about_vehicle_com_not_applied_to_root"
            ),
            "has_submitted_wrench": self._has_submitted_wrench,
            "external_wrench_b": self._last_external.vector.tolist(),
            "total_wrench_b": self._last_total.vector.tolist(),
            "applied_force_b_n": self._last_total.force_b.tolist(),
            "applied_torque_b_nm": self._last_total.torque_b.tolist(),
            "applied_wrench_source": (
                "last_submitted_motor_plus_explicit_external_wrench_about_vehicle_com_"
                "excludes_gravity_contact_and_native_damping"
            ),
            **motors,
            **self._native_acceleration_telemetry(),
        }
