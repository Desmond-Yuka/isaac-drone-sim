"""Inspect authored ARL USD data without launching Isaac Sim (``python -m isaac_drone audit-usd``).

Requires the optional OpenUSD Python bindings (``usd-core``). Reports source
USD values, not PhysX-resolved mass properties; automatic CoM sentinels remain
unresolved. The current YAML supplies rotor names, reaction-torque signs and
coefficient. Native spawn overrides are reported but are not simulated here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _numbers(value):
    return np.asarray(value, dtype=float).tolist()


def inspect(config: dict, asset_path: Path) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not asset_path.is_file():
        raise ValueError(f"Local asset does not exist: {asset_path}")
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {asset_path}")
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("USD has no default prim")
    base = stage.GetPrimAtPath(root.GetPath().AppendChild("base_link"))
    if not base:
        raise ValueError("ARL base_link is missing")
    cache = UsdGeom.XformCache()
    root_inverse = cache.GetLocalToWorldTransform(base).GetInverse()
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    kilograms = UsdPhysics.GetStageKilogramsPerUnit(stage)
    bodies, joints, transforms = [], [], {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            transform = cache.GetLocalToWorldTransform(prim) * root_inverse
            transforms[prim.GetName()] = transform
            mass = UsdPhysics.MassAPI(prim)
            center = np.asarray(mass.GetCenterOfMassAttr().Get(), dtype=float)
            center_resolved = bool(np.isfinite(center).all())
            axes = mass.GetPrincipalAxesAttr().Get()
            q = [float(axes.GetReal()), *_numbers(axes.GetImaginary())]
            authored_mass = mass.GetMassAttr().Get()
            diagonal = np.asarray(mass.GetDiagonalInertiaAttr().Get(), dtype=float)
            bodies.append(
                {
                    "path": str(prim.GetPath()),
                    "name": prim.GetName(),
                    "position_relative_to_base_link_m": (np.asarray(transform.ExtractTranslation()) * meters).tolist(),
                    "authored_mass_kg": float(authored_mass) * kilograms,
                    "mass_is_authored": mass.GetMassAttr().HasAuthoredValueOpinion(),
                    "center_of_mass_local_m": (center * meters).tolist() if center_resolved else None,
                    "center_of_mass_status": "explicit_finite"
                    if center_resolved
                    else "USD_automatic_sentinel_requires_PhysX_resolution",
                    "center_of_mass_is_authored": mass.GetCenterOfMassAttr().HasAuthoredValueOpinion(),
                    "authored_diagonal_inertia_kg_m2": (diagonal * kilograms * meters**2).tolist(),
                    "principal_axes_wxyz": q,
                    "density_kg_m3": float(mass.GetDensityAttr().Get()) * kilograms / meters**3,
                }
            )
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "enabled": joint.GetJointEnabledAttr().Get(),
                    "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                    "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()],
                    "fixed": prim.IsA(UsdPhysics.FixedJoint),
                }
            )
    vehicle = config["vehicle"]
    rotor_names = vehicle["thrusters"]["thruster_names_expr"]
    directions = np.asarray(vehicle["rotor_directions"], dtype=float)
    coefficient = float(vehicle["thrusters"]["torque_to_thrust_ratio"])
    local_axis = np.asarray(vehicle.get("native_overrides", {}).get("thruster_force_direction", [0, 0, 1]), dtype=float)
    positions, axes = [], []
    for name in rotor_names:
        if name not in transforms:
            raise ValueError(f"Configured rotor is not a rigid body in USD: {name}")
        transform = transforms[name]
        positions.append(np.asarray(transform.ExtractTranslation(), dtype=float) * meters)
        axis = np.asarray(transform.TransformDir(Gf.Vec3d(*local_axis)), dtype=float)
        axes.append(axis / np.linalg.norm(axis))
    positions, axes = np.asarray(positions), np.asarray(axes)
    allocation_origin = np.concatenate(
        (axes.T, (np.cross(positions, axes) + coefficient * directions[:, None] * axes).T)
    )
    reference = np.asarray(vehicle["allocation_matrix"], dtype=float)
    layers = []
    for layer in stage.GetUsedLayers():
        filename = layer.realPath
        path = Path(filename) if filename else None
        layers.append(
            {
                "identifier": layer.identifier,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
                "local_file": bool(path and path.is_file()),
            }
        )
    authored_masses = [body["authored_mass_kg"] for body in bodies]
    return {
        "inspection_kind": "offline_authored_USD_not_PhysX_simulation",
        "asset_path": str(asset_path.resolve()),
        "default_prim": str(root.GetPath()),
        "meters_per_unit": meters,
        "kilograms_per_unit": kilograms,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "layers": sorted(layers, key=lambda x: x["identifier"]),
        "bodies": bodies,
        "joints": joints,
        "authored_positive_body_mass_sum_kg": sum(authored_masses) if all(m > 0 for m in authored_masses) else None,
        "all_enabled_internal_joints_fixed": all(j["fixed"] for j in joints if j["enabled"]),
        "aggregate_com_b": None,
        "aggregate_inertia_com_b": None,
        "mass_property_limit": "CoM auto-sentinels require simulation collision mass resolution. No zero CoM or "
        "aggregate inertia is assumed.",
        "rotor_names": rotor_names,
        "rotor_positions_b_m": positions.tolist(),
        "rotor_axes_b": axes.tolist(),
        "configured_reaction_torque_signs": directions.tolist(),
        "rotor_direction_source": vehicle["rotor_direction_source"],
        "rotor_direction_note": "Signs come from configuration, not USD or real-hardware measurements.",
        "allocation_matrix_about_root_origin": allocation_origin.tolist(),
        "collective_roll_pitch_yaw_rank_about_root_origin": int(np.linalg.matrix_rank(allocation_origin[[2, 3, 4, 5]])),
        "upstream_reference_allocation": reference.tolist(),
        "difference_from_reference_about_root_origin": (allocation_origin - reference).tolist(),
        "native_spawn_overrides_not_applied": vehicle.get("native_overrides", {}),
        "usd_overrides_not_applied": vehicle.get("usd_overrides", []),
        "runtime_requirement": "Backend must resolve actual PhysX mass, COM, principal axes, and inertia after "
        "spawning with overrides.",
    }
