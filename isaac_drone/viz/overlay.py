"""Viewport overlays: the reference path, the flown CoM path and the current setpoint.

The overlays are purely visual USD geometry. No physics, collision or mass API
is applied, so they cannot interact with the simulated vehicle. The NumPy
helpers import without Isaac Sim; USD code imports pxr only when used.
"""
from __future__ import annotations

import numpy as np

from isaac_drone.core.validation import finite_array

TARGET_COLOR = (0.10, 0.85, 0.30)
FLOWN_COLOR = (1.00, 0.35, 0.05)


def reference_path_points(trajectory, start_s: float, end_s: float, step_s: float = 0.005) -> np.ndarray:
    """Sample reference CoM positions over [start_s, end_s] at a fixed time step."""
    if not (np.isfinite(start_s) and np.isfinite(end_s) and np.isfinite(step_s)) or step_s <= 0 or end_s < start_s:
        raise ValueError("reference path needs finite start <= end and a positive step")
    times = np.append(np.arange(start_s, end_s, step_s), end_s)
    return np.array([trajectory.sample(float(time)).position_w for time in times], dtype=float)


def resample_by_arc_length(points, spacing_m: float) -> np.ndarray:
    """Return points evenly spaced along the polyline; stationary samples are dropped."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("path points must be a nonempty finite (N, 3) array")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("arc-length spacing must be positive")
    moving = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]
    points = points[moving]
    if len(points) < 2:
        return points
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    grid = np.linspace(0.0, arc[-1], max(2, int(np.ceil(arc[-1]/spacing_m))+1))
    return np.column_stack([np.interp(grid, arc, points[:, axis]) for axis in range(3)])


def dash_segments(points, dash_m: float, gap_m: float, spacing_m: float = 0.005) -> list[np.ndarray]:
    """Split a path into dashes of about dash_m separated by gaps of about gap_m."""
    if not all(np.isfinite(value) and value > 0 for value in (dash_m, gap_m)):
        raise ValueError("dash and gap lengths must be positive")
    even = resample_by_arc_length(points, spacing_m)
    dash, gap = max(1, round(dash_m/spacing_m)), max(1, round(gap_m/spacing_m))
    segments = [even[index:index+dash+1] for index in range(0, len(even)-1, dash+gap)]
    return [segment for segment in segments if len(segment) >= 2]


class PathTrail:
    """Growing flown path; a point is kept once it is min_spacing_m from the last kept point."""

    def __init__(self, min_spacing_m: float = 0.005, capacity: int = 4096):
        if not np.isfinite(min_spacing_m) or min_spacing_m < 0:
            raise ValueError("trail spacing must be finite and nonnegative")
        self.min_spacing_m = float(min_spacing_m)
        self._buffer = np.empty((max(2, int(capacity)), 3))
        self._count = 0

    def add(self, position_w) -> bool:
        position = finite_array(position_w, (3,), "trail position")
        if self._count and np.linalg.norm(position-self._buffer[self._count-1]) < self.min_spacing_m:
            return False
        if self._count == len(self._buffer):
            self._buffer = np.concatenate([self._buffer, np.empty_like(self._buffer)])
        self._buffer[self._count] = position
        self._count += 1
        return True

    @property
    def points(self) -> np.ndarray:
        return self._buffer[:self._count].copy()


class PathOverlay:
    """USD overlay: dashed reference path, continuous flown path, current setpoint sphere.

    The reference is dashed so the flown path stays visible where tracking
    error is smaller than the line width.
    """

    def __init__(self, stage, root_path: str = "/World/Visuals", *, target_color=TARGET_COLOR,
                 flown_color=FLOWN_COLOR, target_width_m=0.02, flown_width_m=0.012,
                 setpoint_radius_m=0.035):
        from pxr import UsdGeom

        self._stage = stage
        UsdGeom.Scope.Define(stage, root_path)
        target_material = self._material(f"{root_path}/Looks/Target", target_color)
        flown_material = self._material(f"{root_path}/Looks/Flown", flown_color)
        self._target = self._curves(f"{root_path}/TargetPath", target_color, target_width_m, target_material)
        self._flown = self._curves(f"{root_path}/FlownPath", flown_color, flown_width_m, flown_material)
        self._target_width, self._flown_width = float(target_width_m), float(flown_width_m)
        sphere = UsdGeom.Sphere.Define(stage, f"{root_path}/Setpoint")
        sphere.CreateRadiusAttr(float(setpoint_radius_m))
        radius = float(setpoint_radius_m)
        sphere.CreateExtentAttr([(-radius,)*3, (radius,)*3])
        sphere.CreateDisplayColorPrimvar().Set([tuple(map(float, target_color))])
        self._bind(sphere.GetPrim(), target_material)
        self._setpoint = sphere.AddTranslateOp()

    def _material(self, path, color):
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(self._stage, path)
        shader = UsdShade.Shader.Define(self._stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        rgb = Gf.Vec3f(*map(float, color))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(rgb*0.5)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    @staticmethod
    def _bind(prim, material):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def _curves(self, path, color, width, material):
        from pxr import UsdGeom

        curves = UsdGeom.BasisCurves.Define(self._stage, path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreateWidthsAttr([float(width)])
        curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([tuple(map(float, color))])
        curves.CreateCurveVertexCountsAttr([])
        curves.CreatePointsAttr([])
        curves.CreateExtentAttr([(0.0,)*3, (0.0,)*3])
        self._bind(curves.GetPrim(), material)
        return curves

    @staticmethod
    def _write_curves(curves, segments, width):
        from pxr import Vt

        segments = [np.asarray(segment, dtype=np.float32) for segment in segments if len(segment) >= 2]
        if segments:
            points = np.concatenate(segments)
            lower, upper = points.min(axis=0)-width/2, points.max(axis=0)+width/2
            extent = [tuple(map(float, lower)), tuple(map(float, upper))]
            curves.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        else:
            extent = [(0.0,)*3, (0.0,)*3]
            curves.GetPointsAttr().Set([])
        curves.GetCurveVertexCountsAttr().Set([len(segment) for segment in segments])
        curves.GetExtentAttr().Set(extent)

    def set_reference(self, points, *, dash_m=0.08, gap_m=0.05) -> None:
        from pxr import Sdf

        with Sdf.ChangeBlock():
            self._write_curves(self._target, dash_segments(points, dash_m, gap_m), self._target_width)

    def update(self, flown_points, setpoint_position_w) -> None:
        from pxr import Gf, Sdf

        setpoint = finite_array(setpoint_position_w, (3,), "setpoint position")
        with Sdf.ChangeBlock():
            self._write_curves(self._flown, [flown_points], self._flown_width)
            self._setpoint.Set(Gf.Vec3d(*map(float, setpoint)))
