"""Path overlay geometry: reference sampling, dashes and the flown trail."""

import numpy as np
import pytest

from isaac_drone.core.types import VehicleState
from isaac_drone.trajectories import TRAJECTORIES
from isaac_drone.viz.overlay import PathOverlay, PathTrail, dash_segments, resample_by_arc_length


def helix():
    trajectory = TRAJECTORIES.build(
        {
            "kind": "helix",
            "takeoff_height_m": 1.0,
            "takeoff_duration_s": 4.0,
            "radius_m": 1.0,
            "turns": 2.0,
            "climb_height_m": 3.0,
            "helix_duration_s": 24.0,
            "initial_phase_rad": 0.0,
            "yaw_mode": "fixed",
            "yaw_offset_rad": 0.0,
            "rest_speed_tolerance_m_s": 1e-3,
            "rest_angular_speed_tolerance_rad_s": 1e-3,
        },
        "trajectory",
    )
    trajectory.reset(VehicleState(0.0, np.array([0.0, 0.0, 0.1]), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3)))
    return trajectory


def test_reference_path_covers_takeoff_and_helix_to_the_endpoint():
    trajectory = helix()
    points = trajectory.path_points()
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(points[-1], trajectory.endpoint_position_w, atol=1e-12)
    center = np.array([-1.0, 0.0])
    helical = points[:, 2] > 1.1 + 1e-6
    np.testing.assert_allclose(np.linalg.norm(points[helical, :2] - center, axis=1), 1.0, atol=1e-9)
    assert points[:, 2].max() == pytest.approx(4.1)


def test_arc_length_resampling_is_even_and_drops_stationary_samples():
    points = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 2, 0]], dtype=float)
    even = resample_by_arc_length(points, 0.1)
    steps = np.linalg.norm(np.diff(even, axis=0), axis=1)
    np.testing.assert_allclose(steps, 0.1, atol=1e-12)
    np.testing.assert_allclose(even[[0, -1]], [[0, 0, 0], [1, 2, 0]])
    assert len(resample_by_arc_length(np.zeros((5, 3)), 0.1)) == 1


def test_dashes_alternate_with_gaps_along_the_path():
    line = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    dashes = dash_segments(line, dash_m=0.08, gap_m=0.05, spacing_m=0.01)
    starts = [segment[0, 0] for segment in dashes]
    np.testing.assert_allclose(np.diff(starts), 0.13, atol=1e-12)
    lengths = [np.linalg.norm(segment[-1] - segment[0]) for segment in dashes]
    np.testing.assert_allclose(lengths[:-1], 0.08, atol=1e-12)
    assert all(len(segment) >= 2 for segment in dashes)


def test_trail_keeps_points_after_the_minimum_spacing_and_grows():
    trail = PathTrail(min_spacing_m=0.01, capacity=2)
    assert trail.add([0, 0, 0])
    assert not trail.add([0.005, 0, 0])
    for index in range(1, 6):
        assert trail.add([0.02 * index, 0, 0])
    np.testing.assert_allclose(trail.points[:, 0], [0, 0.02, 0.04, 0.06, 0.08, 0.10])
    with pytest.raises(ValueError):
        trail.add([np.nan, 0, 0])


def test_usd_overlay_writes_curves_and_setpoint():
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom

    stage = Usd.Stage.CreateInMemory()
    overlay = PathOverlay(stage)
    trajectory = helix()
    overlay.set_reference(trajectory.path_points())
    overlay.update(np.array([[0, 0, 0.1], [0, 0, 0.2], [0, 0.1, 0.3]]), [1.0, 2.0, 3.0])
    target = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Visuals/TargetPath"))
    counts = target.GetCurveVertexCountsAttr().Get()
    assert len(counts) > 50 and sum(counts) == len(target.GetPointsAttr().Get())
    flown = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Visuals/FlownPath"))
    assert list(flown.GetCurveVertexCountsAttr().Get()) == [3]
    lower, upper = flown.GetExtentAttr().Get()
    assert lower[2] < 0.1 and upper[2] > 0.3
    translate = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Visuals/Setpoint")).GetOrderedXformOps()[0].Get()
    assert tuple(translate) == (1.0, 2.0, 3.0)
    overlay.update(np.zeros((1, 3)), [0, 0, 0])
    assert list(flown.GetCurveVertexCountsAttr().Get()) == []
