"""Framing and actual-state tracking, independent of rendering/physics."""
import numpy as np
import pytest

from isaac_drone.viz.camera_views import RecordingViews


def test_tracking_cameras_follow_actual_position_and_overview_stays_fixed():
    views = RecordingViews(["overview", "follow", "top"], [[0, 0, 0], [2, 2, 6]], 1280, 720)
    before = views.poses([0, 0, 0])
    after = views.poses([3, -2, 4])
    for name in ("follow", "top"):
        np.testing.assert_allclose(after[name]["eye"] - before[name]["eye"], [3, -2, 4])
        np.testing.assert_allclose(after[name]["target"], [3, -2, 4])
    for field in ("eye", "target", "up"):
        np.testing.assert_array_equal(after["overview"][field], before["overview"][field])
    # A valid top-down look-at basis must not have collinear direction/up.
    top = after["top"]
    assert np.linalg.norm(np.cross(top["target"]-top["eye"], top["up"])) > 0


@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280)])
def test_overview_fits_entire_path_for_landscape_and_portrait(width, height):
    points = np.array([[0, 0, 0], [-5, 2, 10], [2, -3, 4]])
    pose = RecordingViews(["overview"], points, width, height).poses([0, 0, 0])["overview"]
    forward = pose["target"] - pose["eye"]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, pose["up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - pose["eye"]
    depth = relative @ forward
    assert (depth > 0).all()
    assert (np.abs(relative @ right) / depth < 0.75).all()
    assert (np.abs(relative @ up) / depth < 0.75 * height / width).all()
