"""Camera framing without simulator dependencies or changes to vehicle state."""

from __future__ import annotations

import numpy as np

from isaac_drone.core.validation import finite_array


class RecordingViews:
    """A fixed overview plus world-aligned cameras tracking the actual CoM.

    The rig uses a 36 mm horizontal aperture and 24 mm focal length. Fit the
    overview to the complete planned path with room for the vehicle/tracking
    error. Follow cameras translate with truth, never with the desired path or
    vehicle roll/yaw, keeping the horizon stable during aggressive maneuvers.
    """

    def __init__(self, names, reference_points, width, height):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names) or set(self.names) - {"overview", "follow", "top"}:
            raise ValueError("Choose unique cameras from overview, follow, top")
        points = np.asarray(reference_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("reference_points must be finite (N, 3) positions")
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(1.0, float(np.linalg.norm(points - center, axis=1).max()) + 0.6)
        half_fov = np.arctan(0.75 * min(1.0, height / width))
        direction = np.array([1.0, -1.0, 0.7])
        distance = 1.1 * radius / np.sin(half_fov)
        self._overview = {
            "eye": center + distance * direction / np.linalg.norm(direction),
            "target": center,
            "up": np.array([0.0, 0.0, 1.0]),
        }

    def poses(self, position_w):
        position = finite_array(position_w, (3,), "recording camera target")
        views = {
            "overview": self._overview,
            "follow": {"eye": position + [2.5, -3.5, 1.8], "target": position, "up": np.array([0.0, 0.0, 1.0])},
            "top": {"eye": position + [0.0, 0.0, 6.0], "target": position, "up": np.array([0.0, 1.0, 0.0])},
        }
        return {name: {key: value.copy() for key, value in views[name].items()} for name in self.names}
