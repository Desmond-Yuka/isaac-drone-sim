"""A constant CoM target: the simplest closed-loop check (no motion segments)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaac_drone.core.params import Vec3

from .base import TRAJECTORIES, SegmentedTrajectory


@dataclass(frozen=True)
class HoldParams:
    """``trajectory: {kind: hold}``; null position/yaw keep the actual reset CoM position/heading."""

    position_w_m: Vec3 | None = None
    yaw_rad: float | None = None


class HoldTrajectory(SegmentedTrajectory):
    def __init__(self, params: HoldParams | None = None):
        super().__init__()
        self.params = params or HoldParams()

    def _build(self, position_w, yaw_rad):
        target = position_w if self.params.position_w_m is None else np.array(self.params.position_w_m, dtype=float)
        return [], target, yaw_rad if self.params.yaw_rad is None else self.params.yaw_rad


@TRAJECTORIES.register("hold", params=HoldParams)
def build_hold(params: HoldParams) -> HoldTrajectory:
    """Hold a fixed CoM position and heading."""
    return HoldTrajectory(params)
