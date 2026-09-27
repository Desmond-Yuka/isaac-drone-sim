"""A constant CoM target for initial closed-loop integration checks."""
import math
import numpy as np

from isaac_drone.types import TrajectorySetpoint, VehicleState, finite_array


class HoldTrajectory:
    def __init__(self, position_w_m=None, yaw_rad=None):
        self._position = None if position_w_m is None else finite_array(position_w_m, (3,), "hold position")
        self._yaw = yaw_rad
        if yaw_rad is not None and not np.isfinite(yaw_rad):
            raise ValueError("hold yaw must be finite")
        self._target = None

    def reset(self, initial_state: VehicleState) -> None:
        w, x, y, z = initial_state.quaternion_wxyz
        yaw = math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z)) if self._yaw is None else self._yaw
        self._target = TrajectorySetpoint(
            initial_state.position_w if self._position is None else self._position,
            yaw_rad=float(yaw))

    def sample(self, time_s: float) -> TrajectorySetpoint:
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("trajectory time must be finite and nonnegative")
        if self._target is None:
            raise RuntimeError("reset trajectory before sampling")
        return self._target
