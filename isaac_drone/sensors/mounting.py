"""Sensor mounting: position relative to the root link and orientation as Z-Y-X Euler angles."""

from __future__ import annotations

import numpy as np

from isaac_drone.core.rotations import rpy_to_rotation
from isaac_drone.core.types import MassProperties
from isaac_drone.core.validation import finite_array


class Mounting:
    """Where a sensor sits on the vehicle.

    ``position_b_m`` is the sensor position relative to the root link origin
    (the USD ``base_link``), in body axes; ``orientation_rpy_rad`` rotates body
    axes into sensor axes (roll, pitch, yaw of the sensor frame in the body).
    The lever arm relative to the CoM, needed for point kinematics, is known
    only after the backend reports its mass properties (``bind``).
    """

    def __init__(self, position_b_m, orientation_rpy_rad=(0.0, 0.0, 0.0)):
        self.position_b = finite_array(position_b_m, (3,), "position_b_m")
        self.orientation_rpy = finite_array(orientation_rpy_rad, (3,), "orientation_rpy_rad")
        # rpy_to_rotation gives sensor-to-body; readings need body-to-sensor.
        self.rotation_sb = rpy_to_rotation(*self.orientation_rpy).T
        self.lever_arm_b = None

    def bind(self, mass: MassProperties) -> None:
        self.lever_arm_b = self.position_b - mass.com_b

    def to_sensor(self, vector_b) -> np.ndarray:
        return self.rotation_sb @ finite_array(vector_b, (3,), "vector_b")

    def describe(self) -> dict:
        return {
            "position_b_m": self.position_b.tolist(),
            "orientation_rpy_rad": self.orientation_rpy.tolist(),
            "lever_arm_from_com_b_m": None if self.lever_arm_b is None else self.lever_arm_b.tolist(),
        }
