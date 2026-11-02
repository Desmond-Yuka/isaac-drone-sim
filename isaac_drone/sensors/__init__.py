"""Synthetic sensor readings from simulator truth: IMU, magnetometer, barometer and GNSS.

Error models follow the MATLAB ``imuSensor``, ``altimeterSensor`` and ``gpsSensor``
documentation (parameters, processing order and noise scalings), with default
values for the Pixhawk 6C components in ``configs/arl_robot_1.yaml``. Pure NumPy.
"""

from .gps import GPS_PROFILES, GpsSensor, resolve_gps_parameters
from .noise import NOISE_TYPES, DecayingNoise, GaussMarkov, RandomWalk, random_walk_step_std, white_noise_std
from .suite import SEED_DOMAIN, SENSOR_ORDER, SensorSuite, build_sensor_suite, delay_steps, sample_divisor
from .triaxial import TriaxialModel, TriaxialParams, misalignment_matrix
from .truth import ATTITUDE_PAIRINGS, DEFAULT_ATTITUDE_PAIRING, StepTruth

__all__ = [
    "ATTITUDE_PAIRINGS",
    "DEFAULT_ATTITUDE_PAIRING",
    "GPS_PROFILES",
    "NOISE_TYPES",
    "SEED_DOMAIN",
    "SENSOR_ORDER",
    "DecayingNoise",
    "GaussMarkov",
    "GpsSensor",
    "RandomWalk",
    "SensorSuite",
    "StepTruth",
    "TriaxialModel",
    "TriaxialParams",
    "build_sensor_suite",
    "delay_steps",
    "misalignment_matrix",
    "random_walk_step_std",
    "resolve_gps_parameters",
    "sample_divisor",
    "white_noise_std",
]
