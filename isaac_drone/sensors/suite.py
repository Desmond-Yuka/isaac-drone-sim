"""All configured sensors behind one ``sample`` call: rates, delays and random streams.

Every sensor rate must divide the physics rate and every delay must be a whole
number of physics steps. A sensor samples at the end of the physics step whose
end time is a multiple of its period; its reading is released ``delay`` steps
later and keeps the time stamp of the instant it describes. Random numbers come
from ``SeedSequence([simulation.seed, SEED_DOMAIN])`` with one child stream per
sensor in a fixed order, so the flight itself (which draws from
``SeedSequence(seed)`` in ``effects/pipeline.py``) is unchanged by enabling,
disabling or reconfiguring sensors.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np

from isaac_drone.core.types import MassProperties, VehicleState
from isaac_drone.core.validation import finite_array, finite_scalar

from .barometer import BAROMETER_FIELDS, BarometerSensor
from .gps import GPS_FIELDS, GpsSensor
from .imu import IMU_FIELDS, ImuSensor
from .magnetometer import MAGNETOMETER_FIELDS, MagnetometerSensor
from .truth import DEFAULT_ATTITUDE_PAIRING, StepTruth

SENSOR_ORDER = ("imu", "magnetometer", "barometer", "gps")
SENSOR_CLASSES = {
    "imu": ImuSensor,
    "magnetometer": MagnetometerSensor,
    "barometer": BarometerSensor,
    "gps": GpsSensor,
}
SENSOR_FIELDS = {
    "imu": IMU_FIELDS,
    "magnetometer": MAGNETOMETER_FIELDS,
    "barometer": BAROMETER_FIELDS,
    "gps": GPS_FIELDS,
}
SEED_DOMAIN = 0x53454E53  # "SENS": separates the sensor streams from the effects' SeedSequence(seed)


def sample_divisor(rate_hz: float, dt_s: float, path: str) -> int:
    """Physics steps per sample; the rate must divide the physics rate exactly."""
    rate = finite_scalar(rate_hz, f"{path}.rate_hz")
    dt = finite_scalar(dt_s, "dt_s")
    if rate <= 0 or dt <= 0:
        raise ValueError(f"{path}.rate_hz and simulation.dt must be positive")
    steps = 1.0 / (rate * dt)
    rounded = round(steps)
    if rounded < 1 or not np.isclose(steps, rounded, rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"{path}.rate_hz = {rate:g} Hz must divide the physics rate {1.0 / dt:g} Hz (dt = {dt:g} s) exactly"
        )
    return int(rounded)


def delay_steps(delay_s: float, dt_s: float, path: str) -> int:
    """Delay in physics steps; the delay must be a whole number of steps (0.12 s is 24 steps at 200 Hz)."""
    delay = finite_scalar(delay_s, f"{path}.delay_s")
    dt = finite_scalar(dt_s, "dt_s")
    if delay < 0:
        raise ValueError(f"{path}.delay_s must be nonnegative")
    steps = delay / dt
    rounded = round(steps)
    if not np.isclose(steps, rounded, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{path}.delay_s = {delay:g} s must be a whole number of physics steps of {dt:g} s")
    return int(rounded)


class SensorSuite:
    """The enabled sensors of a ``sensors:`` section, sampled once per completed physics step."""

    def __init__(self, section: Mapping, dt_s: float, gravity_w):
        self.dt_s = finite_scalar(dt_s, "dt_s")
        self.gravity_w = finite_array(gravity_w, (3,), "gravity_w")
        self.sensors = {}
        self.divisors = {}
        self.delays = {}
        for name in SENSOR_ORDER:
            sub = section[name]
            if not sub["enabled"]:
                continue
            path = f"sensors.{name}"
            self.sensors[name] = SENSOR_CLASSES[name](sub, path)
            self.divisors[name] = sample_divisor(sub["rate_hz"], self.dt_s, path)
            self.delays[name] = delay_steps(sub["delay_s"], self.dt_s, path)
        self.attitude_pairing = DEFAULT_ATTITUDE_PAIRING
        self._queues = {name: deque() for name in self.sensors}
        self._bound = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.sensors)

    def bind_vehicle(self, mass: MassProperties, attitude_pairing: str = DEFAULT_ATTITUDE_PAIRING) -> None:
        """Give the sensors the plant's mass properties (lever arms, weight) and specific-force pairing."""
        for sensor in self.sensors.values():
            sensor.bind(mass, self.gravity_w)
        self.attitude_pairing = attitude_pairing
        self._bound = True

    def reset(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("the sensor suite needs the nonnegative integer simulation.seed")
        streams = np.random.SeedSequence([int(seed), SEED_DOMAIN]).spawn(len(SENSOR_ORDER))
        for name, stream in zip(SENSOR_ORDER, streams):
            if name in self.sensors:
                self.sensors[name].reset(stream)
        for queue in self._queues.values():
            queue.clear()

    def sample(self, step_index: int, before: VehicleState, after: VehicleState, telemetry: Mapping) -> dict:
        """Readings released at ``step_index`` (the index of the step just completed); ``None`` when none is due."""
        if not self._bound:
            raise RuntimeError("bind_vehicle before sampling")
        if not isinstance(telemetry, Mapping):
            raise TypeError("backend telemetry must be a mapping")
        truth = None
        readings = {name: None for name in SENSOR_ORDER}
        for name, sensor in self.sensors.items():
            if step_index % self.divisors[name] == 0:
                if truth is None:
                    truth = StepTruth(before, after, self.gravity_w, self.attitude_pairing)
                self._queues[name].append((step_index + self.delays[name], sensor.measure(truth, telemetry)))
            queue = self._queues[name]
            while queue and queue[0][0] <= step_index:
                readings[name] = queue.popleft()[1]
        return readings

    def describe(self) -> dict:
        """Metadata: which sensors run, at what rate and delay, with their derived noise levels."""
        return {
            "attitude_pairing": self.attitude_pairing,
            "seed_domain": SEED_DOMAIN,
            "sensors": {
                name: {
                    "rate_hz": self.sensors[name].rate_hz,
                    "steps_per_sample": self.divisors[name],
                    "delay_s": self.delays[name] * self.dt_s,
                    "delay_steps": self.delays[name],
                    **self.sensors[name].describe(),
                }
                for name in self.sensors
            },
        }


def validate_sensors_section(section: Mapping, dt_s: float, gravity_w) -> None:
    """Reject a malformed ``sensors:`` section; every sensor is constructed whether enabled or not.

    Rates and delays are matched to the physics step only for sensors that will actually sample,
    so a switched-off section never constrains ``simulation.dt``.
    """
    if not isinstance(section, Mapping):
        raise TypeError("sensors must be a mapping")
    expected = {"enabled", *SENSOR_ORDER}
    if set(section) != expected:
        raise ValueError(f"unknown={sorted(set(section) - expected)}, missing={sorted(expected - set(section))}")
    if not isinstance(section["enabled"], bool):
        raise TypeError("sensors.enabled must be true or false")
    finite_array(gravity_w, (3,), "simulation.gravity")
    for name in SENSOR_ORDER:
        sub = section[name]
        path = f"sensors.{name}"
        if not isinstance(sub, Mapping):
            raise TypeError(f"{path} must be a mapping")
        fields = SENSOR_FIELDS[name]
        if set(sub) != set(fields):
            raise ValueError(
                f"{path}: unknown={sorted(set(sub) - set(fields))}, missing={sorted(set(fields) - set(sub))}"
            )
        if not isinstance(sub["enabled"], bool):
            raise TypeError(f"{path}.enabled must be true or false")
        SENSOR_CLASSES[name](sub, path)
        if section["enabled"] and sub["enabled"]:
            sample_divisor(sub["rate_hz"], dt_s, path)
            delay_steps(sub["delay_s"], dt_s, path)


def build_sensor_suite(section: Mapping | None, dt_s: float, gravity_w) -> SensorSuite | None:
    """A suite for an enabled ``sensors:`` section with at least one enabled sensor, else None."""
    if section is None or not section.get("enabled", False):
        return None
    suite = SensorSuite(section, dt_s, gravity_w)
    return suite if suite.sensors else None
