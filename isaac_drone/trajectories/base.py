"""Trajectory framework: geometric segments traversed by rest-to-rest time laws, then a hold.

A trajectory is a sequence of ``Segment`` s (e.g. a vertical takeoff line, a
helix) starting from the vehicle's actual reset state, followed by an unlimited
hold at the endpoint. Each segment is a fixed path parameterized by progress
sigma in [0, 1]; a time law (``timing``) decides how fast it is traversed:

* a configured duration uses the ninth-degree C4 ``HermiteTimeLaw``;
* ``None`` leaves the segment to the minimum-time planner, which assigns an
  accelerate/cruise/decelerate ``CruiseTimeLaw`` from the actual vehicle limits.

Because every law starts and ends at rest with four zero derivatives, the
references are C4 across segment joints. Sampling is pure: repeated or
out-of-order calls never change state.

Add a trajectory by subclassing ``SegmentedTrajectory`` (implement
``_build``), declaring a parameter dataclass and registering a factory::

    @TRAJECTORIES.register("my_path", params=MyParams)
    def build(params: MyParams) -> MyTrajectory: ...

then select it with ``trajectory: {kind: my_path, ...params}`` in YAML.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from isaac_drone.core.registry import Registry
from isaac_drone.core.rotations import quaternion_to_matrix
from isaac_drone.core.types import TrajectorySetpoint, VehicleState
from isaac_drone.core.validation import finite_array, finite_scalar

from .timing import CruiseTimeLaw, HermiteTimeLaw

TRAJECTORIES = Registry("trajectory")
HOLD_PHASE = "hold"


class Trajectory(Protocol):
    """What the control loop needs. ``SegmentedTrajectory`` adds planning, feasibility and path queries;
    other implementations (e.g. an online generator) get neither minimum-time planning nor screening."""

    start_time_s: float

    def reset(self, initial_state: VehicleState, start_time_s: float | None = None) -> None: ...

    def sample(self, time_s: float) -> TrajectorySetpoint: ...

    def phase(self, time_s: float) -> str: ...


@dataclass(frozen=True)
class Segment:
    """One geometric path piece.

    ``setpoint(progress)`` maps ``[sigma, sigma', sigma'', sigma''', sigma'''']``
    to a full reference (position..snap, yaw and its rates). Yaw must be affine
    in sigma, changing by ``yaw_change_rad`` over the segment, so the yaw rate
    peaks where sigma' does. ``revolutions`` (|turns| of a periodic path, else
    0) sets the sampling density of planning/feasibility checks: 16 samples per
    revolution, never fewer than 33; ``hermite_samples`` is used for a
    fixed-duration law.
    """

    name: str
    setpoint: Callable[[np.ndarray], TrajectorySetpoint]
    length_m: float
    yaw_change_rad: float
    duration_s: float | None
    revolutions: float = 0.0
    hermite_samples: int = 33


class SegmentedTrajectory:
    """Base class: segments from the reset state, then an endpoint hold (phase ``hold``).

    ``reset(initial_state, start_time_s)`` anchors geometry at the actual CoM
    and heading and starts the first segment at ``start_time_s`` (default: the
    state's time). Sampling earlier than that is an error; a caller holding
    the vehicle before the start (e.g. motor spin-up) samples the start time.
    With nonzero rest tolerances, reset rejects a moving vehicle.
    """

    def __init__(
        self,
        *,
        thrust_utilization: float | None = None,
        rest_speed_tolerance_m_s: float | None = None,
        rest_angular_speed_tolerance_rad_s: float | None = None,
    ):
        self.thrust_utilization = thrust_utilization
        self._rest_speed = rest_speed_tolerance_m_s
        self._rest_angular_speed = rest_angular_speed_tolerance_rad_s
        self._segments: list[Segment] | None = None
        self._laws: dict = {}
        self._timed = False

    # -- subclass hook -------------------------------------------------------------------------
    def _build(self, position_w: np.ndarray, yaw_rad: float) -> tuple[list[Segment], np.ndarray, float]:
        """Return (segments, endpoint position, endpoint yaw) for a start at ``position_w``/``yaw_rad``."""
        raise NotImplementedError

    # -- reset and timing ----------------------------------------------------------------------
    def reset(self, initial_state: VehicleState, start_time_s: float | None = None) -> None:
        self._segments, self._timed = None, False
        position = finite_array(initial_state.position_w, (3,), "initial CoM position")
        velocity = finite_array(initial_state.linear_velocity_w, (3,), "initial CoM velocity")
        omega = finite_array(initial_state.angular_velocity_b, (3,), "initial angular velocity")
        if self._rest_speed is not None and np.linalg.norm(velocity) > self._rest_speed:
            raise ValueError("Trajectory reset requires settled CoM speed within rest_speed_tolerance_m_s")
        if self._rest_angular_speed is not None and np.linalg.norm(omega) > self._rest_angular_speed:
            raise ValueError(
                "Trajectory reset requires settled angular speed within rest_angular_speed_tolerance_rad_s"
            )
        rotation = quaternion_to_matrix(initial_state.quaternion_wxyz)
        if np.linalg.norm(rotation[:2, 0]) < 1e-10:
            raise ValueError("Initial body-X heading is undefined at a vertical body-X orientation")
        origin = finite_scalar(initial_state.time_s if start_time_s is None else start_time_s, "trajectory start")
        if origin < initial_state.time_s:
            raise ValueError("Trajectory start precedes the reset state")
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        segments, endpoint, end_yaw = self._build(position, yaw)
        if not np.all(np.isfinite(np.r_[endpoint, end_yaw])):
            raise ValueError("Trajectory endpoint or heading exceeds finite numerical range")
        names = [segment.name for segment in segments]
        if len(set(names)) != len(names) or HOLD_PHASE in names:
            raise ValueError(f"Segment names must be unique and differ from {HOLD_PHASE!r}: {names}")
        self._origin, self._start_position, self._start_yaw = origin, position, yaw
        self._endpoint, self._end_yaw = np.asarray(endpoint, dtype=float), float(end_yaw)
        self._segments = segments
        self._laws = {
            segment.name: None if segment.duration_s is None else HermiteTimeLaw(segment.duration_s)
            for segment in segments
        }
        if not self.planned_phases:
            self._finalize_timing()

    def set_time_laws(self, **laws) -> None:
        """Assign time laws to the planned (null-duration) segments after reset."""
        self._require_geometry()
        unknown = set(laws) - set(self.planned_phases)
        if unknown:
            raise ValueError(f"Only null-duration segments can be planned; got {sorted(unknown)}")
        for name, law in laws.items():
            if not isinstance(law, (HermiteTimeLaw, CruiseTimeLaw)):
                raise TypeError(f"{name} time law must be a HermiteTimeLaw or CruiseTimeLaw")
            self._laws[name] = law
        self._timed = False
        if all(law is not None for law in self._laws.values()):
            self._finalize_timing()

    def _finalize_timing(self) -> None:
        starts = [self._origin]
        for segment in self._segments:
            starts.append(starts[-1] + self._laws[segment.name].duration_s)
        boundaries = np.array(starts)
        if not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0):
            raise ValueError("Trajectory segments are not representable at the start time")
        self._boundaries = boundaries
        self._timed = True

    def _require_geometry(self) -> None:
        if self._segments is None:
            raise RuntimeError("reset the trajectory before using it")

    def _require_timing(self) -> None:
        self._require_geometry()
        if not self._timed:
            raise RuntimeError(
                f"Plan the null-duration segments {list(self.planned_phases)} "
                "(minimum-time planner) after reset before sampling"
            )

    # -- geometry/timing queries ---------------------------------------------------------------
    @property
    def segments(self) -> list[Segment]:
        self._require_geometry()
        return list(self._segments)

    @property
    def planned_phases(self) -> tuple[str, ...]:
        """Segments whose null configured duration must be planned after reset."""
        self._require_geometry()
        return tuple(segment.name for segment in self._segments if segment.duration_s is None)

    @property
    def time_laws(self) -> dict:
        return dict(self._laws)

    @property
    def start_time_s(self) -> float:
        self._require_geometry()
        return self._origin

    @property
    def start_position_w(self) -> np.ndarray:
        self._require_geometry()
        return self._start_position.copy()

    @property
    def endpoint_position_w(self) -> np.ndarray:
        self._require_geometry()
        return self._endpoint.copy()

    @property
    def motion_duration_s(self) -> float:
        """Total duration of all segments (the endpoint hold is unlimited)."""
        self._require_timing()
        return float(sum(self._laws[segment.name].duration_s for segment in self._segments))

    @property
    def phase_boundaries_s(self) -> list[tuple[float, str]]:
        """(absolute start [s], phase) of every segment and of the endpoint hold."""
        self._require_timing()
        names = [segment.name for segment in self._segments] + [HOLD_PHASE]
        return [(float(start), name) for start, name in zip(self._boundaries, names)]

    def law_segments(self) -> list[tuple[str, str, float, float]]:
        """(phase, law segment, absolute start, absolute end) of every time-law piece."""
        self._require_timing()
        result = []
        for index, segment in enumerate(self._segments):
            start = self._boundaries[index]
            result.extend(
                (segment.name, name, start + begin, start + end)
                for name, begin, end in self._laws[segment.name].segments
            )
        return result

    def timing_report(self) -> dict:
        """Phase boundaries plus each segment's law and peak speed along its path."""
        self._require_timing()
        phases = {}
        for segment in self._segments:
            law = self._laws[segment.name]
            phases[segment.name] = {
                **law.describe(),
                "planned": segment.duration_s is None,
                "path_length_m": segment.length_m,
                "peak_speed_m_s": segment.length_m * law.peak_rate,
            }
        return {
            "phase_boundaries_s": [start for start, _ in self.phase_boundaries_s],
            "phase_names": [name for _, name in self.phase_boundaries_s],
            "motion_duration_s": self.motion_duration_s,
            "thrust_utilization": self.thrust_utilization,
            "phases": phases,
        }

    # -- sampling ------------------------------------------------------------------------------
    def _index(self, time_s: float) -> int:
        """Index of the active segment; len(segments) means the endpoint hold."""
        self._require_timing()
        time = finite_scalar(time_s, "trajectory time_s")
        if time < self._origin:
            raise ValueError("Trajectory sample time precedes its start time")
        return int(np.searchsorted(self._boundaries[1:], time, side="right"))

    def phase(self, time_s: float) -> str:
        index = self._index(time_s)
        return self._segments[index].name if index < len(self._segments) else HOLD_PHASE

    def sample(self, time_s: float) -> TrajectorySetpoint:
        index = self._index(time_s)
        if index < len(self._segments):
            segment = self._segments[index]
            return segment.setpoint(self._laws[segment.name].derivatives(time_s - self._boundaries[index]))
        zero = np.zeros(3)
        return TrajectorySetpoint(
            position_w=self._endpoint.copy(),
            velocity_w=zero.copy(),
            acceleration_w=zero.copy(),
            yaw_rad=self._end_yaw,
            yaw_rate_rad_s=0.0,
            yaw_acceleration_rad_s2=0.0,
            jerk_w=zero.copy(),
            snap_w=zero.copy(),
        )

    def path_points(self, step_s: float = 0.005) -> np.ndarray:
        """Reference CoM positions from the start through the endpoint, sampled every ``step_s``."""
        self._require_timing()
        count = max(2, int(np.ceil(self.motion_duration_s / step_s)) + 1)
        times = np.linspace(self._origin, self._boundaries[-1], count)
        return np.array([self.sample(float(time)).position_w for time in times])
