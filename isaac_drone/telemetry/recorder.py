"""Strict JSONL telemetry, flat engineering CSV, and inspectable field semantics.

Serialization and the entire basic-record validation complete before either
stream is written. Missing fields are left blank in CSV; zero is never used as a
replacement for unavailable measurements. CSV rows are produced only by records
with a non-None ``basic`` mapping, preserving existing JSON-only records/events.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from io import StringIO
from numbers import Real
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dump_json(value):
    return json.dumps(value, default=json_value, ensure_ascii=False, allow_nan=False)


def asset_hashes(asset_path):
    """Hash all local USD layers beside this asset, not just its small root layer."""
    root = Path(asset_path).parent
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.usd"))
    }


@dataclass(frozen=True)
class BasicField:
    name: str
    unit: str
    frame: str
    sampling: str
    source: str
    description: str
    columns: tuple[str, ...]
    components: tuple[str, ...] = ()
    dtype: str = "float64"

    def metadata(self):
        return {
            "columns": list(self.columns),
            "components": list(self.components),
            "shape": [len(self.components)] if self.components else [],
            "dtype": self.dtype,
            "unit": self.unit,
            "frame": self.frame,
            "sampling": self.sampling,
            "source": self.source,
            "description": self.description,
            "missing": "None or absent is an empty CSV cell; no value is inferred",
        }


_END = "Endpoint at sample_time_s, after the physics step"
_START = "Actuation for the interval beginning at step_start_time_s"
_MOTOR_START = (
    "After the native motor update, used for the physics interval beginning at "
    "step_start_time_s; read back after the completed step"
)
_INTERVAL = "Average over [step_start_time_s, sample_time_s] of length interval_dt_s"
_REFERENCE = "Held target sampled at reference_sample_time_s, compared with the current endpoint"
_TRUTH = "Post-step simulator truth for the whole-vehicle centre of mass"
_ROTORS = ("BL", "BR", "FL", "FR")
_AXES = ("x", "y", "z")
_RPY = ("roll", "pitch", "yaw")
_QUATERNION = ("w", "x", "y", "z")
_DESIRED_ATTITUDE = "Geometric-controller desired attitude R_d, held since reference_sample_time_s"


def _scalar(name, unit, sampling, source, description, *, dtype="float64"):
    return BasicField(name, unit, "not applicable", sampling, source, description, (name,), dtype=dtype)


def _vector(name, prefix, suffix, unit, frame, sampling, source, description, components=_AXES):
    columns = tuple(f"{prefix}_{component}_{suffix}" if suffix else f"{prefix}_{component}" for component in components)
    return BasicField(name, unit, frame, sampling, source, description, columns, tuple(components))


BASIC_FIELDS = (
    _scalar(
        "step_start_time_s",
        "s",
        "Interval start",
        "Runtime physics clock",
        "Simulation time before this physics interval; not wall-clock time.",
    ),
    _scalar(
        "sample_time_s",
        "s",
        "Interval end",
        "Runtime physics clock",
        "Simulation time at the reported post-step state.",
    ),
    _scalar(
        "interval_dt_s",
        "s",
        "This physics interval",
        "Runtime physics timestep",
        "Duration used for this interval's kinematic differences; not the logging cadence.",
    ),
    _scalar(
        "reference_sample_time_s",
        "s",
        "Last controller/reference update",
        "Runtime control clock",
        "The target may be held over several physics steps when control is decimated.",
    ),
    _vector(
        "position_w_m",
        "position_w",
        "m",
        "m",
        "world",
        _END,
        _TRUTH,
        "Actual whole-vehicle centre-of-mass position, not a target or link-origin pose.",
    ),
    _vector(
        "velocity_w_m_s",
        "velocity_w",
        "m_s",
        "m/s",
        "world",
        _END,
        _TRUTH,
        "Actual whole-vehicle centre-of-mass linear velocity.",
    ),
    _vector(
        "acceleration_w_m_s2",
        "acceleration_w",
        "m_s2",
        "m/s^2",
        "world",
        _INTERVAL,
        "(velocity_end_w - velocity_start_w) / interval_dt_s",
        "Kinematic acceleration includes gravity's motion contribution; not IMU specific force.",
    ),
    _vector(
        "acceleration_native_w_m_s2",
        "acceleration_native_w",
        "m_s2",
        "m/s^2",
        "world",
        _END,
        "Mass-weighted PhysX link-CoM accelerations; see acceleration_native_source",
        "Post-step native solver acceleration of the whole-vehicle CoM, including gravity; distinct from "
        "interval-average velocity differences. None without a fresh solver step.",
    ),
    _scalar(
        "acceleration_native_source",
        "text",
        _END,
        "Backend supplied native-acceleration provenance string",
        "Identifies the solver acceleration accessor or explains unavailability; not an IMU measurement.",
        dtype="string",
    ),
    _vector(
        "quaternion_wxyz",
        "quaternion_wxyz",
        "",
        "1",
        "body to world",
        _END,
        "Post-step simulator body orientation",
        "Active unit quaternion mapping body coordinates to world coordinates, scalar first.",
        components=_QUATERNION,
    ),
    _vector(
        "attitude_rpy_rad",
        "attitude",
        "rad",
        "rad",
        "body to world, Z-Y-X Euler angles",
        _END,
        "quaternion_wxyz converted to roll, pitch, yaw",
        "Readable attitude; quaternion_wxyz stays the unambiguous record.",
        _RPY,
    ),
    _vector(
        "angular_velocity_b_rad_s",
        "angular_velocity_b",
        "rad_s",
        "rad/s",
        "body at sample_time_s",
        _END,
        "Post-step simulator angular velocity expressed in endpoint body axes",
        "Body axes share the USD root link orientation.",
    ),
    _vector(
        "angular_velocity_w_rad_s",
        "angular_velocity_w",
        "rad_s",
        "rad/s",
        "world",
        _END,
        "Endpoint body-to-world rotation applied to angular_velocity_b_rad_s",
        "Endpoint angular velocity in the common world basis used for angular-acceleration differencing.",
    ),
    _vector(
        "angular_acceleration_w_rad_s2",
        "angular_acceleration_w",
        "rad_s2",
        "rad/s^2",
        "world",
        _INTERVAL,
        "(angular_velocity_end_w - angular_velocity_start_w) / interval_dt_s",
        "Difference is taken in a common world basis, not between differently oriented body bases.",
    ),
    _vector(
        "angular_acceleration_b_rad_s2",
        "angular_acceleration_b",
        "rad_s2",
        "rad/s^2",
        "body at sample_time_s",
        _INTERVAL,
        "Endpoint world-to-body rotation applied to angular_acceleration_w_rad_s2",
        "World interval-average angular acceleration expressed in endpoint body axes.",
    ),
    _vector(
        "target_position_w_m",
        "target_position_w",
        "m",
        "m",
        "world",
        _REFERENCE,
        "Held trajectory/controller setpoint",
        "Target whole-vehicle centre-of-mass position.",
    ),
    _vector(
        "target_velocity_w_m_s",
        "target_velocity_w",
        "m_s",
        "m/s",
        "world",
        _REFERENCE,
        "Held trajectory/controller setpoint",
        "Target centre-of-mass linear velocity.",
    ),
    _vector(
        "target_acceleration_w_m_s2",
        "target_acceleration_w",
        "m_s2",
        "m/s^2",
        "world",
        _REFERENCE,
        "Held trajectory/controller setpoint",
        "Target centre-of-mass kinematic acceleration.",
    ),
    _vector(
        "target_quaternion_wxyz",
        "target_quaternion_wxyz",
        "",
        "1",
        "body to world",
        _REFERENCE,
        _DESIRED_ATTITUDE,
        "Desired attitude: body +Z along the desired force, heading from the trajectory yaw.",
        _QUATERNION,
    ),
    _vector(
        "target_attitude_rpy_rad",
        "target_attitude",
        "rad",
        "rad",
        "body to world, Z-Y-X Euler angles",
        _REFERENCE,
        "target_quaternion_wxyz converted to roll, pitch, yaw",
        "Readable desired attitude.",
        _RPY,
    ),
    _vector(
        "target_angular_velocity_b_rad_s",
        "target_angular_velocity_b",
        "rad_s",
        "rad/s",
        "body at sample_time_s",
        _REFERENCE,
        "Controller desired angular velocity rotated from desired-body into endpoint body axes",
        "The desired rate expressed where the controller compares it with the measured rate.",
    ),
    _vector(
        "position_error_w_m",
        "position_error_w",
        "m",
        "m",
        "world",
        _END,
        "target_position_w_m - position_w_m",
        "Held target minus actual endpoint position; a positive component means target lies ahead on that axis.",
    ),
    _scalar(
        "position_error_norm_m",
        "m",
        _END,
        "Euclidean norm of position_error_w_m",
        "Magnitude of the endpoint position-tracking error.",
    ),
    _vector(
        "velocity_error_w_m_s",
        "velocity_error_w",
        "m_s",
        "m/s",
        "world",
        _END,
        "target_velocity_w_m_s - velocity_w_m_s",
        "Held target minus actual endpoint velocity.",
    ),
    _scalar(
        "velocity_error_norm_m_s",
        "m/s",
        _END,
        "Euclidean norm of velocity_error_w_m_s",
        "Magnitude of the velocity-tracking error.",
    ),
    _vector(
        "acceleration_error_w_m_s2",
        "acceleration_error_w",
        "m_s2",
        "m/s^2",
        "world",
        _INTERVAL,
        "target_acceleration_w_m_s2 - acceleration_w_m_s2",
        "Held target minus the interval-average kinematic acceleration.",
    ),
    _scalar(
        "acceleration_error_norm_m_s2",
        "m/s^2",
        _INTERVAL,
        "Euclidean norm of acceleration_error_w_m_s2",
        "Magnitude of the acceleration-tracking error.",
    ),
    _vector(
        "attitude_error_rpy_rad",
        "attitude_error",
        "rad",
        "rad",
        "Z-Y-X Euler angle differences",
        _END,
        "target_attitude_rpy_rad - attitude_rpy_rad, each wrapped to [-pi, pi)",
        "Per-angle difference for reading; Euler differences are not a rotation vector.",
        _RPY,
    ),
    _scalar(
        "attitude_error_angle_rad",
        "rad",
        _END,
        "|log(R_actual^T R_target)|",
        "Angle of the single rotation that takes the actual attitude to the desired one.",
    ),
    _vector(
        "angular_velocity_error_b_rad_s",
        "angular_velocity_error_b",
        "rad_s",
        "rad/s",
        "body at sample_time_s",
        _END,
        "target_angular_velocity_b_rad_s - angular_velocity_b_rad_s",
        "Desired minus actual body rate, both in endpoint body axes.",
    ),
    _scalar(
        "angular_velocity_error_norm_rad_s",
        "rad/s",
        _END,
        "Euclidean norm of angular_velocity_error_b_rad_s",
        "Magnitude of the angular-velocity-tracking error.",
    ),
    _vector(
        "motor_speed_rps",
        "motor_speed",
        "rps",
        "rev/s",
        "rotor magnitude",
        _MOTOR_START,
        "Direct integrated motor speed state; see motor_speed_source",
        "Magnitude in fixed BL, BR, FL, FR order; not an encoder measurement and no rotor spin sign.",
        _ROTORS,
    ),
    _vector(
        "motor_speed_rpm",
        "motor_speed",
        "rpm",
        "rev/min",
        "rotor magnitude",
        _MOTOR_START,
        "motor_speed_rps * 60; see motor_speed_source",
        "Unavailable source speeds remain unavailable; no nominal speed is substituted.",
        _ROTORS,
    ),
    _vector(
        "motor_speed_rad_s",
        "motor_speed",
        "rad_s",
        "rad/s",
        "rotor magnitude",
        _MOTOR_START,
        "motor_speed_rps * 2*pi; see motor_speed_source",
        "Positive speed magnitude in fixed BL, BR, FL, FR order, without spin-direction signs.",
        _ROTORS,
    ),
    _vector(
        "motor_speed_command_rps",
        "motor_speed_command",
        "rps",
        "rev/s",
        "rotor magnitude",
        _START,
        "RPS command sent to the motor model after allocation, startup envelope and running limits",
        "Ideal rotor speed for this interval; the motor model lags behind it.",
        _ROTORS,
    ),
    _vector(
        "motor_speed_command_rpm",
        "motor_speed_command",
        "rpm",
        "rev/min",
        "rotor magnitude",
        _START,
        "motor_speed_command_rps * 60",
        "Commanded rotor speed in rev/min.",
        _ROTORS,
    ),
    _vector(
        "motor_speed_error_rps",
        "motor_speed_error",
        "rps",
        "rev/s",
        "rotor magnitude",
        _MOTOR_START,
        "motor_speed_command_rps - motor_speed_rps",
        "Positive means the rotor is slower than commanded (spin-up lag).",
        _ROTORS,
    ),
    _vector(
        "motor_speed_error_rpm",
        "motor_speed_error",
        "rpm",
        "rev/min",
        "rotor magnitude",
        _MOTOR_START,
        "motor_speed_error_rps * 60",
        "Rotor speed error in rev/min.",
        _ROTORS,
    ),
    _vector(
        "motor_thrust_command_n",
        "motor_thrust_command",
        "n",
        "N",
        "along each configured rotor thrust axis",
        _START,
        "Backend commanded per-rotor thrust after allocation",
        "Fixed BL, BR, FL, FR order; command before native actuator dynamics, not applied thrust.",
        _ROTORS,
    ),
    _vector(
        "motor_thrust_applied_n",
        "motor_thrust_applied",
        "n",
        "N",
        "along each configured rotor thrust axis",
        _MOTOR_START,
        "This interval's native motor/thruster applied thrust state",
        "Fixed BL, BR, FL, FR order; unavailable values remain blank rather than copied from commands.",
        _ROTORS,
    ),
    _vector(
        "motor_thrust_error_n",
        "motor_thrust_error",
        "n",
        "N",
        "along each configured rotor thrust axis",
        _MOTOR_START,
        "motor_thrust_command_n - motor_thrust_applied_n",
        "Per-rotor thrust shortfall caused by motor dynamics.",
        _ROTORS,
    ),
)


def _wrench_fields():
    fields = []
    descriptions = {
        "command": (
            "Controller requested wrench before actuator allocation",
            "Requested force/couple; not the realized dynamics.",
        ),
        "allocated": (
            "Allocation matrix applied to allocated motor thrust commands",
            "Allocated command before native motor dynamics; may differ from actual submitted motor force.",
        ),
        "motor": (
            "This interval's native motor/thruster state and resolved configured geometry",
            "Motor contribution actually represented by the backend; None if unavailable, never copied from the "
            "command.",
        ),
        "disturbance": (
            "External EffectsPipeline wrench for this interval",
            "Configured external effects only; no duplicate simulator gravity, contacts or native damping.",
        ),
        "submitted": (
            "Backend motor contribution plus configured external effects submitted for this interval",
            "Submitted actuation excludes simulator gravity, collisions and native damping; unavailable if "
            "motor data is unavailable.",
        ),
    }
    for kind, (source, description) in descriptions.items():
        sampling = _MOTOR_START if kind in ("motor", "submitted") else _START
        fields.extend(
            [
                _vector(
                    f"force_{kind}_b_n",
                    f"force_{kind}_b",
                    "n",
                    "N",
                    "body at step_start_time_s",
                    sampling,
                    source,
                    description,
                ),
                _vector(
                    f"torque_{kind}_b_nm",
                    f"torque_{kind}_b",
                    "nm",
                    "N m",
                    "body at step_start_time_s, about CoM",
                    sampling,
                    source,
                    description,
                ),
            ]
        )
    fields.extend(
        [
            _vector(
                "force_net_w_n",
                "force_net_w",
                "n",
                "N",
                "world",
                _INTERVAL,
                "Vehicle mass * (velocity_end_w - velocity_start_w) / interval_dt_s",
                "Inferred net force from motion, including all effects reflected in the velocity change; not a "
                "force-sensor reading.",
            ),
            _vector(
                "torque_net_w_nm",
                "torque_net_w",
                "nm",
                "N m",
                "world, about CoM",
                _INTERVAL,
                "(world_angular_momentum_end - world_angular_momentum_start) / interval_dt_s",
                "Inferred net moment from angular-momentum change; not a torque-sensor reading or merely inertia "
                "times angular acceleration.",
            ),
            _vector(
                "force_error_b_n",
                "force_error_b",
                "n",
                "N",
                "body at step_start_time_s",
                _MOTOR_START,
                "force_command_b_n - force_motor_b_n",
                "Requested minus motor-produced force: allocation saturation plus motor lag. Not a gravity/contact "
                "balance.",
            ),
            _scalar(
                "force_error_norm_n",
                "N",
                _MOTOR_START,
                "Euclidean norm of force_error_b_n",
                "Magnitude of the force actuation error.",
            ),
            _vector(
                "torque_error_b_nm",
                "torque_error_b",
                "nm",
                "N m",
                "body at step_start_time_s, about CoM",
                _MOTOR_START,
                "torque_command_b_nm - torque_motor_b_nm",
                "Requested minus motor-produced torque: allocation saturation plus motor lag.",
            ),
            _scalar(
                "torque_error_norm_nm",
                "N m",
                _MOTOR_START,
                "Euclidean norm of torque_error_b_nm",
                "Magnitude of the torque actuation error.",
            ),
        ]
    )
    return tuple(fields)


BASIC_FIELDS += _wrench_fields() + (
    _scalar(
        "motor_speed_source",
        "text",
        _MOTOR_START,
        "Backend supplied provenance string",
        "Identifies the native state/configured coefficient used to infer speeds; never labels inferred values as "
        "encoder measurements.",
        dtype="string",
    ),
)
BASIC_CSV_COLUMNS = tuple(column for field in BASIC_FIELDS for column in field.columns)
_BASIC_NAMES = frozenset(field.name for field in BASIC_FIELDS)


def telemetry_schema():
    """Return a fresh, JSON-serializable schema with physical/temporal provenance."""
    return {
        "schema_version": 1,
        "files": {
            "telemetry.jsonl": {
                "format": "JSON Lines",
                "encoding": "UTF-8",
                "nonfinite_numbers": "rejected",
                "records": "All original records, including events and legacy records without basic",
            },
            "basic.csv": {
                "format": "CSV",
                "encoding": "UTF-8",
                "delimiter": ",",
                "header": True,
                "records": "One row for each record containing a non-None basic mapping",
                "missing_value": "",
                "column_order": list(BASIC_CSV_COLUMNS),
            },
        },
        "conventions": {
            "units": "SI, except explicitly marked rotor rev/s and rev/min",
            "world_frame": "Simulation world coordinates, Z-up",
            "body_frame": "Axes have the USD root link orientation; CoM is the reference point for all wrenches",
            "quaternion_order": ["w", "x", "y", "z"],
            "rotor_order": list(_ROTORS),
            "rotor_order_meaning": {"BL": "back left", "BR": "back right", "FL": "front left", "FR": "front right"},
            "sampling": "State is the post-step endpoint. Actuation uses step-start body axes. References are held "
            "from reference_sample_time_s.",
            "logging_cadence": "Only recorded physics steps produce rows; a gap between rows is not the "
            "differentiation interval.",
            "errors": "Every *_error field is ideal (target or command) minus actual, in the same frame and units as "
            "the pair it compares.",
            "missing_data": "Absent fields and None values stay blank. The recorder does not infer missing physical "
            "values or convert unavailable motors to zero.",
            "derived_quantities": "Velocity-difference acceleration and net force/torque are interval-average "
            "estimates from state changes, not sensor readings. Native acceleration is separately identified as an "
            "endpoint solver quantity.",
            "transforms": "To compare submitted body wrenches to world interval net wrenches, account for differing "
            "frames/times and gravity/contact/damping contributions.",
        },
        "fields": {field.name: field.metadata() for field in BASIC_FIELDS},
        "columns": [
            {
                "name": name,
                "field": field.name,
                "component": field.components[index] if field.components else None,
                "unit": field.unit,
                "frame": field.frame,
                "sampling": field.sampling,
                "source": field.source,
            }
            for field in BASIC_FIELDS
            for index, name in enumerate(field.columns)
        ],
    }


def _csv_line(values):
    buffer = StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue()


def _numeric_cell(value, field_name):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"basic.{field_name} must contain real numbers or None, not strings/booleans")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"basic.{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"basic.{field_name} must be finite")
    return result


def _basic_csv_row(basic):
    if not isinstance(basic, Mapping):
        raise ValueError("record.basic must be a mapping or None")
    unknown = set(basic) - _BASIC_NAMES
    if unknown:
        raise ValueError("record.basic has unsupported fields: " + ", ".join(sorted(map(str, unknown))))
    row = []
    for field in BASIC_FIELDS:
        value = basic.get(field.name)
        if value is None:
            row.extend("" for _ in field.columns)
        elif field.dtype == "string":
            if not isinstance(value, str):
                raise ValueError(f"basic.{field.name} must be a string or None")
            row.append(value)
        elif field.components:
            if isinstance(value, np.ndarray):
                if value.shape != (len(field.components),):
                    raise ValueError(f"basic.{field.name} must have shape ({len(field.components)},)")
                components = value.tolist()
            elif isinstance(value, (list, tuple)):
                components = value
            else:
                raise ValueError(f"basic.{field.name} must be a vector or None")
            if len(components) != len(field.components):
                raise ValueError(f"basic.{field.name} must have length {len(field.components)}")
            row.extend(_numeric_cell(component, field.name) for component in components)
        else:
            row.append(_numeric_cell(value, field.name))
    return _csv_line(row)


class RunRecorder:
    def __init__(self, directory, config, metadata):
        # Validate serializability/config before creating a partially initialized
        # run directory. JSON disallows NaN/infinity everywhere, including metadata.
        config_json = dump_json(config) + "\n"
        metadata_json = dump_json(metadata) + "\n"
        schema_json = dump_json(telemetry_schema()) + "\n"
        interval = config["logging"]["flush_every_n_records"]
        if isinstance(interval, (bool, np.bool_)) or not isinstance(interval, (int, np.integer)) or interval < 1:
            raise ValueError("flush_every_n_records must be a positive integer")
        label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self.path = Path(directory).expanduser() / label
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "config.json").write_text(config_json, encoding="utf-8")
        (self.path / "metadata.json").write_text(metadata_json, encoding="utf-8")
        (self.path / "telemetry_schema.json").write_text(schema_json, encoding="utf-8")
        self._streams = ExitStack()
        try:
            self._stream = self._streams.enter_context(
                (self.path / "telemetry.jsonl").open("w", encoding="utf-8", newline="")
            )
            self._csv_stream = self._streams.enter_context(
                (self.path / "basic.csv").open("w", encoding="utf-8", newline="")
            )
            self._csv_stream.write(_csv_line(BASIC_CSV_COLUMNS))
            self._csv_stream.flush()
        except BaseException:
            self._streams.close()
            raise
        self._flush_interval = int(interval)
        self._count = 0
        self._closed = False
        self._failed = False

    def write(self, record):
        if self._closed:
            raise ValueError("Cannot write to a closed RunRecorder")
        if self._failed:
            raise RuntimeError("Recorder had an I/O failure; close it before recording a new run")
        # Finish both conversions first. A malformed basic field must not leave
        # a JSONL record without its matching CSV row, nor the reverse.
        json_line = dump_json(record) + "\n"
        basic = record.get("basic") if isinstance(record, Mapping) else None
        csv_line = None if basic is None else _basic_csv_row(basic)
        try:
            self._stream.write(json_line)
            if csv_line is not None:
                self._csv_stream.write(csv_line)
            self._count += 1
            if self._count % self._flush_interval == 0:
                self.flush()
        except (OSError, ValueError):
            # Cross-file writes cannot be atomic on an arbitrary filesystem.
            # Stop after an I/O failure rather than knowingly continue divergent logs.
            self._failed = True
            raise

    def flush(self):
        """Flush both streams so callers can inspect a live run consistently."""
        if self._closed:
            raise ValueError("Cannot flush a closed RunRecorder")
        try:
            self._stream.flush()
            self._csv_stream.flush()
        except (OSError, ValueError):
            self._failed = True
            raise

    def close(self):
        """Flush/close both streams; safe to call repeatedly and from a context."""
        if not self._closed:
            self._closed = True
            self._streams.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("Cannot enter a closed RunRecorder")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
