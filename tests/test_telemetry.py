"""Recorder checks for physical field provenance, missing data and paired output."""
from __future__ import annotations

import csv
import json
import numpy as np
import pytest

from isaac_drone.telemetry.measurements import build_basic_record
from isaac_drone.telemetry import BASIC_CSV_COLUMNS, RunRecorder
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench


def config(flush=1):
    return {"logging": {"flush_every_n_records": flush}}


def record():
    before = VehicleState(0, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    after = VehicleState(0.1, [0.01, 0.02, 0.03], [1, 0, 0, 0], [0.2, 0.4, 0.6], [0.1, 0.2, 0.3])
    reference = TrajectorySetpoint(position_w=np.array([1, 2, 3]))
    mass = MassProperties(2, np.diag([0.1, 0.12, 0.15]), np.zeros(3))
    backend = {
        "motor_speed_rps": np.array([10, 20, 30, 40]),
        "motor_speed_rpm": np.array([600, 1200, 1800, 2400]),
        "motor_speed_rad_s": 2*np.pi*np.array([10, 20, 30, 40]),
        "motor_speed_source": 'native "thruster" state\n合成测试数据',
        "commanded_thrust_n": np.array([1, 2, 3, 4]),
        "applied_thrust_n": np.array([0.9, 1.9, 2.9, 3.9]),
        "linear_acceleration_com_w_m_s2": np.array([2.1, 4.1, 6.1]),
        "linear_acceleration_source": "synthetic native solver fixture",
        "motor_wrench_b": np.array([0, 0, 10, 1, 2, 3]),
        "total_wrench_b": np.array([1, 1, 11, 2, 3, 4]),
        "has_submitted_wrench": True,
    }
    command = Wrench([0, 0, 12], [2, 3, 4])
    allocated = Wrench([0, 0, 11], [1, 2, 3])
    disturbance = Wrench([1, 1, 1], [1, 1, 1])
    basic = build_basic_record(before, after, reference, 0, mass, backend,
                               command, allocated, disturbance)
    return {"step": 0, "basic": basic, "legacy_payload": {"native": np.array([1, 2, 3])}}


def csv_rows(path):
    with (path / "basic.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return reader.fieldnames, rows


def json_rows(path):
    return [json.loads(line) for line in (path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]


def test_complete_runtime_record_exports_csv_and_inspectable_schema(tmp_path):
    original = record()
    with RunRecorder(tmp_path, config(), {"mass_kg": np.float64(2)}) as recorder:
        recorder.write(original)
        path = recorder.path
        # Automatic flushing makes both current outputs visible while still open.
        header, rows = csv_rows(path)
        assert len(rows) == len(json_rows(path)) == 1
    row = rows[0]
    assert header == list(BASIC_CSV_COLUMNS)
    assert len(header) == len(set(header))
    assert float(row["sample_time_s"]) == 0.1
    assert float(row["position_w_z_m"]) == 0.03
    assert float(row["velocity_w_y_m_s"]) == 0.4
    assert float(row["acceleration_w_x_m_s2"]) == 2
    assert float(row["acceleration_native_w_x_m_s2"]) == 2.1
    assert row["acceleration_native_source"] == "synthetic native solver fixture"
    assert float(row["quaternion_wxyz_w"]) == 1
    assert float(row["angular_velocity_w_z_rad_s"]) == 0.3
    assert float(row["motor_speed_BL_rps"]) == 10
    assert float(row["motor_speed_FR_rpm"]) == 2400
    assert float(row["motor_speed_FL_rad_s"]) == pytest.approx(60*np.pi)
    assert float(row["motor_thrust_command_BR_n"]) == 2
    assert float(row["motor_thrust_applied_BR_n"]) == 1.9
    assert float(row["force_command_b_z_n"]) == 12
    assert float(row["force_motor_b_z_n"]) == 10
    assert float(row["torque_submitted_b_z_nm"]) == 4
    assert float(row["force_net_w_x_n"]) == 4
    assert float(row["position_error_w_x_m"]) == 0.99
    assert row["motor_speed_source"] == original["basic"]["motor_speed_source"]
    assert json_rows(path)[0]["legacy_payload"]["native"] == [1, 2, 3]
    schema = json.loads((path / "telemetry_schema.json").read_text())
    assert schema["files"]["basic.csv"]["column_order"] == header
    assert schema["conventions"]["rotor_order"] == ["BL", "BR", "FL", "FR"]
    assert set(schema["fields"]) == set(original["basic"])
    assert len(schema["columns"]) == len(header)
    for field in schema["fields"].values():
        assert field["unit"] and field["frame"] and field["source"] and field["sampling"]
    assert schema["fields"]["torque_net_w_nm"]["unit"] == "N m"
    assert "angular_momentum" in schema["fields"]["torque_net_w_nm"]["source"]
    assert schema["fields"]["force_command_b_n"]["frame"] == "body at step_start_time_s"
    assert schema["fields"]["angular_acceleration_b_rad_s2"]["frame"] == "body at sample_time_s"
    assert "not IMU" in schema["fields"]["acceleration_w_m_s2"]["description"]
    assert "not an encoder" in schema["fields"]["motor_speed_rps"]["description"]
    assert "After the native motor update" in schema["fields"]["motor_speed_rps"]["sampling"]
    assert "resolved configured geometry" in schema["fields"]["force_motor_b_n"]["source"]


def test_legacy_records_and_events_do_not_create_csv_rows(tmp_path):
    events = [{"old_state": [1, 2, 3]}, {"event": "finished"}, {"basic": None, "event": "aborted"}]
    with RunRecorder(tmp_path, config(), {}) as recorder:
        for event in events:
            recorder.write(event)
        path = recorder.path
    header, rows = csv_rows(path)
    assert header == list(BASIC_CSV_COLUMNS)
    assert rows == []
    assert json_rows(path) == events
    assert (path / "telemetry_schema.json").is_file()


def test_missing_optional_or_legacy_fields_are_blank_not_zero(tmp_path):
    partial = {"basic": {"sample_time_s": 2, "motor_speed_rps": None,
                         "position_w_m": [1, None, 3], "force_motor_b_n": None}}
    with RunRecorder(tmp_path, config(), {}) as recorder:
        recorder.write(partial)
        path = recorder.path
    _, rows = csv_rows(path)
    row = rows[0]
    assert row["position_w_x_m"] == "1.0"
    assert row["position_w_y_m"] == ""
    assert row["target_position_w_x_m"] == ""
    assert row["motor_speed_BL_rps"] == ""
    assert row["motor_speed_BL_rpm"] == ""
    assert row["force_motor_b_z_n"] == ""
    assert row["motor_speed_source"] == ""
    assert json_rows(path) == [partial]


@pytest.mark.parametrize("bad_record", [
    {"basic": {"position_w_m": [1, 2]}},
    {"basic": {"position_w_m": [[1], [2], [3]]}},
    {"basic": {"position_w_m": np.zeros((1, 3))}},
    {"basic": {"motor_speed_rps": [1, 2, 3]}},
    {"basic": {"position_w_m": [1, True, 3]}},
    {"basic": {"position_w_m": [1, "2", 3]}},
    {"basic": {"position_w_m": [1, float("nan"), 3]}},
    {"basic": {"sample_time_s": float("inf")}},
    {"basic": {"motor_speed_rpm": [1, 2, -np.inf, 4]}},
    {"basic": {"interval_dt_s": False}},
    {"basic": {"sample_time_s": "1.0"}},
    {"basic": {"motor_speed_source": 4}},
    {"basic": {"position_word_m": [1, 2, 3]}},
    {"basic": []},
    {"basic": {"sample_time_s": 1}, "outside_basic": np.nan},
    {"basic": {"sample_time_s": 1}, "unserializable": {1, 2}},
])
def test_bad_record_is_rejected_before_either_file_changes(tmp_path, bad_record):
    with RunRecorder(tmp_path, config(flush=100), {}) as recorder:
        recorder.write({"basic": {"sample_time_s": 0.1}})
        recorder.flush()
        before = [(recorder.path / name).read_bytes() for name in ("telemetry.jsonl", "basic.csv")]
        with pytest.raises((ValueError, TypeError)):
            recorder.write(bad_record)
        recorder.flush()
        after = [(recorder.path / name).read_bytes() for name in ("telemetry.jsonl", "basic.csv")]
        assert before == after
        assert recorder._count == 1
        # Validation failure does not poison the recorder or advance its cadence.
        recorder.write({"basic": {"sample_time_s": 0.2}})
        path = recorder.path
    assert len(json_rows(path)) == len(csv_rows(path)[1]) == 2


def test_context_exception_closes_both_files_and_close_is_idempotent(tmp_path):
    recorder = RunRecorder(tmp_path, config(flush=100), {})
    with pytest.raises(RuntimeError, match="flight interrupted"):
        with recorder:
            recorder.write(record())
            raise RuntimeError("flight interrupted")
    assert len(json_rows(recorder.path)) == len(csv_rows(recorder.path)[1]) == 1
    assert recorder._stream.closed and recorder._csv_stream.closed
    recorder.close()
    with pytest.raises(ValueError, match="closed"):
        recorder.write(record())
    with pytest.raises(ValueError, match="closed"):
        recorder.flush()


@pytest.mark.parametrize("flush", [0, -1, True, 1.5, "1"])
def test_invalid_flush_configuration_creates_no_run_directory(tmp_path, flush):
    with pytest.raises(ValueError):
        RunRecorder(tmp_path, config(flush), {})
    assert list(tmp_path.iterdir()) == []


def test_nonfinite_metadata_is_rejected_before_creating_a_run(tmp_path):
    with pytest.raises(ValueError):
        RunRecorder(tmp_path, config(), {"mass_kg": np.nan})
    assert list(tmp_path.iterdir()) == []


def test_empty_run_still_has_csv_header_and_schema(tmp_path):
    with RunRecorder(tmp_path, config(), {}) as recorder:
        path = recorder.path
        assert csv_rows(path) == (list(BASIC_CSV_COLUMNS), [])
    assert json_rows(path) == []
    assert json.loads((path / "telemetry_schema.json").read_text())["schema_version"] == 1
