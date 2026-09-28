"""Figures are drawn from basic.csv alone and never change the data files."""

from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from isaac_drone.analysis.plots import latest_run, load_basic_csv, phase_starts, plot_run
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench
from isaac_drone.telemetry import RunRecorder
from isaac_drone.telemetry.measurements import build_basic_record

HELIX = {"kind": "helix", "spiral": {"start_delay_s": 0.05, "takeoff_duration_s": 0.05, "spiral_duration_s": 0.1}}
FIGURES = {
    "trajectory_3d.png",
    "position.png",
    "velocity.png",
    "acceleration.png",
    "attitude.png",
    "angular_velocity.png",
    "motor_speed.png",
    "rotor_thrust.png",
    "force.png",
    "torque.png",
}


def recorded_run(tmp_path, steps=40):
    mass = MassProperties(2.0, np.diag([0.1, 0.12, 0.15]), np.zeros(3))
    config = {"logging": {"flush_every_n_records": 10}, "trajectory": HELIX}
    with RunRecorder(tmp_path, config, {}) as recorder:
        for index in range(steps):
            t0, t1 = index * 0.005, (index + 1) * 0.005
            before = VehicleState(t0, [np.cos(t0), np.sin(t0), t0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
            yaw = 3.1 + t1  # crosses +180 deg, which the plot must not draw as a jump
            after = VehicleState(
                t1,
                [np.cos(t1), np.sin(t1), t1],
                [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                [0, 0, 0.01 * index],
                [0, 0, 0.1],
            )
            telemetry = {
                "motor_speed_rps": [100 + index] * 4,
                "commanded_thrust_n": [5] * 4,
                "applied_thrust_n": [4.9] * 4,
                "motor_wrench_b": [0, 0, 19.6, 0, 0, 0],
                "motor_speed_source": "synthetic",
            }
            basic = build_basic_record(
                before,
                after,
                TrajectorySetpoint([1, 0, t1]),
                t0,
                mass,
                telemetry,
                Wrench([0, 0, 19.62], [0, 0, 0]),
                Wrench.zero(),
                Wrench.zero(),
                desired_rotation_w=np.eye(3),
                desired_angular_velocity_d=[0, 0, 0],
                motor_command_rps=[110] * 4,
            )
            recorder.write({"step": index, "basic": basic})
        return recorder.path


def test_plot_run_writes_every_figure_and_leaves_data_untouched(tmp_path):
    run = recorded_run(tmp_path)
    before = {name: (run / name).read_bytes() for name in ("basic.csv", "telemetry.jsonl", "config.json")}
    paths = plot_run(run)
    assert {path.name for path in paths} == FIGURES
    assert all(path.parent == run / "plots" and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for path in paths)
    assert {name: (run / name).read_bytes() for name in before} == before
    assert latest_run(tmp_path) == run


def test_blank_cells_are_nan_and_text_columns_are_skipped(tmp_path):
    data = load_basic_csv(recorded_run(tmp_path, steps=3) / "basic.csv")
    assert np.isnan(data["acceleration_native_w_x_m_s2"]).all()
    assert "motor_speed_source" not in data
    np.testing.assert_allclose(data["motor_speed_error_BL_rps"], [10, 9, 8])


def test_runs_recorded_before_error_columns_still_plot(tmp_path):
    # Older basic.csv files have targets and actuals but no error/attitude-angle columns.
    columns = ["sample_time_s", "quaternion_wxyz_w", "quaternion_wxyz_x", "quaternion_wxyz_y", "quaternion_wxyz_z"]
    for axis in "xyz":
        columns += [
            f"position_w_{axis}_m",
            f"target_position_w_{axis}_m",
            f"velocity_w_{axis}_m_s",
            f"target_velocity_w_{axis}_m_s",
        ]
    with (tmp_path / "basic.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(5):
            writer.writerow([index * 0.005, 1, 0, 0, 0] + [index * 0.01] * 12)
    assert {path.name for path in plot_run(tmp_path)} == FIGURES


def test_phase_markers_follow_the_helix_config():
    assert phase_starts({"trajectory": HELIX}) == [(0.0, "delay"), (0.05, "takeoff"), (0.1, "helix"), (0.2, "hold")]
    no_delay = {"kind": "helix", "spiral": {**HELIX["spiral"], "start_delay_s": 0}}
    assert [name for _, name in phase_starts({"trajectory": no_delay})] == ["takeoff", "helix", "hold"]
    assert phase_starts({"trajectory": {"kind": "hold"}}) == []
    # Durations computed at reset: planned boundaries come from metadata, or no markers at all.
    computed = {"kind": "helix", "spiral": {**HELIX["spiral"], "takeoff_duration_s": None}}
    assert phase_starts({"trajectory": computed}) == []
    planned = {"trajectory_feasibility": {"phase_boundaries_s": [0, 2, 5.5, 31]}}
    assert phase_starts({"trajectory": computed}, planned) == [
        (0.0, "delay"),
        (2.0, "takeoff"),
        (5.5, "helix"),
        (31.0, "hold"),
    ]


def test_empty_run_is_rejected(tmp_path):
    (tmp_path / "basic.csv").write_text("sample_time_s\n")
    with pytest.raises(ValueError, match="no data rows"):
        plot_run(tmp_path)
