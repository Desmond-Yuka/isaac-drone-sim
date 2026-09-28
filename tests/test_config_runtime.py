"""Configuration failure modes and physics/control scheduling without Isaac Sim."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from isaac_drone.config import (
    HELIX_CONFIG,
    ConfigurationError,
    assert_flight_ready,
    load_config,
    resolve_asset,
    validate_config,
)
from isaac_drone.trajectories.timing import HermiteTimeLaw
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.telemetry import RunRecorder
from isaac_drone.trajectories import TRAJECTORIES
from isaac_drone.core.types import MassProperties, VehicleState, Wrench


def test_default_config_has_explicit_source_and_local_usd():
    cfg = load_config()
    assert cfg["vehicle"]["rotor_direction_source"] == "simulation_diagonal_pairs"
    assert cfg["vehicle"]["rotor_directions"] == [-1, 1, 1, -1]
    assert resolve_asset(cfg).is_file()
    assert not cfg["power"]["enabled"]
    assert cfg["power"]["capacity_ah"] is None


@pytest.mark.parametrize("change", [
    lambda c: c["simulation"].update(dt=0),
    lambda c: c["simulation"].update(control_decimation=1.5),
    lambda c: c["simulation"].update(gravity=[0, 0, 0]),
    lambda c: c["controller"].update(position_kp=[1, np.nan, 2]),
    lambda c: c["vehicle"]["thrusters"].update(tau_dec_range=[0, 0]),
    lambda c: c["vehicle"]["thrusters"].update(thrust_range=[10, 0]),
    lambda c: c["vehicle"]["initial_state"].update(quaternion_wxyz=[0, 0, 0, 0]),
    lambda c: c["simulation"]["native_overrides"].update(dt=0.1),
    lambda c: c["vehicle"]["native_overrides"].update(actuators={}),
    lambda c: c["controller"].update(position_kkp=[1, 1, 1]),
    lambda c: c["controller"].update(kind="unknown_controller"),
    lambda c: c["limits"].update(max_tilt_rad=2.0),
    lambda c: c["trajectory"].update(kind="helix"),
    lambda c: c.update(schema_version=1),
])
def test_bad_configuration_fails_instead_of_silent_default(change):
    cfg = load_config()
    change(cfg)
    with pytest.raises((ConfigurationError, ValueError)):
        validate_config(cfg)


def test_duplicate_yaml_keys_fail(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 2\nschema_version: 2\n")
    with pytest.raises(ConfigurationError, match="Duplicate"):
        load_config(path)


def test_lfs_pointer_fails(tmp_path):
    path = tmp_path / "robot.usd"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
    cfg = load_config()
    cfg["vehicle"]["asset_path"] = str(path)
    with pytest.raises(ConfigurationError, match="LFS"):
        resolve_asset(cfg)


def test_reference_directions_cannot_control_actual_geometry():
    cfg = load_config()
    matrix = FakeBackend().allocation_matrix_b.copy()
    assert_flight_ready(cfg, matrix)
    matrix[-1] = 0.07*np.array([-1, 1, -1, 1])
    with pytest.raises(ConfigurationError, match="rank"):
        assert_flight_ready(cfg, matrix)


class FakeBackend:
    """Only a scheduling double; it deliberately does not simulate flight."""
    allocation_matrix_b = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1],
                                    [.1, -.1, .1, -.1], [.1, .1, -.1, -.1],
                                    [-.07, .07, .07, -.07]])

    def __init__(self):
        self.applied = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.applied.clear()

    def mass_properties(self):
        return MassProperties(1.24, np.diag([.013, .014, .013]), np.array([.01, .02, .03]))

    def read_state(self, time_s):
        return VehicleState(time_s, np.array([.1, -.2, 1.1]), np.array([1., 0, 0, 0]), np.zeros(3), np.zeros(3))

    def motor_parameters(self):
        return {"sampled_kf": np.full(4, 1e-5), "thrust_min_n": np.full(4, .1),
                "thrust_max_n": np.full(4, 10.), "rps_min": np.full(4, 100.),
                "rps_max": np.full(4, 1000.)}

    def thrust_to_motor_speeds(self, thrust):
        return np.sqrt(thrust/1e-5)

    def apply_motor_speeds(self, target_rps, external):
        self.apply(1e-5*target_rps**2, external)

    def apply(self, thrusts_n, external_wrench):
        self.applied.append((thrusts_n.copy(), external_wrench))

    def telemetry(self):
        return {"applied_thrust_n": self.applied[-1][0].tolist() if self.applied else [0]*4}


def test_control_decimation_preserves_each_physics_motor_and_effect_tick():
    cfg = load_config()
    backend = FakeBackend()
    loop = MotionControlLoop(cfg, backend)
    loop.reset()
    calls = []
    compute = loop.controller.compute
    loop.controller.compute = lambda *args: (calls.append(args[2]) or compute(*args))
    for _ in range(7):
        loop.prepare_step()
        loop.finish_step()
    assert len(backend.applied) == 7
    assert len(calls) == 4
    assert calls == [cfg["simulation"]["dt"]*cfg["simulation"]["control_decimation"]]*4
    np.testing.assert_allclose(loop.setpoint.position_w, [.1, -.2, 1.1])
    assert loop.time_s == 7*cfg["simulation"]["dt"]
    loop.reset()
    assert backend.reset_count == 2
    assert loop.time_s == 0


def test_loop_pairing_is_explicit_and_kind_switch_replaces_parameters():
    cfg = load_config()
    loop = MotionControlLoop(cfg, FakeBackend())
    with pytest.raises(RuntimeError):
        loop.prepare_step()
    loop.reset()
    with pytest.raises(RuntimeError):
        loop.finish_step()
    loop.prepare_step()
    with pytest.raises(RuntimeError):
        loop.prepare_step()
    helix = load_config(HELIX_CONFIG)["trajectory"]
    assert cfg["trajectory"] == {"kind": "hold", "position_w_m": None, "yaw_rad": None}
    trajectory = TRAJECTORIES.build(helix, "trajectory")
    trajectory.reset(loop.backend.read_state(0))
    trajectory.set_time_laws(takeoff=HermiteTimeLaw(1.), helix=HermiteTimeLaw(5.))
    np.testing.assert_allclose(trajectory.sample(trajectory.motion_duration_s).position_w,
                               trajectory.endpoint_position_w)
    assert trajectory.endpoint_position_w[2] == pytest.approx(loop.backend.read_state(0).position_w[2] + 6)


def test_recording_is_strict_json_and_records_actual_config(tmp_path):
    cfg = load_config()
    with RunRecorder(tmp_path, cfg, {"mass_kg": np.float64(1.24)}) as recorder:
        recorder.write({"wrench": Wrench.zero().vector})
        path = recorder.path
        with pytest.raises(ValueError):
            recorder.write({"invalid": np.nan})
    assert json.loads((path/"metadata.json").read_text())["mass_kg"] == 1.24
    assert json.loads((path/"telemetry.jsonl").read_text())["wrench"] == [0]*6


class SpeedOnlyGroundBackend(FakeBackend):
    """No plant simulation: verifies the public runtime command boundary only."""
    def ground_start_position(self, ground_z_m, clearance_m):
        self.ground_request = (ground_z_m, clearance_m)
        return np.array([.1, -.2, ground_z_m+.25+clearance_m])

    def reset(self, initial_position_w=None):
        super().reset()
        self.start = initial_position_w.copy()
        self.speed_commands = []

    def read_state(self, time_s):
        return replace(super().read_state(time_s), position_w=self.start.copy())

    def apply_motor_speeds(self, target_rps, external):
        self.speed_commands.append(target_rps.copy())
        self.applied.append((1e-5*target_rps**2, external))


def test_helix_starts_with_zero_motor_speed_and_retains_measured_ground_origin():
    cfg = load_config(HELIX_CONFIG)
    backend = SpeedOnlyGroundBackend()
    loop = MotionControlLoop(cfg, backend)
    loop.reset()
    assert backend.ground_request == (0., .001)
    first = loop.prepare_step()
    np.testing.assert_array_equal(first["motor_command_rps"], np.zeros(4))
    np.testing.assert_array_equal(loop.setpoint.position_w, backend.start)
    assert first["mission_phase"] == "spin_up"
    after = loop.finish_step()
    assert not after["mission"]["achieved"]
    second = loop.prepare_step()
    assert np.all(second["motor_command_rps"] >= backend.motor_parameters()["rps_min"])
    assert second["startup_fraction"] > 0
    assert np.all(second["motor_command_thrust_n"] < second["allocated_thrust_n"])
    assert len(backend.speed_commands) == 2
    loop.finish_step()
    # The preset's null durations are planned for minimum time at reset.
    plan = loop.feasibility["minimum_time_plan"]
    assert plan["planned_phases"] == ["takeoff", "helix"]
    assert loop.phase_schedule()[0] == (0.0, "spin_up")
    assert loop.phase_schedule()[1:] == loop.trajectory.phase_boundaries_s
    assert [name for _, name in loop.phase_schedule()] == ["spin_up", "takeoff", "helix", "hold"]
    end = loop.trajectory.phase_boundaries_s[-1][0]
    assert cfg["simulation"]["duration_s"] >= end
    short = load_config(HELIX_CONFIG)
    short["simulation"]["duration_s"] = end
    with pytest.raises(ValueError, match="dwell"):
        MotionControlLoop(short, SpeedOnlyGroundBackend()).reset()


def test_ground_launch_reference_anchors_to_truth_even_with_biased_estimator():
    backend = SpeedOnlyGroundBackend()
    class BiasedEstimate:
        def reset(self):
            pass
        def read_state(self, time_s):
            return replace(backend.read_state(time_s), position_w=backend.start+[1, 2, 3])
    loop = MotionControlLoop(load_config(HELIX_CONFIG), backend, state_provider=BiasedEstimate())
    loop.reset()
    record = loop.prepare_step()
    np.testing.assert_allclose(record["setpoint"]["position_w"], backend.start)
    np.testing.assert_allclose(record["state"]["position_w"], backend.start+[1, 2, 3])
