"""Sensors and the estimation state provider inside the control loop, the records and the derived files."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_golden_regression import assert_matches_golden, fly

from isaac_drone.config import DEFAULT_SENSORS, HELIX_CONFIG, ConfigurationError, load_config, validate_config
from isaac_drone.core.rotations import matrix_to_quaternion, rotation_exp
from isaac_drone.core.types import VehicleState
from isaac_drone.estimation import TruthPassthrough, attitude_errors, build_state_provider, estimation_errors
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.sim.synthetic import SyntheticBackend, run_simulation

QUIET = ["effects.ou_wrench.enabled=false"]


def steps(config, count, state_provider=None):
    plant = SyntheticBackend(config)
    loop = MotionControlLoop(config, plant, state_provider=state_provider)
    loop.reset()
    records = []
    for _ in range(count):
        loop.prepare_step()
        plant.step()
        records.append(loop.finish_step())
    return loop, records


def test_closed_loop_passthrough_reproduces_the_golden_truth_flights():
    overrides = ["estimation.mode=closed_loop", "estimation.kind=passthrough"]
    rows, _, record = fly("hold", 1000, overrides=overrides)
    assert_matches_golden("hold", rows)
    assert record["state_source"] == "estimate:passthrough"
    rows, _, record = fly("helix", 800, overrides=overrides)
    assert_matches_golden("helix", rows)
    assert record["estimation"]["used_by_controller"] and record["estimation"]["error"]["position_norm_m"] == 0.0


def test_enabling_sensors_does_not_change_the_flight():
    config = load_config(overrides=QUIET)
    _, with_sensors = steps(config, 80)
    _, without = steps(load_config(overrides=[*QUIET, "sensors.enabled=false"]), 80)
    for a, b in zip(with_sensors, without):
        np.testing.assert_array_equal(a["basic"]["position_w_m"], b["basic"]["position_w_m"])
        np.testing.assert_array_equal(a["requested_wrench_b"], b["requested_wrench_b"])
    assert without[-1]["sensors"] is None and with_sensors[-1]["sensors"]["imu"] is not None
    # Disturbances draw from SeedSequence(seed); the sensors from a different domain, so neither shifts the other.
    noisy = load_config()
    _, loud = steps(noisy, 40)
    _, loud_without = steps(load_config(overrides=["sensors.enabled=false"]), 40)
    np.testing.assert_array_equal(loud[-1]["external_wrench_b"], loud_without[-1]["external_wrench_b"])


def test_each_mode_records_its_state_source_and_estimation_fields():
    truth_loop, truth = steps(load_config(overrides=QUIET), 5)
    assert truth[-1]["state_source"] == "simulation_truth" and truth[-1]["estimation"] is None
    assert (
        truth_loop.state_provider is truth_loop.backend and truth_loop.run_metadata()["estimation"]["mode"] == "truth"
    )
    shadow_loop, shadow = steps(load_config(overrides=[*QUIET, "estimation.mode=shadow"]), 5)
    record = shadow[-1]["estimation"]
    assert shadow[-1]["state_source"] == "simulation_truth" and not record["used_by_controller"]
    assert record["kind"] == "passthrough" and record["mode"] == "shadow"
    assert record["error"]["position_norm_m"] == 0.0 and record["error"]["tilt_error_rad"] == 0.0
    assert record["covariance_diag"] is None and record["nis"] is None
    np.testing.assert_allclose(
        record["control_error"]["position_w_m"], shadow[-1]["basic"]["position_error_w_m"], atol=1e-12
    )
    assert isinstance(shadow_loop.state_provider, TruthPassthrough) and shadow_loop.state_provider.updates == 5
    closed_loop, closed = steps(load_config(overrides=[*QUIET, "estimation.mode=closed_loop"]), 5)
    assert closed[-1]["state_source"] == "estimate:passthrough" and closed[-1]["estimation"]["used_by_controller"]
    assert closed_loop.run_metadata()["state_source"] == "estimate:passthrough"
    assert closed_loop.run_metadata()["sensors"]["sensors"]["gps"]["profile"] == "rtk"


class Shifted:
    """Legacy injected provider without update(): the controller still flies on it."""

    def __init__(self, backend):
        self.backend = backend

    def reset(self):
        pass

    def read_state(self, time_s):
        return replace(
            self.backend.read_state(time_s), position_w=self.backend.read_state(time_s).position_w + [0, 0, 0.1]
        )


def test_injected_provider_keeps_closed_loop_semantics_and_is_recorded():
    config = load_config(overrides=QUIET)
    plant = SyntheticBackend(config)
    loop = MotionControlLoop(config, plant, state_provider=Shifted(plant))
    loop.reset()
    loop.prepare_step()
    plant.step()
    record = loop.finish_step()
    assert record["state_source"] == "injected_state_provider"
    assert record["estimation"]["kind"] == "injected:Shifted" and record["estimation"]["mode"] == "injected"
    np.testing.assert_allclose(record["estimation"]["error"]["position_w_m"], [0, 0, -0.1], atol=1e-12)


def test_provider_must_answer_at_the_queried_time():
    config = load_config(overrides=QUIET)
    plant = SyntheticBackend(config)

    class Stale(TruthPassthrough):
        def read_state(self, time_s):
            return self.backend.read_state(max(time_s - 0.005, 0.0))

    loop = MotionControlLoop(config, plant, state_provider=Stale(plant))
    loop.reset()
    loop.prepare_step()
    plant.step()
    with pytest.raises(ValueError, match="queried loop time"):
        loop.finish_step()


def test_configs_without_the_new_sections_stay_valid_and_bad_sections_fail():
    config = load_config()
    del config["sensors"], config["estimation"]
    validate_config(config)
    # A configuration from before the sections existed records no readings, whatever its physics step.
    legacy = load_config()
    del legacy["sensors"]
    legacy["simulation"]["dt"] = 0.1
    validate_config(legacy)
    assert MotionControlLoop(load_config(overrides=["sensors.enabled=false"]), SyntheticBackend(legacy)).sensors is None
    assert DEFAULT_SENSORS["enabled"] is False and load_config()["sensors"]["enabled"] is True
    assert load_config()["sensors"] == {**DEFAULT_SENSORS, "enabled": True}
    for override in ("estimation.mode=open_loop", "estimation.kind=ekf15", "estimation.extra=1"):
        with pytest.raises(ConfigurationError, match="estimation"):
            load_config(overrides=[override])
    with pytest.raises(ConfigurationError, match="not implemented"):
        build_state_provider({"mode": "closed_loop", "kind": "mekf"}, None)


def test_error_conventions_are_ideal_minus_actual():
    truth = VehicleState(1.0, [1.0, 2.0, 3.0], [1, 0, 0, 0], [0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    yawed = matrix_to_quaternion(rotation_exp([0.0, 0.0, 0.1]))
    estimate = VehicleState(1.0, [1.1, 2.0, 3.0], yawed, [0.1, 0.0, 0.3], [0.0, 0.02, 0.03])
    errors = estimation_errors(truth, estimate)
    np.testing.assert_allclose(errors["position_w_m"], [-0.1, 0.0, 0.0])
    np.testing.assert_allclose(errors["velocity_w_m_s"], [0.0, 0.2, 0.0])
    np.testing.assert_allclose(errors["angular_velocity_b_rad_s"], [0.01, 0.0, 0.0])
    assert errors["yaw_error_rad"] == pytest.approx(-0.1) and errors["tilt_error_rad"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(errors["attitude_error_rad"], [0.0, 0.0, -0.1], atol=1e-12)
    tilted = matrix_to_quaternion(rotation_exp([0.2, 0.0, 0.0]))
    assert attitude_errors([1, 0, 0, 0], tilted)["tilt_error_rad"] == pytest.approx(0.2)


def test_run_directory_carries_sensor_records_metrics_card_and_figures(tmp_path):
    config = load_config(
        HELIX_CONFIG, overrides=["simulation.duration_s=0.6", "completion=null", "estimation.mode=shadow"]
    )
    figures = pytest.importorskip("matplotlib") is not None
    result = run_simulation(config, figures=figures, log=lambda *_: None, log_root=tmp_path)
    records = [json.loads(line) for line in (result.run_dir / "telemetry.jsonl").read_text().splitlines()]
    step_records = [record for record in records if record.get("basic") is not None]
    assert all("sensors" in record and "estimation" in record for record in step_records)
    gps = [record["sensors"]["gps"] for record in step_records if record["sensors"]["gps"] is not None]
    assert gps and gps[0]["time_s"] == pytest.approx(0.1) and len(gps[0]["position_w_m"]) == 3
    metadata = json.loads((result.run_dir / "metadata.json").read_text())
    assert metadata["state_source"] == "simulation_truth" and metadata["sensors"]["sensors"]["imu"]["rate_hz"] == 200
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics["overall"]["estimation_position_error_rms_m"] == 0.0
    assert metrics["overall"]["control_position_error_rms_m"] == pytest.approx(
        metrics["overall"]["position_error_rms_m"]
    )
    assert metrics["phases"]["spin_up"]["estimation_velocity_error_rms_m_s"] == 0.0
    assert metrics["run"]["estimation_mode"] == "shadow"
    card = (result.run_dir / "README.md").read_text(encoding="utf-8")
    assert "| State estimation | shadow: passthrough runs and is recorded, controller flies on truth |" in card
    sensors_row = "IMU 200 Hz (vibration), magnetometer 50 Hz, barometer 50 Hz, GPS 10 Hz (rtk, 0.12 s delay)"
    assert f"| Sensors (synthetic readings) | {sensors_row} |" in card
    assert "| Estimation error RMS, position [m] |" in card
    names = {path.name for path in (result.run_dir / "plots").glob("*.png")}
    expected = {
        "sensors_accelerometer.png",
        "sensors_gyroscope.png",
        "sensors_magnetometer.png",
        "sensors_barometer.png",
        "sensors_gps_position.png",
        "sensors_gps_velocity.png",
        "estimation_position.png",
        "estimation_velocity.png",
        "estimation_attitude.png",
        "estimation_angular_velocity.png",
    }
    assert expected <= names
    truth_only = run_simulation(
        load_config(overrides=["simulation.duration_s=0.05", "sensors.enabled=false"]),
        figures=figures,
        log=lambda *_: None,
        log_root=tmp_path,
    )
    assert not {path.name for path in (truth_only.run_dir / "plots").glob("*.png")} & expected
    assert "| Estimation error RMS" not in (truth_only.run_dir / "README.md").read_text(encoding="utf-8")
