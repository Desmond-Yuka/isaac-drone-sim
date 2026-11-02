"""Sensor models: MATLAB imuSensor/altimeterSensor/gpsSensor statistics, mounting kinematics, rates and delays."""

from copy import deepcopy

import numpy as np
import pytest

from isaac_drone.config import DEFAULT_SENSORS, ConfigurationError, load_config, validate_config
from isaac_drone.core.rotations import matrix_to_quaternion, rotation_exp
from isaac_drone.core.types import MassProperties, VehicleState
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.sensors import (
    DecayingNoise,
    GaussMarkov,
    SensorSuite,
    StepTruth,
    TriaxialModel,
    TriaxialParams,
    delay_steps,
    misalignment_matrix,
    resolve_gps_parameters,
    sample_divisor,
    white_noise_std,
)
from isaac_drone.sensors.barometer import BarometerSensor
from isaac_drone.sensors.gps import GpsSensor
from isaac_drone.sensors.magnetometer import MagnetometerSensor
from isaac_drone.sim.synthetic import SyntheticBackend

DT = 0.005
GRAVITY = np.array([0.0, 0.0, -9.81])
MASS = MassProperties(1.24, np.diag([0.008, 0.009, 0.014]), np.array([0.006, -0.004, 0.002]))


def quiet_params(**overrides) -> dict:
    """accelparams-like section with every error source off, then ``overrides``."""
    section = {
        "measurement_range": None,
        "resolution": 0.0,
        "constant_bias": 0.0,
        "axes_misalignment": 0.0,
        "noise_density": 0.0,
        "bias_instability": 0.0,
        "bias_instability_time_constant_s": 100.0,
        "random_walk": 0.0,
        "noise_type": "double-sided",
        "temperature_bias": 0.0,
        "temperature_scale_factor": 0.0,
    }
    section.update(overrides)
    return section


def model(rate_hz=200.0, temperature_c=25.0, gyroscope=False, seed=0, **overrides) -> TriaxialModel:
    section = quiet_params(**overrides)
    if gyroscope:
        section.setdefault("acceleration_bias", 0.0)
    result = TriaxialModel(TriaxialParams.from_config(section, "test", gyroscope=gyroscope), rate_hz, temperature_c)
    result.reset(np.random.default_rng(seed))
    return result


def samples(sensor: TriaxialModel, count: int, true=None, **kwargs) -> np.ndarray:
    true = np.zeros(3) if true is None else true
    return np.array([sensor.measure(true, **kwargs) for _ in range(count)])


def allan_deviation(series: np.ndarray, dt: float, clusters) -> np.ndarray:
    """Overlapping Allan deviation of one channel at averaging times ``clusters * dt``."""
    result = []
    for m in clusters:
        n = len(series) // m * m
        means = series[:n].reshape(-1, m).mean(axis=1)
        result.append(np.sqrt(0.5 * np.mean(np.diff(means) ** 2)))
    return np.array(result)


def quiet_sensor_config() -> dict:
    """The default sensor section with every random term switched off."""
    section = deepcopy(DEFAULT_SENSORS)
    section["enabled"] = True
    for key in ("accelerometer", "gyroscope"):
        section["imu"][key].update(
            noise_density=0.0, bias_instability=0.0, random_walk=0.0, constant_bias=0.0, axes_misalignment=0.0
        )
        section["imu"][key]["resolution"] = 0.0
    section["imu"]["gyroscope"]["acceleration_bias"] = 0.0
    section["imu"]["vibration"]["enabled"] = False
    section["magnetometer"]["model"].update(noise_density=0.0, bias_instability=0.0, constant_bias=0.0)
    section["magnetometer"]["model"].update(axes_misalignment=0.0, resolution=0.0)
    section["barometer"].update(noise_density_m_sqrt_hz=0.0, bias_instability_m=0.0, constant_bias_m=0.0)
    section["gps"].update(horizontal_accuracy_m=0.0, vertical_accuracy_m=0.0, velocity_accuracy_m_s=0.0)
    return section


def rest_truth(time_s=1.0, quaternion=(1.0, 0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), pairing="step_start"):
    state = VehicleState(time_s, [1.0, 2.0, 3.0], quaternion, np.zeros(3), omega)
    after = VehicleState(time_s + DT, [1.0, 2.0, 3.0], quaternion, np.zeros(3), omega)
    return StepTruth(state, after, GRAVITY, pairing)


# --- noise processes -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "noise_type, expected", [("double-sided", 0.01 * np.sqrt(100.0)), ("single-sided", 0.01 * np.sqrt(200.0))]
)
def test_white_noise_follows_the_matlab_scaling_and_has_allan_slope_minus_half(noise_type, expected):
    assert white_noise_std(0.01, 200.0, noise_type)[0] == pytest.approx(expected)
    readings = samples(model(noise_density=0.01, noise_type=noise_type), 40000)
    assert np.abs(readings.mean(axis=0)).max() < 0.03 * expected
    np.testing.assert_allclose(readings.std(axis=0), expected, rtol=0.03)
    clusters = np.array([2, 4, 8, 16, 32])
    deviation = allan_deviation(readings[:, 0], DT, clusters)
    slope = np.polyfit(np.log(clusters * DT), np.log(deviation), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.06)


def test_random_walk_increments_follow_the_matlab_scaling_and_have_allan_slope_plus_half():
    sensor = model(random_walk=0.01)  # double-sided: q = 0.01 / sqrt(200 / 2)
    readings = samples(sensor, 40000)
    increments = np.diff(readings, axis=0)
    np.testing.assert_allclose(increments.std(axis=0), 0.01 / np.sqrt(100.0), rtol=0.03)
    clusters = np.array([2, 4, 8, 16, 32])
    slope = np.polyfit(np.log(clusters * DT), np.log(allan_deviation(readings[:, 1], DT, clusters)), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.06)


def test_gauss_markov_has_the_configured_stationary_std_and_correlation_time():
    process = GaussMarkov(1.0, 0.5)
    rng = np.random.default_rng(1)
    process.reset(rng)
    series = np.array([process.step(DT, rng) for _ in range(200000)])[:, 0]
    assert series.std() == pytest.approx(1.0, rel=0.05)
    lag = round(0.5 / DT)
    correlation = np.corrcoef(series[:-lag], series[lag:])[0, 1]
    assert correlation == pytest.approx(np.exp(-1.0), abs=0.05)
    bias = model(bias_instability=0.2, bias_instability_time_constant_s=0.5)
    readings = samples(bias, 100000)[:, 0]
    assert readings.std() == pytest.approx(0.2, rel=0.06)
    assert np.corrcoef(readings[:-lag], readings[lag:])[0, 1] == pytest.approx(np.exp(-1.0), abs=0.06)


def test_decay_recursion_is_white_at_zero_random_walk_at_one_and_stationary_std_in_between():
    rng = np.random.default_rng(2)
    white = DecayingNoise(0.3, 0.0, size=1)
    white.reset()
    series = np.array([white.step(rng)[0] for _ in range(50000)])
    assert series.std() == pytest.approx(0.3, rel=0.03)
    assert abs(np.corrcoef(series[:-1], series[1:])[0, 1]) < 0.02
    walk = DecayingNoise(0.3, 1.0, size=1)
    walk.reset()
    steps = np.array([walk.step(rng)[0] for _ in range(50000)])
    np.testing.assert_allclose(np.diff(steps).std(), 0.3, rtol=0.03)
    assert np.isinf(walk.stationary_std[0])
    markov = DecayingNoise(0.3, 0.9, size=1)
    markov.reset()
    series = np.array([markov.step(rng)[0] for _ in range(100000)])
    assert markov.stationary_std[0] == pytest.approx(0.3 / np.sqrt(1 - 0.81))
    assert series.std() == pytest.approx(markov.stationary_std[0], rel=0.05)


# --- imuSensor error chain -------------------------------------------------------------------


def test_misalignment_matrix_accepts_the_three_matlab_forms():
    np.testing.assert_array_equal(misalignment_matrix(0.0), 100.0 * np.eye(3))
    np.testing.assert_array_equal(misalignment_matrix(2.0), np.array([[100, 2, 2], [2, 100, 2], [2, 2, 100]]))
    np.testing.assert_array_equal(
        misalignment_matrix([1.0, 2.0, 3.0]), np.array([[100, 2, 3], [1, 100, 3], [1, 2, 100]])
    )
    matrix = np.arange(9.0).reshape(3, 3) + 100 * np.eye(3)
    np.testing.assert_array_equal(misalignment_matrix(matrix), matrix)
    with pytest.raises(ValueError):
        misalignment_matrix([1.0, 2.0])


def test_bulk_model_temperature_terms_g_sensitivity_saturation_and_quantization_are_deterministic():
    sensor = model(
        gyroscope=True,
        temperature_c=35.0,
        constant_bias=[0.1, 0.2, 0.3],
        axes_misalignment=[1.0, 2.0, 3.0],
        temperature_bias=[0.01, 0.02, 0.03],
        temperature_scale_factor=[0.0, 0.0, 10.0],
        acceleration_bias=[0.001, 0.0, 0.0],
        measurement_range=2.0,
        resolution=0.01,
    )
    true = np.array([1.0, -2.0, 3.0])
    specific_force = np.array([9.81, 0.0, 0.0])
    reading = sensor.measure(true, specific_force_s=specific_force)
    expected = misalignment_matrix([1.0, 2.0, 3.0]) / 100.0 @ true + [0.1, 0.2, 0.3]
    expected = expected + 10.0 * np.array([0.01, 0.02, 0.03]) + np.array([0.001, 0.0, 0.0]) * specific_force
    expected = expected * (1.0 + 10.0 / 100.0 * np.array([0.0, 0.0, 10.0]))
    expected = np.clip(expected, -2.0, 2.0)
    expected = 0.01 * np.round(expected / 0.01)
    np.testing.assert_allclose(reading, expected, atol=1e-12)
    assert reading[2] == pytest.approx(2.0)  # saturated at +MeasurementRange
    np.testing.assert_array_equal(sensor.measure(true, specific_force_s=specific_force), reading)
    with pytest.raises(ValueError, match="specific force"):
        sensor.measure(true)


def test_params_reject_bad_values():
    with pytest.raises(ValueError, match="unknown"):
        TriaxialParams.from_config(quiet_params(extra=1.0), "test")
    with pytest.raises(ValueError, match="noise_type"):
        TriaxialParams.from_config(quiet_params(noise_type="triple"), "test")
    with pytest.raises(ValueError, match="nonnegative"):
        TriaxialParams.from_config(quiet_params(noise_density=-1.0), "test")
    with pytest.raises(ValueError):
        TriaxialParams.from_config(quiet_params(bias_instability_time_constant_s=0.0), "test")


# --- truth kinematics ------------------------------------------------------------------------


def test_rest_specific_force_is_plus_g_in_body_axes_and_rotations_follow_the_attitude():
    truth = rest_truth()
    np.testing.assert_allclose(truth.specific_force_com_b(), [0.0, 0.0, 9.81], atol=1e-12)
    rolled = matrix_to_quaternion(rotation_exp([np.pi / 2, 0.0, 0.0]))  # body +Y points world up
    truth = rest_truth(quaternion=rolled)
    np.testing.assert_allclose(truth.specific_force_com_b(), [0.0, 9.81, 0.0], atol=1e-12)


def test_lever_arm_adds_centripetal_and_euler_terms_and_point_velocity():
    omega = np.array([0.0, 0.0, 2.0])
    truth = rest_truth(omega=omega)
    r = np.array([0.1, 0.0, 0.0])
    np.testing.assert_allclose(truth.specific_force_at_b(r), [0.0, 0.0, 9.81] + np.cross(omega, np.cross(omega, r)))
    np.testing.assert_allclose(truth.point_velocity_w(r), np.cross(omega, r))
    np.testing.assert_allclose(truth.point_position_w(r), truth.after.position_w + r)
    spinning_up = StepTruth(
        truth.before,
        VehicleState(
            truth.after.time_s, truth.after.position_w, truth.after.quaternion_wxyz, np.zeros(3), omega + [0, 0, 0.01]
        ),
        GRAVITY,
    )
    alpha = np.array([0.0, 0.0, 0.01 / DT])
    np.testing.assert_allclose(spinning_up.angular_acceleration_b(), alpha, atol=1e-9)
    expected = [0.0, 0.0, 9.81] + np.cross(alpha, r) + np.cross(omega + [0, 0, 0.01], np.cross(omega + [0, 0, 0.01], r))
    np.testing.assert_allclose(spinning_up.specific_force_at_b(r), expected, atol=1e-9)


def test_midpoint_pairing_uses_the_slerp_midpoint_attitude():
    before = VehicleState(0.0, np.zeros(3), [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    after_q = matrix_to_quaternion(rotation_exp([0.0, 0.0, 0.2]))
    after = VehicleState(DT, np.zeros(3), after_q, [DT * 1.0, 0.0, 0.0], [0.0, 0.0, 0.2 / DT])
    start = StepTruth(before, after, GRAVITY, "step_start")
    middle = StepTruth(before, after, GRAVITY, "midpoint")
    acceleration = np.array([1.0, 0.0, 9.81])
    np.testing.assert_allclose(start.specific_force_com_b(), acceleration, atol=1e-12)
    np.testing.assert_allclose(
        middle.specific_force_com_b(), rotation_exp([0.0, 0.0, 0.1]).T @ acceleration, atol=1e-12
    )
    with pytest.raises(ValueError, match="attitude_pairing"):
        StepTruth(before, after, GRAVITY, "end")
    assert SyntheticBackend.specific_force_attitude_pairing == "midpoint"


# --- the four sensors and the suite ---------------------------------------------------------


def test_magnetometer_rotates_the_earth_field_and_adds_motor_interference():
    section = quiet_sensor_config()["magnetometer"]
    section["field_w_ut"] = [0.0, 20.0, -40.0]
    section["motor_interference_b_ut_per_n15"] = [0.0, 0.0, 0.1]
    sensor = MagnetometerSensor(section)
    sensor.bind(MASS, GRAVITY)
    sensor.reset(np.random.SeedSequence(0))
    yawed = matrix_to_quaternion(rotation_exp([0.0, 0.0, np.pi / 2]))  # body X points world north (+Y)
    reading = sensor.measure(rest_truth(quaternion=yawed), {"applied_thrust_n": [4.0, 4.0, 4.0, 4.0]})
    np.testing.assert_allclose(reading["magnetic_field_ut"], [20.0, 0.0, -40.0 + 0.1 * 4 * 8.0], atol=1e-9)
    np.testing.assert_allclose(reading["motor_interference_s_ut"], [0.0, 0.0, 3.2])


def test_barometer_and_gps_measure_their_mounting_points():
    section = quiet_sensor_config()
    section["barometer"]["position_b_m"] = [0.0, 0.0, 0.1]
    section["gps"]["position_b_m"] = [0.2, 0.0, 0.0]
    barometer, gps = BarometerSensor(section["barometer"]), GpsSensor(section["gps"])
    for sensor in (barometer, gps):
        sensor.bind(MASS, GRAVITY)
        sensor.reset(np.random.SeedSequence(0))
    omega = np.array([0.0, 0.0, 1.0])
    truth = rest_truth(omega=omega)
    lever = np.array([0.0, 0.0, 0.1]) - MASS.com_b
    assert barometer.measure(truth, {})["altitude_m"] == pytest.approx(3.0 + lever[2])
    reading = gps.measure(truth, {})
    antenna = np.array([0.2, 0.0, 0.0]) - MASS.com_b
    np.testing.assert_allclose(reading["position_w_m"], truth.after.position_w + antenna)
    np.testing.assert_allclose(reading["velocity_w_m_s"], np.cross(omega, antenna))


def test_gps_profiles_and_explicit_overrides():
    section = {**DEFAULT_SENSORS["gps"], "profile": "standard", "velocity_accuracy_m_s": 0.3}
    values = resolve_gps_parameters(section)
    assert values["horizontal_accuracy_m"] == 0.35 and values["velocity_accuracy_m_s"] == 0.3
    with pytest.raises(ValueError, match="profile"):
        resolve_gps_parameters({**section, "profile": "military"})
    sensor = GpsSensor({**DEFAULT_SENSORS["gps"], "profile": "rtk"})
    assert sensor.describe()["position_error_stationary_std_m"][0] == pytest.approx(0.01 / np.sqrt(1 - 0.81))


def test_suite_samples_at_each_rate_releases_after_the_delay_and_keeps_the_sample_time():
    suite = SensorSuite(quiet_sensor_config(), DT, GRAVITY)
    suite.bind_vehicle(MASS, "step_start")
    suite.reset(0)
    state = VehicleState(0.0, [0.0, 0.0, 1.0], [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    released = {name: [] for name in suite.names}
    for step in range(1, 101):
        before = VehicleState((step - 1) * DT, state.position_w, state.quaternion_wxyz, np.zeros(3), np.zeros(3))
        after = VehicleState(step * DT, state.position_w, state.quaternion_wxyz, np.zeros(3), np.zeros(3))
        readings = suite.sample(step, before, after, {"applied_thrust_n": [3.0] * 4})
        for name, reading in readings.items():
            if reading is not None:
                released[name].append((step, reading["time_s"]))
    assert [step for step, _ in released["imu"]] == list(range(1, 101))
    assert [step for step, _ in released["magnetometer"]] == list(range(4, 101, 4))
    assert [step for step, _ in released["barometer"]] == list(range(4, 101, 4))
    assert released["gps"] == [(44, pytest.approx(0.1)), (64, pytest.approx(0.2)), (84, pytest.approx(0.3))]
    assert all(stamp == pytest.approx(step * DT) for step, stamp in released["imu"])
    description = suite.describe()
    assert description["sensors"]["gps"]["delay_steps"] == 24 and description["attitude_pairing"] == "step_start"


def test_rates_must_divide_the_physics_rate_and_delays_must_be_whole_steps():
    assert sample_divisor(50.0, DT, "x") == 4 and delay_steps(0.12, DT, "x") == 24
    with pytest.raises(ValueError, match="divide"):
        sample_divisor(60.0, DT, "x")
    with pytest.raises(ValueError, match="whole number"):
        delay_steps(0.012, DT, "x")
    for override in (
        "sensors.imu.rate_hz=60",
        "sensors.gps.delay_s=0.012",
        "sensors.gps.profile=military",
        "sensors.imu.accelerometer.noise_type=triple",
        "sensors.barometer.decay_factor=1.5",
        "sensors.magnetometer.model.axes_misalignment=[1,2]",
    ):
        with pytest.raises(ConfigurationError, match="sensors"):
            load_config(overrides=[override])
    broken = load_config()
    broken["sensors"]["imu"]["extra"] = 1.0
    with pytest.raises(ConfigurationError, match="unknown"):
        validate_config(broken)


def test_disabled_sensor_sections_are_still_validated():
    config = load_config()
    config["sensors"]["gps"]["enabled"] = False
    config["sensors"]["gps"]["profile"] = "military"
    with pytest.raises(ConfigurationError, match="profile"):
        validate_config(config)


def test_hovering_synthetic_plant_reads_plus_g_with_a_quiet_imu():
    config = load_config(overrides=["effects.ou_wrench.enabled=false", "simulation.duration_s=2.0"])
    config["sensors"] = quiet_sensor_config()
    plant = SyntheticBackend(config)
    loop = MotionControlLoop(config, plant)
    loop.reset()
    record = None
    for _ in range(300):
        loop.prepare_step()
        plant.step()
        record = loop.finish_step()
    imu = record["sensors"]["imu"]
    np.testing.assert_allclose(imu["accelerometer_m_s2"], [0.0, 0.0, 9.81], atol=0.05)
    np.testing.assert_allclose(imu["gyroscope_rad_s"], 0.0, atol=0.01)
    np.testing.assert_allclose(imu["true_specific_force_s_m_s2"], imu["accelerometer_m_s2"])
    assert loop.sensors.attitude_pairing == "midpoint"
