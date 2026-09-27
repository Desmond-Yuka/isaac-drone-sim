"""Independent boundary, geometric and derivative tests for the spiral reference."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
from numpy.polynomial import Polynomial
import pytest

from isaac_drone.trajectories.spiral import SpiralTrajectory, validate_spiral_config, _progress_derivatives
from isaac_drone.types import VehicleState


def config(**changes):
    values = {
        "start_delay_s": 2.0, "takeoff_height_m": 1.0, "takeoff_duration_s": 4.0,
        "radius_m": 1.0, "turns": 2.0, "climb_height_m": 3.0,
        "spiral_duration_s": 24.0, "initial_phase_rad": .3,
        "yaw_mode": "fixed", "yaw_offset_rad": .35,
        "initial_speed_tolerance_m_s": 1e-3,
        "initial_angular_speed_tolerance_rad_s": 1e-3,
    }
    values.update(changes)
    return values


def initial_state(time=0.0, yaw=-.4):
    return VehicleState(time, np.array([2.3, -1.7, .42]),
                        np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]), np.zeros(3), np.zeros(3))


def trajectory(**changes):
    result = SpiralTrajectory(config(**changes))
    result.reset(initial_state())
    return result


def derivatives(reference):
    return np.array([reference.velocity_w, reference.acceleration_w, reference.jerk_w, reference.snap_w])


def test_ninth_degree_progress_matches_independent_polynomial_and_boundary_constraints():
    polynomial = Polynomial([0, 0, 0, 0, 0, 126, -420, 540, -315, 70])
    duration = 3.7
    for u in np.linspace(0, 1, 41):
        expected = np.array([polynomial.deriv(order)(u)/duration**order for order in range(5)])
        np.testing.assert_allclose(_progress_derivatives(u, duration), expected, atol=2e-10, rtol=1e-10)
    np.testing.assert_array_equal(_progress_derivatives(-.1, duration), np.zeros(5))
    np.testing.assert_array_equal(_progress_derivatives(1.1, duration), [1, 0, 0, 0, 0])
    near_end = _progress_derivatives(np.nextafter(1., 0.), duration)
    assert 0 <= near_end[0] <= 1 and near_end[1] >= 0


@pytest.mark.parametrize("yaw_mode", ["fixed", "tangent"])
def test_phase_endpoints_have_exact_zero_first_four_position_derivatives(yaw_mode):
    path = trajectory(yaw_mode=yaw_mode)
    p0 = initial_state().position_w
    assert path.mission_duration_s == 30
    for time, name, position in [(0, "delay", p0), (2, "takeoff", p0),
                                  (6, "spiral", p0+[0, 0, 1]),
                                  (30, "hold", p0+[0, 0, 4])]:
        assert path.phase(time) == name
        reference = path.sample(time)
        np.testing.assert_allclose(reference.position_w, position, atol=2e-14)
        np.testing.assert_array_equal(derivatives(reference), np.zeros((4, 3)))
        assert reference.yaw_rate_rad_s == 0
        assert reference.yaw_acceleration_rad_s2 == 0
    np.testing.assert_array_equal(path.sample(1e6).position_w, path.endpoint_position_w)
    np.testing.assert_array_equal(derivatives(path.sample(1e6)), np.zeros((4, 3)))


@pytest.mark.parametrize("yaw_mode", ["fixed", "tangent"])
def test_position_and_heading_are_continuous_across_every_phase(yaw_mode):
    path = trajectory(yaw_mode=yaw_mode, turns=-1.25)
    for join in (2., 6., 30.):
        left, middle, right = [path.sample(t) for t in (join-1e-6, join, join+1e-6)]
        for neighbor in (left, right):
            np.testing.assert_allclose(neighbor.position_w, middle.position_w, atol=1e-11)
            np.testing.assert_allclose(derivatives(neighbor), derivatives(middle), atol=1e-3)
            assert neighbor.yaw_rad == pytest.approx(middle.yaw_rad, abs=1e-11)
            assert abs(neighbor.yaw_rate_rad_s) < 1e-9
            assert abs(neighbor.yaw_acceleration_rad_s2) < 1e-7


def test_takeoff_is_vertical_and_helix_center_preserves_starting_position():
    cfg = config()
    path = trajectory()
    p0 = initial_state().position_w
    center = p0[:2]-cfg["radius_m"]*np.array([np.cos(cfg["initial_phase_rad"]), np.sin(cfg["initial_phase_rad"])])
    for time in np.linspace(2, 6, 11):
        np.testing.assert_array_equal(path.sample(time).position_w[:2], p0[:2])
    for time in np.linspace(6, 30, 101):
        reference = path.sample(time)
        assert np.linalg.norm(reference.position_w[:2]-center) == pytest.approx(cfg["radius_m"], abs=2e-14)
        assert reference.velocity_w[2] >= 0
    assert not np.allclose(center, p0[:2])


@pytest.mark.parametrize("turns", [2., -1.5, .3])
def test_position_derivatives_through_snap_match_independent_finite_differences(turns):
    path = trajectory(turns=turns, yaw_mode="tangent")
    attributes = ("position_w", "velocity_w", "acceleration_w", "jerk_w", "snap_w")
    step = 2e-4
    for time in (2.7, 3.9, 5.1, 7.8, 11.4, 17.3, 24.1, 28.5):
        samples = {offset: path.sample(time+offset*step) for offset in (-2, -1, 0, 1, 2)}
        for order in range(1, 5):
            numerical = (-getattr(samples[2], attributes[order-1])+8*getattr(samples[1], attributes[order-1])
                         -8*getattr(samples[-1], attributes[order-1])+getattr(samples[-2], attributes[order-1]))/(12*step)
            np.testing.assert_allclose(getattr(samples[0], attributes[order]), numerical, atol=3e-8, rtol=3e-8)
        yaw_velocity = (-samples[2].yaw_rad+8*samples[1].yaw_rad-8*samples[-1].yaw_rad+samples[-2].yaw_rad)/(12*step)
        yaw_acceleration = (-samples[2].yaw_rate_rad_s+8*samples[1].yaw_rate_rad_s
                            -8*samples[-1].yaw_rate_rad_s+samples[-2].yaw_rate_rad_s)/(12*step)
        assert samples[0].yaw_rate_rad_s == pytest.approx(yaw_velocity, abs=2e-8)
        assert samples[0].yaw_acceleration_rad_s2 == pytest.approx(yaw_acceleration, abs=3e-8)


@pytest.mark.parametrize("turns", [2., -2.])
def test_helix_speed_increases_then_decreases_with_analytic_midpoint_peak(turns):
    cfg = config(turns=turns)
    path = trajectory(turns=turns)
    times = np.linspace(6, 30, 201)
    speeds = np.array([np.linalg.norm(path.sample(t).velocity_w) for t in times])
    assert np.all(np.diff(speeds[:101]) > 0)
    assert np.all(np.diff(speeds[100:]) < 0)
    length = np.hypot(2*np.pi*turns*cfg["radius_m"], cfg["climb_height_m"])
    assert speeds[100] == pytest.approx(length*315/(128*cfg["spiral_duration_s"]), abs=1e-13)
    np.testing.assert_allclose(speeds, speeds[::-1], atol=3e-14)
    # Zero tangential acceleration at peak speed does not remove centripetal acceleration.
    assert np.linalg.norm(path.sample(18).acceleration_w[:2]) > 0.1


def test_fixed_yaw_offset_is_reached_smoothly_during_takeoff():
    path = trajectory(yaw_mode="fixed", yaw_offset_rad=.8)
    assert path.sample(0).yaw_rad == pytest.approx(-.4)
    assert path.sample(2).yaw_rad == pytest.approx(-.4)
    assert path.sample(4).yaw_rad == pytest.approx(0.0)
    for time in (6, 10, 20, 30, 50):
        result = path.sample(time)
        assert result.yaw_rad == pytest.approx(.4)
        assert result.yaw_rate_rad_s == 0
        assert result.yaw_acceleration_rad_s2 == 0


@pytest.mark.parametrize("turns", [2., -1.5])
def test_tangent_yaw_follows_direction_of_motion_without_wrapping(turns):
    offset = .2
    path = trajectory(turns=turns, yaw_mode="tangent", yaw_offset_rad=offset)
    start_yaw = path.sample(6).yaw_rad
    assert abs(start_yaw-(-.4)) <= np.pi
    yaws = []
    for time in np.linspace(6.01, 29.99, 80):
        result = path.sample(time)
        heading = np.arctan2(result.velocity_w[1], result.velocity_w[0])+offset
        np.testing.assert_allclose([np.cos(result.yaw_rad), np.sin(result.yaw_rad)],
                                   [np.cos(heading), np.sin(heading)], atol=1e-13)
        assert result.yaw_rate_rad_s*turns > 0
        yaws.append(result.yaw_rad)
    assert np.all(np.diff(yaws)*turns > 0)
    assert path.sample(30).yaw_rad-start_yaw == pytest.approx(2*np.pi*turns)


def test_nonzero_reset_time_phase_boundaries_and_pure_order_independent_sampling():
    path = SpiralTrajectory(config())
    assert path.mission_duration_s == 30
    origin = 7.2
    path.reset(initial_state(time=origin))
    for elapsed, phase in [(0, "delay"), (2, "takeoff"), (6, "spiral"), (30, "hold")]:
        assert path.phase(origin+elapsed) == phase
    expected = path.sample(origin+12.3)
    path.sample(origin+100)
    path.sample(origin)
    again = path.sample(origin+12.3)
    np.testing.assert_array_equal(again.position_w, expected.position_w)
    np.testing.assert_array_equal(derivatives(again), derivatives(expected))
    endpoint = path.endpoint_position_w
    endpoint[:] = -999
    assert np.all(path.endpoint_position_w > -999)
    with pytest.raises(ValueError, match="precedes"):
        path.sample(origin-.01)
    path.reset(initial_state(time=2, yaw=.9))
    assert path.sample(2).yaw_rad == pytest.approx(.9)


def test_zero_delay_starts_takeoff_and_fractional_turns_end_at_correct_point():
    path = trajectory(start_delay_s=0, turns=.25, initial_phase_rad=0)
    p0 = initial_state().position_w
    assert path.phase(0) == "takeoff"
    np.testing.assert_allclose(path.endpoint_position_w, p0+[-1, 1, 4], atol=1e-14)


@pytest.mark.parametrize("field,value", [("takeoff_height_m", 0), ("takeoff_duration_s", -1),
    ("radius_m", 0), ("turns", 0), ("climb_height_m", -1), ("spiral_duration_s", 0),
    ("initial_phase_rad", np.nan), ("start_delay_s", -1), ("yaw_mode", "point_at_center"),
    ("yaw_offset_rad", np.inf), ("initial_speed_tolerance_m_s", -1),
    ("initial_angular_speed_tolerance_rad_s", -1), ("turns", True), ("turns", "2")])
def test_bad_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        validate_spiral_config(config(**{field: value}))


def test_missing_unknown_configuration_and_uninitialized_sampling_fail():
    missing = config(); del missing["radius_m"]
    for invalid in (missing, {**config(), "center_w_m": [0, 0, 0]}, {**config(), 4: "bad"}):
        with pytest.raises(ValueError):
            SpiralTrajectory(invalid)
    path = SpiralTrajectory(config())
    for operation in (lambda: path.sample(0), lambda: path.phase(0), lambda: path.endpoint_position_w):
        with pytest.raises(RuntimeError, match="reset"):
            operation()


def test_initial_motion_outside_explicit_settling_tolerances_is_not_silently_discarded():
    path = trajectory()
    for moving in (replace(initial_state(), linear_velocity_w=np.array([.1, 0, 0])),
                   replace(initial_state(), angular_velocity_b=np.array([0, .1, 0]))):
        with pytest.raises(ValueError, match="settled"):
            path.reset(moving)
        with pytest.raises(RuntimeError):
            path.sample(0)
    path.reset(replace(initial_state(), linear_velocity_w=np.array([1e-4, 0, 0])))
    # The accepted measured state is never rewritten. The reference starts at rest.
    np.testing.assert_array_equal(path.sample(0).velocity_w, np.zeros(3))


def test_caller_configuration_mutation_does_not_change_mission():
    cfg = config()
    original = deepcopy(cfg)
    path = SpiralTrajectory(cfg)
    cfg["radius_m"] = 100
    path.reset(initial_state())
    assert path.config == original


def test_helix_name_is_the_same_three_dimensional_trajectory():
    from isaac_drone.trajectories.spiral import HelixTrajectory
    assert HelixTrajectory is SpiralTrajectory
