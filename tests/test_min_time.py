"""Minimum-time time laws and planner invariants; nominal screening only, not flight proof."""
import math

import numpy as np
import pytest

from isaac_drone.trajectories.feasibility import nominal_requirement, validate_helix_feasibility
from isaac_drone.trajectories.min_time import plan_minimum_time, timing_summary
from isaac_drone.trajectories.helix import HelixTrajectory, validate_spiral_config
from isaac_drone.trajectories.timing import CruiseTimeLaw, HermiteTimeLaw, _progress_derivatives, _progress_integral
from isaac_drone.core.types import MassProperties, VehicleState

GRAVITY = np.array([0., 0., -9.81])
LOWER, UPPER = np.full(4, .1), np.full(4, 10.)


def config(**changes):
    values = dict(start_delay_s=2., takeoff_height_m=1., takeoff_duration_s=None, radius_m=1., turns=2.,
                  climb_height_m=2., spiral_duration_s=None, initial_phase_rad=-math.pi/2, yaw_mode="tangent",
                  yaw_offset_rad=0., initial_speed_tolerance_m_s=1e-3, initial_angular_speed_tolerance_rad_s=1e-3,
                  thrust_utilization=.85)
    values.update(changes)
    return values


def reset_path(**changes):
    path = HelixTrajectory(config(**changes))
    path.reset(VehicleState(0., np.array([.3, -.2, .1]), np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3)))
    return path


def mass():
    return MassProperties(1.24, np.array([[.0134, .0002, 0.], [.0002, .0144, 0.], [0., 0., .0138]]), np.zeros(3))


def matrix():
    return np.array([[0.]*4, [0.]*4, [1.]*4, [-.13, -.13, .13, .13], [-.13, .13, .13, -.13], [-.07, .07, -.07, .07]])


def control(**changes):
    result = dict(max_tilt_rad=math.radians(80), max_yaw_rate_rad_s=2*math.pi, max_acceleration_m_s2=None)
    result.update(changes)
    return result


def plan(controls=None, lag=(0.0, 0.0), **changes):
    path = reset_path(**changes)
    report = plan_minimum_time(path, control() if controls is None else controls, mass(), GRAVITY, matrix(),
                               LOWER, UPPER, motor_time_constants_s=lag)
    return path, report


@pytest.fixture(scope="module")
def lagged():
    return plan(lag=(.085, .01))


@pytest.fixture(scope="module")
def ideal():
    return plan()


def test_progress_integral_is_the_antiderivative_with_half_total_area():
    assert _progress_integral(1.0) == pytest.approx(.5, abs=1e-15)
    for u in np.linspace(.01, .99, 23):
        assert _progress_integral(u) == pytest.approx(u-.5+_progress_integral(1-u), abs=1e-14)
        slope = (_progress_integral(u+1e-6)-_progress_integral(u-1e-6))/2e-6
        assert slope == pytest.approx(_progress_derivatives(u, 1.0)[0], abs=1e-8)


def test_cruise_law_is_rest_to_rest_monotone_and_its_derivatives_are_consistent():
    law = CruiseTimeLaw(.4, .7, 1.1)
    assert law.cruise_s == pytest.approx(1/.4-.9)
    assert [name for name, *_ in law.segments] == ["accelerate", "cruise", "decelerate"]
    np.testing.assert_array_equal(law.derivatives(0.), np.zeros(5))
    np.testing.assert_allclose(law.derivatives(law.duration_s), [1, 0, 0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(law.derivatives(law.duration_s-1e-9), [1, 0, 0, 0, 0], atol=1e-8)
    junctions = [law.accelerate_s, law.accelerate_s+law.cruise_s]
    times = np.r_[np.linspace(1e-3, law.duration_s-1e-3, 57), junctions]
    h = 1e-6
    for time in times:
        values = law.derivatives(time)
        assert 0 <= values[0] <= 1 and 0 <= values[1] <= law.cruise_rate+1e-15
        difference = (law.derivatives(time+h)-law.derivatives(time-h))/(2*h)
        np.testing.assert_allclose(difference[:4], values[1:], atol=2e-5*max(1., np.abs(values[1:]).max()))
    for time in junctions:  # rate is C4, so position through snap is continuous at the joins
        np.testing.assert_allclose(law.derivatives(time-1e-10), law.derivatives(time+1e-10), atol=1e-7)


def test_cruise_law_rejects_ramps_that_cannot_reach_the_rate():
    with pytest.raises(ValueError, match="too long"):
        CruiseTimeLaw(1., 1.5, 1.)
    with pytest.raises(ValueError, match="positive"):
        CruiseTimeLaw(1., 0., .5)


@pytest.mark.parametrize("utilization", [None, .5, 1.2, True])
def test_null_durations_require_a_valid_thrust_utilization(utilization):
    with pytest.raises(ValueError, match="thrust_utilization"):
        validate_spiral_config(config(thrust_utilization=utilization))


def test_fixed_durations_do_not_need_utilization_and_keep_the_hermite_law():
    path = HelixTrajectory({key: value for key, value in config(takeoff_duration_s=4., spiral_duration_s=24.).items()
                            if key != "thrust_utilization"})
    assert path.planned_phases == ()
    assert all(isinstance(law, HermiteTimeLaw) for law in path.time_laws.values())


def test_null_phases_cannot_be_sampled_until_planned_and_reset_clears_the_plan():
    path = reset_path(takeoff_duration_s=3.)
    assert path.planned_phases == ("spiral",)
    assert path.endpoint_position_w.shape == (3,)
    with pytest.raises(RuntimeError, match="Plan"):
        path.sample(0.)
    with pytest.raises(ValueError, match="null-duration"):
        path.set_time_laws(takeoff=HermiteTimeLaw(1.))
    law = CruiseTimeLaw(.2, 1., 1.)
    path.set_time_laws(spiral=law)
    assert path.mission_duration_s == pytest.approx(2.+3.+law.duration_s)
    assert path.phase(5.5) == "spiral"
    np.testing.assert_allclose(path.sample(path.mission_duration_s).position_w, path.endpoint_position_w)
    path.reset(VehicleState(1., np.zeros(3), np.array([1., 0., 0., 0.]), np.zeros(3), np.zeros(3)))
    with pytest.raises(RuntimeError, match="Plan"):
        path.sample(1.)


def test_plan_reaches_but_never_exceeds_the_thrust_band(lagged):
    path, report = lagged
    assert report["planned_phases"] == ["takeoff", "spiral"]
    lower_band, upper_band = (np.array(item) for item in report["planning_band_n"])
    np.testing.assert_allclose(upper_band, UPPER-.15*(UPPER-LOWER))
    check = validate_helix_feasibility(path, control(), mass(), GRAVITY, matrix(), LOWER, UPPER)
    assert np.all(np.array(check["sampled_max_rotor_thrust_n"]) <= upper_band+1e-2)
    assert np.all(np.array(check["sampled_min_rotor_thrust_n"]) >= lower_band-1e-2)
    # "Maximum motor use": the busiest rotor nearly reaches the top of the band. Minimum time,
    # not utilization, is the objective: with motor lag a slightly lower cruise saves ramp time.
    assert max(check["sampled_max_rotor_thrust_n"]) > .95*upper_band.max()
    fraction = np.array(check["sampled_max_thrust_fraction_of_max"])
    np.testing.assert_allclose(check["sampled_max_motor_speed_fraction_of_max"], np.sqrt(fraction))
    assert check["phase_boundaries_s"] == path.phase_boundaries_s
    assert check["timing"]["phases"]["spiral"]["planned"]
    summary = timing_summary(check)
    assert "spiral" in summary and "(min-time)" in summary


def test_motor_lag_lead_keeps_command_in_band_and_lengthens_ramps(lagged, ideal):
    path, report = lagged
    tau_up, tau_down = report["motor_time_constants_s"]
    upper_band = np.array(report["planning_band_n"][1])
    inverse = np.linalg.inv(matrix()[2:])
    assert report["phases"]["takeoff"]["accelerate_s"] > ideal[1]["phases"]["takeoff"]["accelerate_s"]
    start = path.phase_boundaries_s[1]
    times = np.linspace(start, start+report["phases"]["takeoff"]["accelerate_s"], 400)
    rotors = np.array([inverse@np.r_[nominal_requirement(path.sample(t), 1.24, mass().inertia_com_b, GRAVITY)[:2]]
                       for t in times])
    slope = np.gradient(rotors, times, axis=0)
    lead = rotors+np.where(slope > 0, tau_up, tau_down)*slope
    assert np.all(lead <= upper_band+.05)
    assert path.mission_duration_s > ideal[0].mission_duration_s


def test_higher_utilization_flies_faster(lagged):
    slower, _ = plan(lag=(.085, .01), thrust_utilization=.75)
    assert lagged[0].mission_duration_s < slower.mission_duration_s


def test_tangent_heading_rate_limit_caps_the_cruise_rate():
    path, report = plan(controls=control(max_yaw_rate_rad_s=1.5), takeoff_duration_s=3.)
    assert report["planned_phases"] == ["spiral"]
    spiral = report["phases"]["spiral"]
    assert spiral["peak_rate_per_s"]*abs(path.turn_angle_rad) <= 1.5+1e-9
    validate_helix_feasibility(path, control(max_yaw_rate_rad_s=1.5), mass(), GRAVITY, matrix(), LOWER, UPPER)


def test_hover_outside_the_band_is_rejected():
    with pytest.raises(ValueError, match="Hover"):
        plan(thrust_utilization=.55)
