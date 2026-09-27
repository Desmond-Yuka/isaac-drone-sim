"""Nominal feasibility screening invariants; no test here certifies flight."""

from dataclasses import replace

import numpy as np
import pytest

from isaac_drone.control.math import desired_attitude_kinematics
from isaac_drone.trajectories.feasibility import validate_helix_feasibility
from isaac_drone.trajectories.spiral import HelixTrajectory
from isaac_drone.types import MassProperties, VehicleState


def path(**changes):
    cfg = dict(start_delay_s=2., takeoff_height_m=1., takeoff_duration_s=4., radius_m=1.,
               turns=2., climb_height_m=3., spiral_duration_s=24., initial_phase_rad=.2,
               yaw_mode="fixed", yaw_offset_rad=0., initial_speed_tolerance_m_s=.001,
               initial_angular_speed_tolerance_rad_s=.001)
    cfg.update(changes)
    trajectory = HelixTrajectory(cfg)
    trajectory.reset(VehicleState(0., np.array([1., 2., .3]), np.array([1., 0., 0., 0.]),
                                  np.zeros(3), np.zeros(3)))
    return trajectory


def mass():
    return MassProperties(1.24, np.array([[.022, .003, -.002], [.003, .027, .004], [-.002, .004, .035]]),
                          np.array([.01, -.015, .02]))


def matrix():
    points = np.array([[-.1, .1, 0], [-.1, -.1, 0], [.1, .1, 0], [.1, -.1, 0]])
    axes = np.tile([0., 0., 1.], (4, 1))
    torques = np.cross(points-mass().com_b, axes)+np.array([-.07, .07, .07, -.07])[:, None]*axes
    return np.vstack((axes.T, torques.T))


def control(**changes):
    result = dict(max_tilt_rad=np.pi/3, max_yaw_rate_rad_s=2.0, max_acceleration_m_s2=None)
    result.update(changes)
    return result


def validate(trajectory=None, controls=None, properties=None, allocation=None, lower=None, upper=None):
    return validate_helix_feasibility(path() if trajectory is None else trajectory,
        control() if controls is None else controls, mass() if properties is None else properties,
        np.array([0., 0., -9.81]), matrix() if allocation is None else allocation,
        np.full(4, .1) if lower is None else lower, np.full(4, 10.) if upper is None else upper)


def test_nominal_pass_reports_actual_asymmetric_hover_and_explicit_nonproof_limits():
    result = validate()
    assert result["status"] == "passed_nominal_sampled_checks"
    assert result["sample_count"] == 98
    assert len(result["sample_times_s"]) == 98
    assert result["sample_times_s"][0] == 0 and result["sample_times_s"][-1] == 30
    assert not result["continuous_time_certificate"]
    assert not result["closed_loop_guarantee"]
    assert not result["motor_dynamic_feasibility_checked"]
    expected = np.linalg.solve(matrix()[[2, 3, 4, 5]], [mass().mass_kg*9.81, 0, 0, 0])
    np.testing.assert_allclose(result["hover_thrusts_n"], expected, atol=1e-12)
    assert not np.allclose(expected, mass().mass_kg*9.81/4)
    assert np.all(np.array(result["sampled_min_lower_thrust_margin_n"]) >= 0)
    assert np.all(np.array(result["sampled_min_upper_thrust_margin_n"]) >= 0)


def test_insufficient_collective_or_excessive_minimum_thrust_cannot_hover():
    for options in (dict(properties=replace(mass(), mass_kg=30.)), dict(lower=np.full(4, 4.))):
        with pytest.raises(ValueError, match="Static hover infeasible"):
            validate(**options)


@pytest.mark.parametrize("turns", [2., -2.])
def test_exact_tangent_yaw_peak_is_checked_even_when_only_slightly_over_limit(turns):
    trajectory = path(yaw_mode="tangent", turns=turns)
    peak = abs(2*np.pi*turns)*(315/128)/24
    with pytest.raises(ValueError, match="yaw-rate peak"):
        validate(trajectory, control(max_yaw_rate_rad_s=peak-1e-8))
    report = validate(trajectory, control(max_yaw_rate_rad_s=peak))
    assert report["exact_helix_yaw_rate_peak_rad_s"] == pytest.approx(peak, abs=1e-14)


def test_fixed_mode_takeoff_turn_also_obeys_exact_yaw_peak_limit():
    trajectory = path(yaw_mode="fixed", yaw_offset_rad=np.pi)
    with pytest.raises(ValueError, match="yaw-rate peak"):
        validate(trajectory, control(max_yaw_rate_rad_s=1.0))
    report = validate(trajectory, control(max_yaw_rate_rad_s=2.0))
    assert report["exact_takeoff_yaw_rate_peak_rad_s"] == pytest.approx(np.pi*315/(128*4))
    assert report["exact_helix_yaw_rate_peak_rad_s"] == 0


def test_sampled_acceleration_and_tilt_limits_do_not_silently_clip_the_plan():
    with pytest.raises(ValueError, match="Reference acceleration"):
        validate(controls=control(max_acceleration_m_s2=.1))
    with pytest.raises(ValueError, match="required tilt"):
        validate(controls=control(max_tilt_rad=.01))


def test_hover_capable_bounds_can_still_reject_dynamic_nominal_wrench():
    hover = np.linalg.solve(matrix()[[2, 3, 4, 5]], [mass().mass_kg*9.81, 0, 0, 0])
    with pytest.raises(ValueError, match="feedforward infeasible"):
        validate(upper=hover+.1)


def test_complete_inertia_and_gyroscopic_torque_match_world_momentum_derivative(monkeypatch):
    import isaac_drone.trajectories.feasibility as module
    original_allocate = module.BoundedAllocator.allocate
    requested = []
    def capture(self, wrench, *args, **kwargs):
        requested.append(wrench)
        return original_allocate(self, wrench, *args, **kwargs)
    monkeypatch.setattr(module.BoundedAllocator, "allocate", capture)
    trajectory = path(yaw_mode="tangent")
    report = validate(trajectory)
    times = np.array(report["sample_times_s"])
    index = int(np.argmin(np.abs(times-18.)))
    time = times[index]
    properties = mass()
    def orientation_and_momentum(at):
        reference = trajectory.sample(at)
        rotation, omega, _ = desired_attitude_kinematics(
            properties.mass_kg*(reference.acceleration_w-[0, 0, -9.81]),
            properties.mass_kg*reference.jerk_w, properties.mass_kg*reference.snap_w,
            reference.yaw_rad, reference.yaw_rate_rad_s, reference.yaw_acceleration_rad_s2)
        return rotation, rotation@properties.inertia_com_b@omega
    dt = 1e-5
    rotation, _ = orientation_and_momentum(time)
    world_torque = (orientation_and_momentum(time+dt)[1]-orientation_and_momentum(time-dt)[1])/(2*dt)
    # The initial allocation call is the separate hover check.
    np.testing.assert_allclose(requested[index+1].torque_b, rotation.T@world_torque, atol=2e-9)
    assert np.linalg.norm(requested[index+1].torque_b) > 1e-3


def test_full_rank_canted_axes_invalid_bounds_and_unreset_mission_are_rejected():
    tilted = matrix()
    tilted[0] = np.sin(.1)
    tilted[2] = np.cos(.1)
    with pytest.raises(ValueError, match=r"body \+Z"):
        validate(allocation=tilted)
    with pytest.raises(ValueError, match="bounds"):
        validate(lower=np.ones(4), upper=np.zeros(4))
    unreset = HelixTrajectory(path().config)
    with pytest.raises(RuntimeError, match="reset"):
        validate(unreset)
