"""Physical invariants and constraint optimality, independent of Isaac Sim."""

from dataclasses import replace

import numpy as np
import pytest

from isaac_drone.control import AllocationResult, BoundedAllocator, GeometricController
from isaac_drone.core.rotations import (
    desired_attitude_kinematics,
    hat,
    normalized_derivatives,
    quaternion_to_matrix,
    rotation_log,
    vee,
)
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench


def config(**overrides):
    result = {
        "position_kp": [4.0, 4.0, 2.5],
        "velocity_kd": [3.5, 3.5, 2.0],
        "position_ki": [0.0, 0.0, 0.0],
        "integral_limit_m_s": [1.0, 1.0, 1.0],
        "attitude_kp": [1.85, 1.85, 0.4],
        "angular_rate_kd": [0.5, 0.5, 0.09],
        "max_tilt_rad": np.pi / 3,
        "max_yaw_rate_rad_s": np.pi / 3,
        "max_acceleration_m_s2": None,
    }
    result.update(overrides)
    return result


def mass():
    # Deliberately neither diagonal nor centered at the root-link origin.
    return MassProperties(
        2.3, np.array([[0.10, 0.02, 0.01], [0.02, 0.13, -0.005], [0.01, -0.005, 0.17]]), np.array([0.02, -0.01, 0.03])
    )


def state(time=0.0, **kwargs):
    values = dict(
        time_s=time,
        position_w=np.array([1.0, 2.0, 3.0]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_w=np.zeros(3),
        angular_velocity_b=np.zeros(3),
    )
    values.update(kwargs)
    return VehicleState(**values)


def controller(**kwargs):
    return GeometricController(config(**kwargs), mass(), np.array([0.0, 0.0, -9.81]))


def quad_matrix():
    # Synthetic asymmetric test geometry, not ARL calibration data.
    points = np.array([[0.2, 0.16, 0.02], [0.18, -0.2, 0.01], [-0.21, -0.18, -0.01], [-0.17, 0.22, 0.0]])
    axes = np.tile([0.0, 0.0, 1.0], (4, 1))
    drag = np.array([0.017, -0.016, 0.019, -0.018])
    return np.vstack((axes.T, (np.cross(points, axes) + drag[:, None] * axes).T))


def test_hat_vee_cross_and_quaternion_double_cover():
    a, b = np.array([0.4, -0.2, 0.8]), np.array([0.1, 0.8, -0.7])
    np.testing.assert_allclose(hat(a) @ b, np.cross(a, b))
    np.testing.assert_allclose(vee(hat(a)), a)
    q = np.array([0.7, 0.2, -0.4, 0.1])
    q /= np.linalg.norm(q)
    np.testing.assert_allclose(quaternion_to_matrix(q), quaternion_to_matrix(-q))
    np.testing.assert_allclose(quaternion_to_matrix(q).T @ quaternion_to_matrix(q), np.eye(3), atol=1e-15)
    with pytest.raises(ValueError):
        quaternion_to_matrix(np.array([2.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.4, np.pi - 1e-7, np.pi])
def test_rotation_log_near_identity_and_half_turn(angle):
    axis = np.array([0.2, 0.3, 0.8])
    axis /= np.linalg.norm(axis)
    rotation = quaternion_to_matrix(np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)])
    np.testing.assert_allclose(rotation_log(rotation), axis * angle, atol=2e-8)


def test_normalization_derivatives_match_finite_differences():
    x, dx, ddx = np.array([2.0, 3.0, 4.0]), np.array([1.0, -0.4, 0.7]), np.array([0.3, 0.1, -0.6])
    n, dn, ddn = normalized_derivatives(x, dx, ddx)
    h = 1e-4

    def f(t):
        value = x + dx * t + 0.5 * ddx * t * t
        return value / np.linalg.norm(value)

    np.testing.assert_allclose(dn, (f(h) - f(-h)) / (2 * h), atol=1e-9)
    np.testing.assert_allclose(ddn, (f(h) - 2 * f(0) + f(-h)) / (h * h), atol=3e-8)
    assert abs(np.dot(n, dn)) < 1e-15


def test_tilted_attitude_kinematics_against_rotation_derivatives():
    force, derivative, second = np.array([1.2, -0.7, 8.0]), np.array([0.5, 0.3, -0.2]), np.array([0.1, -0.2, 0.3])
    yaw, rate, acceleration = 0.4, 0.7, -0.3
    rotation, omega, alpha = desired_attitude_kinematics(force, derivative, second, yaw, rate, acceleration)
    h = 1e-4

    def at(t):
        return desired_attitude_kinematics(
            force + derivative * t + 0.5 * second * t * t,
            derivative + second * t,
            second,
            yaw + rate * t + 0.5 * acceleration * t * t,
            rate + acceleration * t,
            acceleration,
        )

    drotation = (at(h)[0] - at(-h)[0]) / (2 * h)
    np.testing.assert_allclose(drotation, rotation @ hat(omega), atol=2e-8)
    np.testing.assert_allclose(alpha, (at(h)[1] - at(-h)[1]) / (2 * h), atol=2e-8)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_attitude_degenerate_directions_remain_proper_and_finite():
    zero = np.zeros(3)
    fallback = quaternion_to_matrix(np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)]))
    rotation, omega, alpha = desired_attitude_kinematics(zero, zero, zero, 0.0, 1.0, 0.1, fallback)
    np.testing.assert_allclose(rotation, fallback)
    np.testing.assert_array_equal(omega, zero)
    np.testing.assert_array_equal(alpha, zero)
    rotation, omega, alpha = desired_attitude_kinematics(np.array([1.0, 0.0, 0.0]), zero, zero, 0.0, 1.0, 1.0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.all(np.isfinite(np.r_[omega, alpha]))


def test_hover_balances_actual_mass_and_gravity():
    ctl = controller()
    actual = ctl.compute(state(), TrajectorySetpoint(state().position_w), 0.01)
    np.testing.assert_allclose(actual.force_b, [0, 0, mass().mass_kg * 9.81])
    np.testing.assert_array_equal(actual.torque_b, np.zeros(3))


def test_velocity_and_acceleration_reference_are_not_discarded():
    ctl = controller()
    sp = TrajectorySetpoint(state().position_w, velocity_w=np.array([0, 0, 0.6]), acceleration_w=np.array([0, 0, 0.4]))
    actual = ctl.compute(state(), sp, 0.01)
    assert actual.force_b[2] == pytest.approx(mass().mass_kg * (9.81 + 2 * 0.6 + 0.4))


def test_gyroscopic_compensation_retains_off_diagonal_inertia():
    omega = np.array([1.1, -0.4, 0.7])
    ctl = controller(attitude_kp=[0, 0, 0], angular_rate_kd=[0, 0, 0])
    actual = ctl.compute(state(angular_velocity_b=omega), TrajectorySetpoint(state().position_w), 0.01)
    expected = np.cross(omega, mass().inertia_com_b @ omega)
    np.testing.assert_allclose(actual.torque_b, expected, atol=1e-15)
    assert not np.allclose(expected, np.cross(omega, np.diag(mass().inertia_com_b) * omega))


def test_attitude_gains_are_direct_torque_gains_and_sign_is_restoring():
    angle = 0.2
    q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0, 0])
    actual = controller().compute(state(quaternion_wxyz=q), TrajectorySetpoint(state().position_w), 0.01)
    np.testing.assert_allclose(actual.torque_b, [-1.85 * np.sin(angle), 0, 0], atol=1e-15)


def test_yaw_rate_and_acceleration_feedforward_with_full_inertia():
    rate, acceleration = 0.6, 0.25
    omega = np.array([0.0, 0.0, rate])
    ctl = controller()
    sp = TrajectorySetpoint(state().position_w, yaw_rate_rad_s=rate, yaw_acceleration_rad_s2=acceleration)
    actual = ctl.compute(state(angular_velocity_b=omega), sp, 0.01)
    expected = mass().inertia_com_b @ np.array([0, 0, acceleration]) + np.cross(omega, mass().inertia_com_b @ omega)
    np.testing.assert_allclose(actual.torque_b, expected, atol=1e-15)
    np.testing.assert_allclose(ctl.desired_angular_velocity_d, omega)


def test_yaw_rate_limit_and_outward_acceleration_limit():
    ctl = controller(max_yaw_rate_rad_s=0.3)
    sp = TrajectorySetpoint(state().position_w, yaw_rate_rad_s=4.0, yaw_acceleration_rad_s2=2.0)
    actual = ctl.compute(state(), sp, 0.01)
    np.testing.assert_allclose(ctl.desired_angular_velocity_d, [0, 0, 0.3])
    np.testing.assert_allclose(ctl.desired_angular_acceleration_d, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(actual.torque_b, [0, 0, 0.09 * 0.3], atol=1e-15)


def test_tilt_and_acceleration_limits_and_no_reverse_thrust():
    ctl = controller(max_tilt_rad=0.4)
    actual = ctl.compute(state(), TrajectorySetpoint(state().position_w + [100, 0, 0]), 0.01)
    assert np.arccos(ctl.desired_rotation_w[2, 2]) <= 0.4 + 1e-12
    assert np.all(np.isfinite(actual.vector))
    limited = controller(max_acceleration_m_s2=2.0)
    actual = limited.compute(state(), TrajectorySetpoint(state().position_w + [0, 0, 100]), 0.01)
    assert actual.force_b[2] == pytest.approx(mass().mass_kg * 11.81)
    downward = controller().compute(state(), TrajectorySetpoint(state().position_w + [0, 0, -100]), 0.01)
    np.testing.assert_array_equal(downward.force_b, np.zeros(3))
    assert np.all(np.isfinite(downward.vector))


def test_integral_allocation_rollback_unwind_and_reset():
    ctl = controller(position_ki=[0.0, 0.0, 1.0])
    sp = TrajectorySetpoint(state().position_w + [0, 0, 0.2])
    ctl.compute(state(), sp, 0.1)
    assert ctl.integral_error_m_s[2] == pytest.approx(0.02)
    bad = AllocationResult(np.zeros(4), Wrench.zero(), np.ones(6), True)
    ctl.notify_allocation(bad)
    np.testing.assert_array_equal(ctl.integral_error_m_s, np.zeros(3))
    ctl.compute(state(0.1), sp, 0.1)
    np.testing.assert_array_equal(ctl.integral_error_m_s, np.zeros(3))
    ctl.notify_allocation(AllocationResult(np.ones(4), Wrench.zero(), np.zeros(6), True))
    ctl.compute(state(0.2), sp, 0.1)
    assert ctl.integral_error_m_s[2] == pytest.approx(0.02)
    # Saturation rolls back only growing components, not useful unwinding.
    ctl.compute(state(0.3), replace(sp, position_w=state().position_w - [0, 0, 0.1]), 0.1)
    ctl.notify_allocation(bad)
    assert ctl.integral_error_m_s[2] == pytest.approx(0.01)
    ctl.reset()
    np.testing.assert_array_equal(ctl.integral_error_m_s, np.zeros(3))
    ctl.compute(state(), sp, 0.1)


def test_internal_limits_do_not_wind_up_integrator():
    ctl = controller(position_ki=[1, 1, 1], max_acceleration_m_s2=1.0)
    ctl.compute(state(), TrajectorySetpoint(state().position_w + [100, 0, 0]), 0.1)
    np.testing.assert_array_equal(ctl.integral_error_m_s, np.zeros(3))


def test_force_derivatives_support_nonuniform_steps():
    ctl = controller(position_kp=[0, 0, 0], velocity_kd=[0, 0, 0])
    # A quadratic force trajectory has an exact three-point derivative.
    for time, dt in [(0.0, 0.1), (0.1, 0.1), (0.25, 0.15)]:
        acceleration = np.array([0.2 * time * time, 0.1 * time, 0.0])
        ctl.compute(state(time), TrajectorySetpoint(state().position_w, acceleration_w=acceleration), dt)
    force = mass().mass_kg * np.array([0.2 * 0.25**2, 0.1 * 0.25, 9.81])
    derivative = mass().mass_kg * np.array([0.4 * 0.25, 0.1, 0.0])
    second = mass().mass_kg * np.array([0.4, 0, 0])
    _, expected_omega, expected_alpha = desired_attitude_kinematics(force, derivative, second, 0, 0, 0)
    np.testing.assert_allclose(ctl.desired_angular_velocity_d, expected_omega, atol=1e-14)
    np.testing.assert_allclose(ctl.desired_angular_acceleration_d, expected_alpha, atol=1e-14)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_tilt_rad", np.pi / 2),
        ("position_kp", [-1, 0, 0]),
        ("max_yaw_rate_rad_s", -1),
        ("max_acceleration_m_s2", 0),
        ("velocity_kd", [1, 2]),
        ("position_ki", [0, np.nan, 0]),
    ],
)
def test_controller_rejects_invalid_configuration(key, value):
    with pytest.raises(ValueError):
        controller(**{key: value})


def test_controller_rejects_invalid_dt_and_repeated_time():
    ctl = controller()
    sp = TrajectorySetpoint(state().position_w)
    with pytest.raises(ValueError):
        ctl.compute(state(), sp, 0)
    ctl.compute(state(), sp, 0.01)
    with pytest.raises(ValueError):
        ctl.compute(state(), sp, 0.01)


def test_allocator_exact_feasible_wrench_with_asymmetric_geometry():
    matrix = quad_matrix()
    true_thrust = np.array([1.0, 2.0, 3.0, 2.5])
    target = matrix @ true_thrust
    result = BoundedAllocator(matrix, np.ones(6), 0).allocate(Wrench.from_vector(target), np.zeros(4), np.full(4, 8.0))
    np.testing.assert_allclose(result.thrusts_n, true_thrust, atol=1e-12)
    np.testing.assert_allclose(result.residual, np.zeros(6), atol=1e-12)
    assert not result.saturated


def test_allocator_saturated_solution_satisfies_kkt_optimality():
    rng = np.random.default_rng(18)
    for _ in range(24):
        matrix = rng.normal(size=(6, 4))
        weights = rng.uniform(0.1, 3.0, size=6)
        lower = rng.uniform(0, 0.5, size=4)
        upper = lower + rng.uniform(0.5, 2.0, size=4)
        previous = rng.uniform(0, 2, size=4)
        target = rng.normal(size=6) * 8
        regularization = 0.04
        result = BoundedAllocator(matrix, weights, regularization).allocate(
            Wrench.from_vector(target), lower, upper, previous
        )
        thrust = result.thrusts_n
        assert np.all(thrust >= lower) and np.all(thrust <= upper)
        gradient = matrix.T @ (weights**2 * (matrix @ thrust - target)) + regularization * (thrust - previous)
        at_lower, at_upper = np.isclose(thrust, lower), np.isclose(thrust, upper)
        assert np.all(gradient[at_lower] >= -1e-8)
        assert np.all(gradient[at_upper] <= 1e-8)
        np.testing.assert_allclose(gradient[~(at_lower | at_upper)], 0, atol=1e-8)
        np.testing.assert_allclose(result.achieved_wrench.vector + result.residual, target, atol=1e-13)


def test_allocator_better_than_pseudoinverse_clipping():
    matrix = quad_matrix()
    weights = np.array([1, 1, 1, 8, 8, 1.0])
    target = np.array([0, 0, 12, 1.6, -1.0, 0.3])
    lower, upper = np.zeros(4), np.array([2, 5, 4, 3.0])
    result = BoundedAllocator(matrix, weights, 0).allocate(Wrench.from_vector(target), lower, upper)
    clipped = np.clip(np.linalg.lstsq(weights[:, None] * matrix, weights * target, rcond=None)[0], lower, upper)
    assert (
        np.linalg.norm(weights * (matrix @ result.thrusts_n - target))
        < np.linalg.norm(weights * (matrix @ clipped - target)) - 1e-3
    )


def test_allocator_failed_rotor_and_all_motors_disabled():
    allocator = BoundedAllocator(quad_matrix(), np.ones(6), 0)
    target = Wrench(np.array([0, 0, 10.0]), np.array([0.3, -0.2, 0.0]))
    result = allocator.allocate(target, np.zeros(4), np.array([0, 5, 5, 5.0]))
    assert result.thrusts_n[0] == 0 and result.saturated
    all_failed = allocator.allocate(target, np.zeros(4), np.zeros(4))
    np.testing.assert_array_equal(all_failed.thrusts_n, np.zeros(4))
    np.testing.assert_array_equal(all_failed.residual, target.vector)


def test_allocator_rank_deficiency_and_regularization_reference():
    matrix = np.zeros((6, 4))
    matrix[2] = 1
    previous = np.array([1, 2, 3, 4.0])
    result = BoundedAllocator(matrix, np.ones(6), 0.1).allocate(
        Wrench(np.array([0, 0, 10.0]), np.zeros(3)), np.zeros(4), np.full(4, 5.0), previous
    )
    np.testing.assert_allclose(result.thrusts_n, previous, atol=1e-12)
    unregularized = BoundedAllocator(matrix, np.ones(6), 0).allocate(
        Wrench(np.array([0, 0, 10.0]), np.zeros(3)), np.zeros(4), np.full(4, 5.0)
    )
    assert np.sum(unregularized.thrusts_n) == pytest.approx(10.0)


def test_allocator_rejects_invalid_bounds_and_weights():
    with pytest.raises(ValueError):
        BoundedAllocator(quad_matrix(), np.zeros(6), 0)
    with pytest.raises(ValueError):
        BoundedAllocator(quad_matrix(), np.ones(6), -1)
    allocator = BoundedAllocator(quad_matrix(), np.ones(6), 0)
    for lower, upper in [(np.ones(4), np.zeros(4)), (-np.ones(4), np.ones(4)), (np.zeros(4), np.full(4, np.inf))]:
        with pytest.raises(ValueError):
            allocator.allocate(Wrench.zero(), lower, upper)


def test_yaw_acceleration_is_zero_outside_clipped_rate_interval():
    ctl = controller(max_yaw_rate_rad_s=0.3)
    sp = TrajectorySetpoint(state().position_w, yaw_rate_rad_s=4.0, yaw_acceleration_rad_s2=-2.0)
    ctl.compute(state(), sp, 0.01)
    np.testing.assert_allclose(ctl.desired_angular_acceleration_d, np.zeros(3), atol=1e-15)


def test_feedforward_transports_tilted_desired_rates_to_current_body():
    angle = 0.24
    q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    rotation = quaternion_to_matrix(q)
    omega = np.array([0.3, -0.1, 0.2])
    yaw_rate, yaw_alpha = 0.4, -0.12
    ctl = controller(attitude_kp=[0, 0, 0], angular_rate_kd=[0, 0, 0])
    sp = TrajectorySetpoint(state().position_w, yaw_rate_rad_s=yaw_rate, yaw_acceleration_rad_s2=yaw_alpha)
    actual = ctl.compute(state(quaternion_wxyz=q, angular_velocity_b=omega), sp, 0.01)
    transported_rate = rotation.T @ np.array([0, 0, yaw_rate])
    transported_alpha = rotation.T @ np.array([0, 0, yaw_alpha]) - np.cross(omega, transported_rate)
    expected = mass().inertia_com_b @ transported_alpha + np.cross(omega, mass().inertia_com_b @ omega)
    np.testing.assert_allclose(actual.torque_b, expected, atol=1e-15)


def test_nonvertical_gravity_is_compensated_as_a_full_vector():
    gravity = np.array([-0.6, 0.3, -8.7])
    ctl = GeometricController(config(), mass(), gravity)
    actual = ctl.compute(state(), TrajectorySetpoint(state().position_w), 0.01)
    desired_z = -gravity / np.linalg.norm(gravity)
    np.testing.assert_allclose(ctl.desired_rotation_w[:, 2], desired_z, atol=1e-15)
    assert actual.force_b[2] == pytest.approx(-gravity[2] * mass().mass_kg)


def test_analytic_jerk_snap_produce_correct_first_frame_attitude_derivatives():
    # Differentiate independently reconstructed reference rotation, not the
    # controller's history, so first-frame feedforward cannot silently be zero.
    acceleration = np.array([0.4, -0.2, 0.3])
    jerk, snap = np.array([0.5, 0.2, -0.1]), np.array([-0.1, 0.15, 0.2])
    yaw, rate, alpha = 0.3, 0.2, -0.07
    ctl = controller(position_kp=[0] * 3, velocity_kd=[0] * 3)
    target = TrajectorySetpoint(
        state().position_w,
        acceleration_w=acceleration,
        yaw_rad=yaw,
        yaw_rate_rad_s=rate,
        yaw_acceleration_rad_s2=alpha,
        jerk_w=jerk,
        snap_w=snap,
    )
    ctl.compute(state(), target, 0.01)

    def rotation(t):
        force = acceleration + jerk * t + 0.5 * snap * t * t + np.array([0, 0, 9.81])
        z = force / np.linalg.norm(force)
        angle = yaw + rate * t + 0.5 * alpha * t * t
        heading = np.array([np.cos(angle), np.sin(angle), 0])
        y = np.cross(z, heading)
        y /= np.linalg.norm(y)
        return np.column_stack([np.cross(y, z), y, z])

    h = 0.0001
    center, plus, minus = rotation(0), rotation(h), rotation(-h)
    first, second = (plus - minus) / (2 * h), (plus - 2 * center + minus) / (h * h)
    expected_omega = vee(center.T @ first)
    expected_alpha = vee(center.T @ second - hat(expected_omega) @ hat(expected_omega))
    np.testing.assert_allclose(ctl.desired_angular_velocity_d, expected_omega, atol=2e-9)
    np.testing.assert_allclose(ctl.desired_angular_acceleration_d, expected_alpha, atol=3e-8)
    assert ctl.force_derivative_source == "analytic_reference_and_numerical_feedback"


def test_limited_force_does_not_use_unprojected_reference_derivatives():
    ctl = controller(max_acceleration_m_s2=0.1)
    sp = TrajectorySetpoint(state().position_w, acceleration_w=[10, 0, 0], jerk_w=[100, 0, 0], snap_w=[200, 0, 0])
    ctl.compute(state(), sp, 0.01)
    assert ctl.force_derivative_source == "numerical_total_force"
    np.testing.assert_array_equal(ctl.desired_angular_velocity_d, np.zeros(3))
    sp = replace(sp, acceleration_w=np.zeros(3))
    ctl.compute(state(0.01), sp, 0.01)
    assert ctl.force_derivative_source == "numerical_total_force"


def test_trajectory_high_derivatives_are_paired_and_finite():
    with pytest.raises(ValueError, match="together"):
        TrajectorySetpoint(np.zeros(3), jerk_w=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        TrajectorySetpoint(np.zeros(3), jerk_w=np.zeros(3), snap_w=[0, 0, np.nan])
