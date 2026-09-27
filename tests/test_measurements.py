"""Independent kinematics and angular-momentum checks for basic flight logging."""
import numpy as np
import pytest

from isaac_drone.measurements import build_basic_record
from isaac_drone.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench


def state(time, pos=(0, 0, 1), vel=(0, 0, 0), omega=(0, 0, 0), quat=(1, 0, 0, 0)):
    return VehicleState(time, pos, quat, vel, omega)


MASS = MassProperties(2.0, np.array([[.3, .02, 0], [.02, .4, .03], [0, .03, .5]]), [.02, -.01, .03])


def record(before, after, target=None, telemetry=None, reference_time=0):
    return build_basic_record(before, after, target or TrajectorySetpoint([0, 0, 1]), reference_time,
                              MASS, telemetry or {}, Wrench([0, 0, 20], [1, 2, 3]),
                              Wrench([0, 0, 19], [.8, 1.8, 2.8]), Wrench([1, 0, 0], [0, .2, 0]))


def test_first_completed_step_records_actual_not_target_acceleration_and_gravity():
    dt = .005
    before = state(0, vel=(2, 3, 4))
    after = state(dt, vel=(2, 3, 4-9.81*dt))
    target = TrajectorySetpoint([1, 2, 3], acceleration_w=[7, 8, 9])
    actual = record(before, after, target)
    np.testing.assert_allclose(actual["acceleration_w_m_s2"], [0, 0, -9.81])
    np.testing.assert_allclose(actual["force_net_w_n"], [0, 0, -19.62])
    np.testing.assert_allclose(actual["target_acceleration_w_m_s2"], [7, 8, 9])
    assert actual["sample_time_s"] == dt
    assert actual["motor_speed_rps"] is None
    assert actual["force_submitted_b_n"] is None


def test_position_error_is_target_minus_actual_endpoint_and_norm_is_euclidean():
    result = record(state(1, pos=(3, 3, 3)), state(1.1, pos=(2, 4, 5)),
                    TrajectorySetpoint([5, 8, 5]), reference_time=.9)
    np.testing.assert_allclose(result["position_error_w_m"], [3, 4, 0])
    assert result["position_error_norm_m"] == 5
    assert result["reference_sample_time_s"] == .9
    assert result["step_start_time_s"] == 1
    assert result["sample_time_s"] == 1.1


def test_world_angular_velocity_difference_does_not_subtract_rotating_components():
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    # Same world angular velocity, different body coordinates at the endpoints.
    result = record(state(0, omega=(1, 0, 0)), state(.1, omega=(0, -1, 0), quat=q))
    np.testing.assert_allclose(result["angular_acceleration_w_rad_s2"], 0, atol=1e-14)
    np.testing.assert_allclose(result["angular_acceleration_b_rad_s2"], 0, atol=1e-14)
    # Full non-spherical inertia rotated 90deg changes world angular momentum.
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = (rotation @ MASS.inertia_com_b @ [0, -1, 0] - MASS.inertia_com_b @ [1, 0, 0]) / .1
    np.testing.assert_allclose(result["torque_net_w_nm"], expected, atol=1e-14)


def test_net_wrench_is_not_confused_with_submitted_wrench():
    telemetry = {"motor_wrench_b": [0, 0, 18, .2, .3, .4],
                 "total_wrench_b": [1, 0, 18, .2, .5, .4], "has_submitted_wrench": True,
                 "motor_speed_rps": [100, 200, 300, 400],
                 "motor_speed_rpm": [6000, 12000, 18000, 24000],
                 "motor_speed_rad_s": np.array([100, 200, 300, 400])*2*np.pi,
                 "motor_speed_source": "native_state"}
    result = record(state(0), state(.1, vel=(0, 0, -.081)), telemetry=telemetry)
    np.testing.assert_allclose(result["force_submitted_b_n"], [1, 0, 18])
    np.testing.assert_allclose(result["force_motor_b_n"], [0, 0, 18])
    np.testing.assert_allclose(result["force_net_w_n"], [0, 0, -1.62])
    np.testing.assert_allclose(result["motor_speed_rpm"], [6000, 12000, 18000, 24000])


@pytest.mark.parametrize("after_time", [0, -.1])
def test_bad_interval_is_rejected(after_time):
    # Both states remain legal; ordering of samples is not.
    before, after = state(1), state(1+after_time)
    with pytest.raises(ValueError, match="positive completed"):
        record(before, after)


def test_future_reference_and_malformed_backend_signals_are_rejected():
    with pytest.raises(ValueError, match="no later"):
        record(state(0), state(.1), reference_time=.05)
    with pytest.raises(ValueError, match="motor_speed_rps"):
        record(state(0), state(.1), telemetry={"motor_speed_rps": [1, 2]})


def test_unsubmitted_wrench_is_unknown_instead_of_zero_after_reset():
    result = record(state(0), state(.1), telemetry={"total_wrench_b": [0]*6, "has_submitted_wrench": False})
    assert result["force_submitted_b_n"] is None
    assert result["torque_submitted_b_nm"] is None
