"""Independent physical/time invariants for basic flight telemetry.

Scripted states below are measurement fixtures, not a replacement simulator or
evidence that the ARL controller has been validated in flight.
"""

from dataclasses import replace

import numpy as np
import pytest

from isaac_drone.config import load_config
from isaac_drone.core.rotations import quaternion_to_matrix
from isaac_drone.telemetry.measurements import build_basic_record
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench


def mass_properties():
    return MassProperties(2.1, np.array([[.040, .006, -.002], [.006, .055, .004],
                                         [-.002, .004, .070]]), np.array([.03, -.02, .01]))


def state(time, *, position=(1., 2., 3.), velocity=(0., 0., 0.), omega=(0., 0., 0.),
          quaternion=(1., 0., 0., 0.)):
    return VehicleState(time, np.array(position), np.array(quaternion),
                        np.array(velocity), np.array(omega))


def quaternion_for_rotation_vector(vector):
    length = np.linalg.norm(vector)
    if length == 0:
        return np.array([1., 0., 0., 0.])
    return np.r_[np.cos(length/2), np.sin(length/2)*np.asarray(vector)/length]


def basic(before, after, *, reference=None, reference_time=None, telemetry=None):
    return build_basic_record(before, after,
        TrajectorySetpoint(np.array([2., 4., 6.])) if reference is None else reference,
        before.time_s if reference_time is None else reference_time,
        mass_properties(), {} if telemetry is None else telemetry,
        Wrench([0, 0, 30], [.1, .2, .3]), Wrench([0, 0, 29], [.08, .1, .25]),
        Wrench([1, 0, 0], [0, .05, 0]))


def test_first_completed_interval_with_nonzero_velocity_has_no_reset_impulse():
    velocity = np.array([3., -2., 1.])
    before = state(0, velocity=velocity)
    after = state(.005, velocity=velocity, position=before.position_w + velocity*.005)
    record = basic(before, after)
    assert record["step_start_time_s"] == 0
    assert record["sample_time_s"] == .005
    assert record["interval_dt_s"] == .005
    np.testing.assert_array_equal(record["acceleration_w_m_s2"], np.zeros(3))
    np.testing.assert_array_equal(record["force_net_w_n"], np.zeros(3))
    np.testing.assert_allclose(record["position_error_w_m"], [2, 4, 6]-after.position_w)


def test_freefall_acceleration_and_net_force_include_gravity():
    gravity = np.array([0., 0., -9.81])
    before = state(1.0, velocity=[.4, -.3, 2.])
    dt = .007
    after = replace(before, time_s=before.time_s+dt, linear_velocity_w=before.linear_velocity_w+gravity*dt)
    record = basic(before, after)
    np.testing.assert_allclose(record["acceleration_w_m_s2"], gravity, atol=1e-12)
    np.testing.assert_allclose(record["force_net_w_n"], mass_properties().mass_kg*gravity, atol=1e-12)
    # A commanded force is a separate quantity, not the inferred net force.
    np.testing.assert_array_equal(record["force_command_b_n"], [0, 0, 30])


def test_full_inertia_constant_angular_rate_still_requires_gyroscopic_torque():
    omega = np.array([1.3, -.7, .6])
    dt = 1e-4
    before = state(0, omega=omega)
    after = state(dt, omega=omega, quaternion=quaternion_for_rotation_vector(omega*dt))
    record = basic(before, after)
    # Rotation about omega leaves that world vector unchanged. A nonspherical
    # inertia's angular momentum is nonetheless changing direction.
    np.testing.assert_allclose(record["angular_acceleration_w_rad_s2"], np.zeros(3), atol=3e-12)
    expected_instantaneous = np.cross(omega, mass_properties().inertia_com_b @ omega)
    assert np.linalg.norm(expected_instantaneous) > .01
    np.testing.assert_allclose(record["torque_net_w_nm"], expected_instantaneous, atol=4e-6)
    diagonal_only = np.cross(omega, np.diag(mass_properties().inertia_com_b)*omega)
    assert not np.allclose(record["torque_net_w_nm"], diagonal_only, atol=1e-4)


def test_conserved_world_angular_momentum_has_zero_net_torque_with_rotating_body():
    before = state(.2, omega=[.7, -.4, 1.1], quaternion=quaternion_for_rotation_vector([.1, .2, -.15]))
    rotation_before = quaternion_to_matrix(before.quaternion_wxyz)
    quaternion_after = quaternion_for_rotation_vector([.12, .18, -.14])
    rotation_after = quaternion_to_matrix(quaternion_after)
    inertia = mass_properties().inertia_com_b
    conserved_momentum = rotation_before @ inertia @ before.angular_velocity_b
    omega_after_b = np.linalg.solve(inertia, rotation_after.T @ conserved_momentum)
    after = state(.205, omega=omega_after_b, quaternion=quaternion_after)
    record = basic(before, after)
    np.testing.assert_allclose(record["torque_net_w_nm"], np.zeros(3), atol=1e-13)
    assert np.linalg.norm(record["angular_acceleration_w_rad_s2"]) > .01


def test_angular_acceleration_uses_world_difference_then_endpoint_body_frame():
    before = state(1, omega=[.4, -.2, .7], quaternion=quaternion_for_rotation_vector([.2, -.1, .4]))
    after = state(1.01, omega=[.5, -.1, .6], quaternion=quaternion_for_rotation_vector([.21, -.08, .39]))
    rotation_before = quaternion_to_matrix(before.quaternion_wxyz)
    rotation_after = quaternion_to_matrix(after.quaternion_wxyz)
    expected_w = (rotation_after @ after.angular_velocity_b - rotation_before @ before.angular_velocity_b)/.01
    record = basic(before, after)
    np.testing.assert_allclose(record["angular_acceleration_w_rad_s2"], expected_w, atol=1e-12)
    np.testing.assert_allclose(record["angular_acceleration_b_rad_s2"], rotation_after.T @ expected_w, atol=1e-12)
    assert not np.allclose(record["angular_acceleration_b_rad_s2"],
                           (after.angular_velocity_b-before.angular_velocity_b)/.01)


def test_quaternion_double_cover_does_not_create_torque_or_acceleration():
    before = state(0, omega=[.2, -.1, .7], quaternion=quaternion_for_rotation_vector([.2, -.4, .1]))
    after = replace(before, time_s=.01, quaternion_wxyz=-before.quaternion_wxyz)
    record = basic(before, after)
    np.testing.assert_array_equal(record["torque_net_w_nm"], np.zeros(3))
    np.testing.assert_array_equal(record["angular_acceleration_w_rad_s2"], np.zeros(3))


def test_absent_motor_measurements_stay_unavailable_and_reference_stays_held():
    before, after = state(.02), state(.025, position=[3, 2, 1])
    record = basic(before, after, reference_time=0)
    assert record["reference_sample_time_s"] == 0
    np.testing.assert_array_equal(record["target_position_w_m"], [2, 4, 6])
    np.testing.assert_array_equal(record["position_error_w_m"], [-1, 2, 5])
    assert record["position_error_norm_m"] == pytest.approx(np.sqrt(30))
    for key in ("motor_speed_rps", "motor_speed_rpm", "motor_speed_rad_s", "force_motor_b_n",
                "torque_motor_b_nm", "force_submitted_b_n", "torque_submitted_b_nm"):
        assert record[key] is None
    assert record["motor_speed_source"] == "unavailable"


@pytest.mark.parametrize("after_time,reference_time", [(0, 0), (.01, .001), (.01, -1)])
def test_incomplete_interval_or_future_reference_rejected(after_time, reference_time):
    with pytest.raises(ValueError):
        basic(state(0), state(after_time), reference_time=reference_time)


class ScriptedBackend:
    """Analytic truth fixtures advanced explicitly between prepare and finish."""
    allocation_matrix_b = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1],
        [.1, -.1, .1, -.1], [.1, .1, -.1, -.1], [-.07, .07, .07, -.07]])

    def __init__(self):
        self.reset_count = 0
        self.acceleration = np.array([.3, -.2, .5])

    def reset(self):
        self.reset_count += 1
        self.current_time = 0.0
        self.initial_velocity = np.array([1., -.5, .2])*self.reset_count
        self.calls = []

    def mass_properties(self):
        return mass_properties()

    def read_state(self, time_s):
        assert time_s == pytest.approx(self.current_time)
        position = np.array([0., 0., 2.])+self.initial_velocity*time_s+.5*self.acceleration*time_s*time_s
        return state(time_s, position=position, velocity=self.initial_velocity+self.acceleration*time_s)

    def motor_parameters(self):
        return {"sampled_kf": np.full(4, 1e-5), "thrust_min_n": np.full(4, .1),
                "thrust_max_n": np.full(4, 10.), "rps_min": np.full(4, 100.),
                "rps_max": np.full(4, 1000.)}

    def thrust_to_motor_speeds(self, thrust):
        return np.sqrt(thrust/1e-5)

    def apply_motor_speeds(self, target_rps, external):
        self.apply(1e-5*target_rps**2, external)

    def apply(self, thrust, external):
        self.calls.append((thrust.copy(), external))

    def advance(self, dt):
        self.current_time += dt

    def telemetry(self):
        return {}


class BiasedEstimate:
    def __init__(self, backend):
        self.backend = backend

    def reset(self):
        pass

    def read_state(self, time_s):
        truth = self.backend.read_state(time_s)
        return replace(truth, position_w=truth.position_w+[100, 0, 0],
                       linear_velocity_w=truth.linear_velocity_w+[20, 0, 0])


class CountingTrajectory:
    """A duck-typed (non-segmented) trajectory: no planning or screening is applied to it."""

    def reset(self, initial_state, start_time_s=None):
        self.start_time_s = initial_state.time_s if start_time_s is None else start_time_s
        self.sample_times = []

    def phase(self, time_s):
        return "scripted"

    def sample(self, time_s):
        self.sample_times.append(time_s)
        return TrajectorySetpoint(np.array([3+time_s, 4., 5.]), velocity_w=np.array([1., 0., 0.]))


def test_runtime_truth_endpoint_reference_decimation_and_reset_do_not_mix_times():
    cfg = load_config()
    cfg["simulation"]["control_decimation"] = 3
    backend, trajectory = ScriptedBackend(), CountingTrajectory()
    loop = MotionControlLoop(cfg, backend, trajectory=trajectory, state_provider=BiasedEstimate(backend))
    dt = cfg["simulation"]["dt"]
    for run_length in (7, 1):
        loop.reset()
        for index in range(run_length):
            loop.prepare_step()
            backend.advance(dt)
            record = loop.finish_step()
            actual = record["basic"]
            reference_time = (index//3)*3*dt
            truth = backend.read_state((index+1)*dt)
            assert record["time_s"] == pytest.approx(index*dt)
            assert actual["step_start_time_s"] == pytest.approx(index*dt)
            assert actual["sample_time_s"] == pytest.approx((index+1)*dt)
            assert actual["reference_sample_time_s"] == pytest.approx(reference_time)
            np.testing.assert_allclose(actual["position_w_m"], truth.position_w)
            np.testing.assert_allclose(actual["velocity_w_m_s"], truth.linear_velocity_w)
            np.testing.assert_allclose(actual["acceleration_w_m_s2"], backend.acceleration, atol=2e-12)
            np.testing.assert_allclose(actual["force_net_w_n"], mass_properties().mass_kg*backend.acceleration, atol=4e-12)
            np.testing.assert_allclose(actual["target_position_w_m"], [3+reference_time, 4, 5])
            np.testing.assert_allclose(actual["position_error_w_m"], np.array([3+reference_time, 4, 5])-truth.position_w)
            # Controller consumes a biased estimate, basic actual values must not.
            assert record["state"]["position_w"][0] > 99
        np.testing.assert_allclose(trajectory.sample_times, np.arange(0, run_length, 3)*dt)


def test_interval_snapshots_survive_mutated_provider_arrays_and_reference_objects():
    class ReusedBuffersBackend(ScriptedBackend):
        def read_state(self, time_s):
            result = super().read_state(time_s)
            self.last_returned = result
            return result

        def advance(self, dt):
            # Reuse/mutate array storage after it was handed to prepare_step.
            # The interval's left endpoint must already have been copied.
            self.last_returned.linear_velocity_w[:] += self.acceleration*dt
            self.last_returned.position_w[:] += 999
            super().advance(dt)

    class MutableTrajectory(CountingTrajectory):
        def sample(self, time_s):
            self.last_reference = super().sample(time_s)
            return self.last_reference

    cfg = load_config()
    backend, trajectory = ReusedBuffersBackend(), MutableTrajectory()
    loop = MotionControlLoop(cfg, backend, trajectory=trajectory)
    loop.reset()
    loop.prepare_step()
    trajectory.last_reference.position_w[:] += 1000
    backend.advance(cfg["simulation"]["dt"])
    record = loop.finish_step()["basic"]
    np.testing.assert_allclose(record["acceleration_w_m_s2"], backend.acceleration, atol=1e-12)
    np.testing.assert_array_equal(record["target_position_w_m"], [3, 4, 5])


def test_optional_native_acceleration_remains_distinct_from_interval_acceleration():
    before, after = state(0), state(.01, velocity=[.01, -.02, .03])
    native = np.array([.8, -1.9, 3.1])
    record = basic(before, after, telemetry={"linear_acceleration_com_w_m_s2": native,
                    "linear_acceleration_source": "native_fixture"})
    np.testing.assert_allclose(record["acceleration_w_m_s2"], [1, -2, 3])
    np.testing.assert_array_equal(record["acceleration_native_w_m_s2"], native)
    assert record["acceleration_native_source"] == "native_fixture"
    np.testing.assert_allclose(record["force_net_w_n"], mass_properties().mass_kg*np.array([1, -2, 3]))
