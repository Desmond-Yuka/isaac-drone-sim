"""Cascaded PID unit behaviour, the controller registry and the rotor-thrust output path."""

import numpy as np
import pytest

from isaac_drone.config import CONFIG_DIR, load_config
from isaac_drone.control import CONTROLLERS, AllocationResult, FlightLimits, VehicleModel, build_controller
from isaac_drone.core.types import MassProperties, TrajectorySetpoint, VehicleState, Wrench
from isaac_drone.core.validation import ConfigurationError
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.sim.synthetic import SyntheticBackend

PID_CONFIG = CONFIG_DIR / "helix_cascaded_pid.yaml"
INERTIA = np.array([[0.02, 0.001, 0.0], [0.001, 0.023, 0.002], [0.0, 0.002, 0.026]])


def vehicle():
    return VehicleModel(
        MassProperties(1.3, INERTIA, np.zeros(3)),
        np.array([0.0, 0.0, -9.81]),
        np.zeros((6, 4)),
        np.full(4, 0.1),
        np.full(4, 10.0),
        0.01,
    )


def controller(**changes):
    section = {**load_config(PID_CONFIG)["controller"], **changes}
    return build_controller(section, vehicle=vehicle(), limits=FlightLimits(1.2, 6.0, None))


def state(position=(0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), time=0.0):
    return VehicleState(time, np.array(position), np.array([1.0, 0.0, 0.0, 0.0]), np.array(velocity), np.array(omega))


def test_registered_with_strict_parameters():
    assert {"geometric", "cascaded_pid"} <= set(CONTROLLERS.names())
    section = load_config(PID_CONFIG)["controller"]
    for bad in (
        {"rate_kp": [1.0, 2.0]},
        {"attitude_kp": [-1.0, 1.0, 1.0]},
        {"max_body_rate_rad_s": [0.0, 1.0, 1.0]},
        {"velocity_kp": [1.0, True, 1.0]},
        {"unknown_gain": 1.0},
    ):
        with pytest.raises(ConfigurationError):
            CONTROLLERS.parse({**section, **bad}, "controller")


def test_hover_at_setpoint_commands_weight_and_no_torque():
    pid = controller()
    wrench = pid.compute(state(), TrajectorySetpoint(np.array([0.0, 0.0, 1.0])), 0.01)
    np.testing.assert_allclose(wrench.force_b, [0.0, 0.0, 1.3 * 9.81], atol=1e-12)
    np.testing.assert_allclose(wrench.torque_b, 0.0, atol=1e-12)
    diagnostics = pid.diagnostics()
    np.testing.assert_allclose(diagnostics.desired_rotation_w, np.eye(3), atol=1e-12)
    assert diagnostics.details["thrust_vector_limited"] is False


def test_position_error_tilts_thrust_toward_target_and_rate_loop_uses_full_inertia():
    pid = controller()
    pid.compute(state(), TrajectorySetpoint(np.array([1.0, 0.0, 1.0])), 0.01)
    desired = pid.diagnostics().desired_rotation_w
    assert desired[0, 2] > 0  # body z leans toward +x
    rate_setpoint = np.array(pid.diagnostics().details["rate_setpoint_b_rad_s"])
    assert rate_setpoint[1] > 0  # positive pitch rate tilts +z toward +x
    omega = np.array([0.3, -0.2, 0.1])
    fresh = controller()
    wrench = fresh.compute(state(omega=omega), TrajectorySetpoint(np.array([0.0, 0.0, 1.0])), 0.01)
    gains = np.array(load_config(PID_CONFIG)["controller"]["rate_kp"])
    expected = INERTIA @ (gains * (np.array(fresh.diagnostics().details["rate_setpoint_b_rad_s"]) - omega))
    np.testing.assert_allclose(wrench.torque_b, expected + np.cross(omega, INERTIA @ omega), atol=1e-12)


def test_integrals_do_not_grow_while_allocation_is_saturated():
    pid = controller(velocity_ki=[2.0, 2.0, 2.0])
    reference = TrajectorySetpoint(np.array([0.0, 0.0, 1.0]), velocity_w=np.array([1.0, 0.0, 0.0]))
    pid.compute(state(), reference, 0.01)
    grown = pid._velocity_integral.copy()
    assert grown[0] > 0
    pid.compute(state(time=0.01), reference, 0.01)
    pid.notify_allocation(AllocationResult(np.zeros(4), Wrench.zero(), np.ones(6), True))
    np.testing.assert_array_equal(pid._velocity_integral, grown)


class DirectThrust:
    """A policy-like controller that commands rotor thrusts directly (no allocation)."""

    output = "rotor_thrust"

    def __init__(self, thrusts):
        self.thrusts = np.asarray(thrusts, dtype=float)
        self.allocations = []

    def reset(self):
        pass

    def compute(self, state, setpoint, dt_s):
        return self.thrusts

    def notify_allocation(self, result):
        self.allocations.append(result)

    def diagnostics(self):
        from isaac_drone.control import ControllerDiagnostics

        return ControllerDiagnostics()


def test_rotor_thrust_controllers_bypass_allocation_and_are_clipped_to_bounds():
    config = load_config()
    backend = SyntheticBackend(config)
    policy = DirectThrust([-1.0, 3.0, 3.2, 20.0])
    loop = MotionControlLoop(config, backend, controller=policy)
    loop.reset()
    record = loop.prepare_step()
    np.testing.assert_allclose(record["allocated_thrust_n"], [0.1, 3.0, 3.2, 10.0])
    assert record["allocation_saturated"]
    assert record["controller"]["kind"] == "injected:DirectThrust"
    np.testing.assert_allclose(record["requested_wrench_b"], backend.allocation_matrix_b @ policy.thrusts)
    backend.step()
    loop.finish_step()
    assert policy.allocations and policy.allocations[0].saturated


@pytest.mark.slow
def test_preset_flies_the_moderate_helix_on_the_synthetic_plant():
    config = load_config(PID_CONFIG)
    backend = SyntheticBackend(config)
    loop = MotionControlLoop(config, backend)
    loop.reset()
    errors = []
    for _ in range(round(config["simulation"]["duration_s"] / config["simulation"]["dt"])):
        loop.prepare_step()
        backend.step()
        record = loop.finish_step()
        errors.append(record["basic"]["position_error_norm_m"])
    assert record["controller"]["kind"] == "cascaded_pid"
    assert max(errors) < 0.3
    assert record["mission"]["achieved"]
