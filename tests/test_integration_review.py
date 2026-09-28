"""Independent integration invariants; test doubles do not certify simulator flight."""

from dataclasses import replace

import numpy as np
import pytest

from isaac_drone.config import ConfigurationError, assert_flight_ready, load_config
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.core.types import MassProperties, VehicleState, Wrench


def physical_matrix(axis=(0.0, 0.0, 1.0)):
    positions = np.array([[-.105, .105, .0245], [-.105, -.105, .0245],
                          [.105, .105, .0245], [.105, -.105, .0245]])
    com = np.array([0.007, -0.004, 0.012])
    axes = np.tile(axis, (4, 1))
    directions = np.array([-1, 1, 1, -1])
    moments = np.cross(positions-com, axes) + .07 * directions[:, None] * axes
    return np.vstack((axes.T, moments.T))


def test_actual_com_geometry_is_controllable_with_explicit_diagonal_spin_pairs():
    cfg = load_config()
    actual = physical_matrix()
    assert np.linalg.matrix_rank(actual[[2, 3, 4, 5]]) == 4
    assert not np.allclose(actual, cfg["vehicle"]["allocation_matrix"])
    assert_flight_ready(cfg, actual)


@pytest.mark.parametrize("axis", [(np.sin(.2), 0.0, np.cos(.2)), (0.0, 0.0, -1.0)])
def test_full_rank_alone_does_not_make_a_nonpositive_z_vehicle_controller_compatible(axis):
    matrix = physical_matrix(axis)
    assert np.linalg.matrix_rank(matrix[[2, 3, 4, 5]]) == 4
    with pytest.raises(ConfigurationError):
        assert_flight_ready(load_config(), matrix)


class Backend:
    allocation_matrix_b = physical_matrix()

    def __init__(self):
        self.calls = []
        self.compatibility_checks = 0

    def reset(self):
        self.calls.clear()

    def assert_control_compatible(self):
        self.compatibility_checks += 1

    def mass_properties(self):
        return MassProperties(1.25, np.array([[.02, .001, 0], [.001, .023, .002], [0, .002, .026]]),
                              np.array([.007, -.004, .012]))

    def read_state(self, time_s):
        return VehicleState(time_s, np.array([.0, .0, 1.0]), np.array([1., 0., 0., 0.]),
                            np.zeros(3), np.zeros(3))

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

    def telemetry(self):
        return {}


class Effects:
    def reset(self, seed):
        self.calls = []

    def evaluate(self, truth, dt_s):
        self.calls.append((truth, dt_s))
        return Wrench.zero()


class Estimator:
    def reset(self):
        pass

    def read_state(self, time_s):
        # Deliberate estimate offset verifies effects consume physical truth.
        return replace(Backend().read_state(time_s), position_w=np.array([10., 0., 1.]))


def test_slow_control_keeps_fast_allocation_motors_effects_and_truth_separate():
    config = load_config()
    config["simulation"]["control_decimation"] = 4
    backend, effects = Backend(), Effects()
    loop = MotionControlLoop(config, backend, effects=effects, state_provider=Estimator())
    loop.reset()
    control_calls, allocation_calls = [], []
    compute, allocate = loop.controller.compute, loop.allocator.allocate
    def track_compute(state, reference, dt):
        control_calls.append((state.time_s, dt))
        return compute(state, reference, dt)
    def track_allocate(*args):
        allocation_calls.append(args[0].vector.copy())
        return allocate(*args)
    loop.controller.compute, loop.allocator.allocate = track_compute, track_allocate
    for _ in range(9):
        loop.prepare_step()
        loop.finish_step()
    assert control_calls == [(0.0, .02), (.02, .02), (.04, .02)]
    assert len(allocation_calls) == len(backend.calls) == len(effects.calls) == 9
    for truth, dt in effects.calls:
        assert dt == .005
        np.testing.assert_array_equal(truth.position_w, [0, 0, 1])
    np.testing.assert_array_equal(loop.setpoint.position_w, [10, 0, 1])


def test_prepare_exception_cannot_retry_partly_advanced_plugin_state_without_reset():
    config = load_config()
    backend, effects = Backend(), Effects()
    loop = MotionControlLoop(config, backend, effects=effects)
    loop.reset()
    def fail(*args):
        raise ValueError("injected effect failure after controller and allocator advanced")
    effects.evaluate = fail
    with pytest.raises(ValueError, match="injected effect failure"):
        loop.prepare_step()
    with pytest.raises(RuntimeError):
        loop.prepare_step()
    assert backend.calls == []
    effects.evaluate = lambda *args: Wrench.zero()
    loop.reset()
    loop.prepare_step()
    loop.finish_step()
    assert len(backend.calls) == 1


@pytest.mark.parametrize("bad_flag,value", [("kinematic_enabled", True), ("disable_gravity", True),
                                             ("rigid_body_enabled", False)])
def test_backend_rejects_actual_incompatible_flags_independently_of_yaml(bad_flag, value):
    from types import SimpleNamespace
    from isaac_drone.sim.isaaclab.backend import ARLBackend
    backend = object.__new__(ARLBackend)
    backend.sim = None
    backend.robot = SimpleNamespace(is_fixed_base=False)
    backend._allocation_rank = 4
    backend._body_physics = {"base_link": {
        "rigid_body_enabled": True, "kinematic_enabled": False, "disable_gravity": False}}
    backend.assert_control_compatible()
    backend._body_physics["base_link"][bad_flag] = value
    with pytest.raises(ValueError, match="dynamic gravity-enabled"):
        backend.assert_control_compatible()


def test_backend_cannot_claim_flight_compatibility_without_physics_audit():
    from types import SimpleNamespace
    from isaac_drone.sim.isaaclab.backend import ARLBackend
    backend = object.__new__(ARLBackend)
    backend.sim = None
    backend.robot = SimpleNamespace(is_fixed_base=False)
    backend._allocation_rank = 4
    backend._body_physics = None
    with pytest.raises(ValueError, match="not been audited"):
        backend.assert_control_compatible()
