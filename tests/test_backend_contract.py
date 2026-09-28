"""Bridge contract tests using fake bodies; these do not validate Isaac Sim."""

import types
from pathlib import Path

import numpy as np
import pytest

from isaac_drone.core.types import Wrench
from isaac_drone.sim import native_thruster
from isaac_drone.sim.actuation import NativeRpsActuator, thrust_to_motor_speeds
from isaac_drone.sim.isaaclab.backend import ARLBackend, _collision_corners_b, override_native

UPSTREAM_SOURCE = native_thruster.find_upstream_source()
NATIVE_METHODS = (
    native_thruster.load_upstream_methods(UPSTREAM_SOURCE) if UPSTREAM_SOURCE else native_thruster.TRANSCRIBED_METHODS
)


CANONICAL = ["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"]


def make_actuator(indices, kf, initial_rps=None, bounds=(0.1, 10.0), scheme="rk4", discrete=False, rate=100000.0):
    """Run the repository's actual integration method bodies with NumPy clamp.

    This verifies reuse/maths without importing Isaac or claiming CUDA/PhysX
    validation. Native tensor construction and GPU execution remain untested.
    Without an upstream checkout the package's NumPy transcription is used;
    test_transcription_matches_upstream_source then documents the skip.
    """
    methods = NATIVE_METHODS
    kf = np.asarray(kf, dtype=float).reshape(1, -1)
    n = kf.shape[1]
    initial = np.zeros_like(kf) if initial_rps is None else np.asarray(initial_rps).reshape(1, n)
    actuator = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            thrust_range=bounds,
            dt=0.005,
            integration_scheme=scheme,
            torque_to_thrust_ratio=0.07,
            thruster_names_expr=CANONICAL,
        ),
        thruster_indices=indices,
        thrust_const=kf,
        curr_thrust=kf * initial**2,
        tau_inc_s=np.full((1, n), 0.06),
        tau_dec_s=np.full((1, n), 0.005),
        max_rate=np.full((1, n), rate),
        rate_calls=0,
        mixing_calls=0,
    )
    for name in methods:
        setattr(actuator, name, types.MethodType(methods[name], actuator))
    native_rate = actuator.motor_model_rate

    def counted_rate(error, mixing):
        actuator.rate_calls += 1
        return native_rate(error, mixing)

    actuator.motor_model_rate = counted_rate
    native_mixing = actuator.discrete_mixing_factor if discrete else actuator.continuous_mixing_factor

    def counted_mixing(tau):
        actuator.mixing_calls += 1
        actuator.last_tau = tau.copy()
        return native_mixing(tau)

    actuator.mixing_factor_function = counted_mixing
    return actuator


def make_backend():
    names = ["front_right_prop", "base_link", "back_left_prop", "front_left_prop", "back_right_prop"]
    positions = np.array([[0.1, -0.1, 0], [0, 0, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0], [-0.1, -0.1, 0]])
    masses = np.array([0.1, 2.0, 0.1, 0.1, 0.1])
    coms = np.zeros((5, 3))
    coms[1] = [0.02, 0.01, 0]
    inertias = np.repeat(np.eye(3)[None] * 0.001, 5, axis=0)
    inertias[1] = np.diag([0.2, 0.3, 0.4])
    quats = np.tile([0, 0, 0, 1.0], (5, 1))
    mass_quats = quats.copy()
    mass_quats[1] = [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)]
    native_names = ["front_right_prop", "back_left_prop", "back_right_prop", "front_left_prop"]
    data = types.SimpleNamespace(
        root_link_pos_w=np.array([[1.0, 2.0, 3.0]]),
        root_link_quat_w=np.array([[0.0, 0.0, 0.0, 1.0]]),
        root_link_lin_vel_w=np.array([[0.2, 0.3, 0.4]]),
        root_link_ang_vel_w=np.array([[0.0, 0.0, 2.0]]),
        body_link_pos_w=positions[None] + [1, 2, 3],
        body_link_quat_w=quats[None],
        body_mass=masses[None],
        body_com_pos_b=coms[None],
        body_com_quat_b=mass_quats[None],
        body_inertia=inertias.reshape(1, 5, 9),
        thrust_target=np.zeros((1, 4)),
        applied_thrust=np.zeros((1, 4)),
        computed_thrust=np.zeros((1, 4)),
        default_thruster_rps=np.full((1, 4), 200.0),
        _sim_timestamp=0.0,
        has_body_ordering=True,
        body_ordering=types.SimpleNamespace(user_to_backend=np.array([4, 0, 2, 3, 1])),
    )
    actuator = make_actuator([0, 1, 2, 3], [1e-5] * 4)
    cfg = types.SimpleNamespace(
        actuators={"thrusters": actuator.cfg},
        prim_path="/lmf2",
        rotor_directions=[-1, 1, 1, -1],
        thruster_force_direction=[0, 0, 1],
        allocation_matrix=np.zeros((6, 4)),
    )
    robot = types.SimpleNamespace(
        num_instances=1,
        num_bodies=5,
        num_joints=0,
        body_names=names,
        cfg=cfg,
        thruster_names=native_names,
        data=data,
        actuators={"thrusters": actuator},
        calls=0,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("The native combined root-wrench flight path must not run")

    def reset(env_ids):
        assert env_ids == [0]
        for motor in robot.actuators.values():
            motor.curr_thrust[:] = motor.thrust_const * motor._init_thruster_rps**2
        data.thrust_target[:] = 200.0

    robot.set_thrust_target = robot._apply_actuator_model = robot._apply_combined_wrench = forbidden
    robot.reset = reset
    robot.write_root_pose_to_sim_index = lambda **kw: setattr(robot, "written_pose", kw["root_pose"])
    robot.write_root_velocity_to_sim_index = lambda **kw: setattr(robot, "written_velocity", kw["root_velocity"])
    robot.update = lambda dt: None
    robot.map_body_ids_to_backend = lambda ids: [int(data.body_ordering.user_to_backend[i]) for i in ids]
    config = {
        "simulation": {"dt": 0.005},
        "vehicle": {
            "rotor_direction_source": "simulation_diagonal_pairs",
            "initial_state": {
                "pos": [4, 5, 6],
                "quaternion_wxyz": [1, 0, 0, 0],
                "lin_vel": [1, 2, 3],
                "ang_vel": [0, 0, 2],
            },
        },
    }
    backend = ARLBackend(None, robot, config)
    backend.submissions = []
    backend._submit_body_wrenches = lambda *arrays: backend.submissions.append(tuple(x.copy() for x in arrays))
    return backend


def test_mass_properties_keep_nonzero_com_and_rotated_full_inertia():
    backend = make_backend()
    mass = backend.mass_properties()
    assert mass.mass_kg == pytest.approx(2.4)
    np.testing.assert_allclose(mass.com_b, [1 / 60, 1 / 120, 0])
    # Rotation creates off-diagonal inertia; offset masses add a separately
    # calculated parallel-axis contribution.
    assert mass.inertia_com_b[0, 1] == pytest.approx(-0.05 - 1 / 15000)
    np.testing.assert_allclose(np.diag(mass.inertia_com_b), [0.258033333333, 0.258133333333, 0.412166666667])
    state = backend.read_state(0.5)
    np.testing.assert_allclose(state.position_w, np.array([1, 2, 3]) + mass.com_b)
    np.testing.assert_allclose(state.linear_velocity_w, [0.2 - 1 / 60, 0.3 + 1 / 30, 0.4])
    np.testing.assert_allclose(state.quaternion_wxyz, [1, 0, 0, 0])


def test_motor_order_once_per_tick_and_external_wrench_is_added_once():
    backend = make_backend()
    external = Wrench([1, 2, 3], [0.1, 0.2, 0.3])
    command = np.array([200.0, 300.0, 400.0, 500.0])
    backend.apply_motor_speeds(command, external)
    actuator = backend.robot.actuators["thrusters"]
    assert actuator.mixing_calls == 1
    assert actuator.rate_calls == 4  # Native RK4 stages, one motor tick.
    assert len(backend.submissions) == 1
    np.testing.assert_allclose(backend.robot.data.thrust_target, 1e-5 * np.array([[500, 200, 300, 400]]) ** 2)
    force, torque, positions = backend.submissions[0]
    # Only explicit disturbance reaches base (backend body 0).
    np.testing.assert_allclose(force[0, 0], external.force_b)
    np.testing.assert_allclose(
        torque[0, 0], external.torque_b + np.cross(backend._mass.com_b - backend._root_com_b, external.force_b)
    )
    applied = np.asarray(backend.telemetry()["applied_thrust_n"])
    for motor, body_id in enumerate([2, 1, 3, 4]):
        np.testing.assert_allclose(force[0, body_id], [0, 0, applied[motor]])
        np.testing.assert_allclose(torque[0, body_id], [0, 0, backend._rotor_directions[motor] * 0.07 * applied[motor]])
        np.testing.assert_allclose(positions[0, body_id], 0)
    # Flight does not set any vehicle pose or velocity.
    assert not hasattr(backend.robot, "written_pose")
    assert not hasattr(backend.robot, "written_velocity")
    with pytest.raises(RuntimeError, match="twice"):
        backend.apply_motor_speeds(command, external)
    assert actuator.mixing_calls == 1


def test_wrench_submission_uses_backend_order_and_shift_to_root_body_com():
    backend = make_backend()
    wrench = Wrench([0, 0, 12], [1, 2, 3])
    force, torque, positions = backend._propeller_wrench_arrays(np.zeros(4), wrench)
    np.testing.assert_allclose(force[0, 0], wrench.force_b)
    np.testing.assert_allclose(torque[0, 0], [1 - 0.02, 2 + 0.04, 3], atol=1e-6)
    np.testing.assert_allclose(positions[0, 0], [0.02, 0.01, 0])
    assert np.count_nonzero(force[0, 1:]) == 0


def test_reset_corrects_rps_target_retaining_native_pose_semantics():
    backend = make_backend()
    backend.reset()
    assert backend.robot.calls == 0
    np.testing.assert_allclose(backend.robot.data.thrust_target, 0.4)
    np.testing.assert_allclose(backend.robot.written_pose[0, :3], [4, 5, 6])
    np.testing.assert_allclose(backend.robot.written_pose[0, 3:], [0, 0, 0, 1])
    np.testing.assert_allclose(backend.robot.written_velocity[0], [1, 2, 3, 0, 0, 2])


def test_audit_allows_rank_deficiency_but_flight_does_not():
    backend = make_backend()
    backend.robot.cfg.rotor_directions = [-1, 1, -1, 1]
    backend._extract_geometry()
    assert backend.telemetry()["collective_roll_pitch_yaw_rank"] == 3
    with pytest.raises(RuntimeError, match="independently control"):
        backend.apply_motor_speeds(np.full(4, 300.0), Wrench.zero())


def test_movable_joint_and_unknown_nativeoverride_native_are_rejected():
    backend = make_backend()
    backend.robot.num_joints = 1
    with pytest.raises(ValueError, match="Movable internal"):
        ARLBackend(None, backend.robot, backend.config)
    cfg = types.SimpleNamespace(known=types.SimpleNamespace(value=3))
    override_native(cfg, {"known": {"value": None}})
    assert cfg.known.value == 3
    with pytest.raises(ValueError, match="Unknown native field"):
        override_native(cfg, {"known": {"typo": 1}})


def test_nullable_schema_slots_and_fragments_remain_writable(monkeypatch):
    import isaac_drone.sim.isaaclab.backend as module

    def factory(name):
        return types.SimpleNamespace(mass=None, density=None, contact_offset=None)

    monkeypatch.setattr(module, "_native_cfg", factory)
    cfg = types.SimpleNamespace(mass_props=None, collision_props=None)
    override_native(cfg, {"mass_props": {"mass": 2.0}, "collision_props": {"contact_offset": 0.001}})
    assert cfg.mass_props.mass == 2.0
    assert cfg.mass_props.density is None
    assert cfg.collision_props.contact_offset == 0.001
    override_native(cfg, {"mass_props": {"/base_link": [{"_type": "MassCfg", "mass": 1.5}]}})
    assert cfg.mass_props["/base_link"][0].mass == 1.5


def test_flight_flags_are_checked_without_blocking_inspection():
    backend = make_backend()
    flags = {
        name: {"rigid_body_enabled": True, "kinematic_enabled": False, "disable_gravity": False}
        for name in backend.robot.body_names
    }
    backend._body_physics = flags
    backend.assert_control_compatible()
    backend.robot.is_fixed_base = True
    assert backend.telemetry()["fixed_base"] is True
    with pytest.raises(ValueError, match="free base"):
        backend.assert_control_compatible()
    backend.robot.is_fixed_base = False
    flags["base_link"]["disable_gravity"] = True
    with pytest.raises(ValueError, match="gravity-enabled"):
        backend.assert_control_compatible()


def test_motor_speed_uses_each_sampled_coefficient_and_direct_rps_state():
    backend = make_backend()
    actuator = backend.robot.actuators["thrusters"]
    actuator.thrust_const[:] = [[1e-5, 2e-5, 4e-5, 8e-5]]
    native_speeds = np.array([[400.0, 100.0, 200.0, 300.0]])
    backend.robot.data.default_thruster_rps[:] = native_speeds
    backend.reset()
    result = backend.telemetry()
    np.testing.assert_allclose(result["motor_speed_rps"], [100, 200, 300, 400])
    np.testing.assert_allclose(result["motor_speed_rpm"], [6000, 12000, 18000, 24000])
    np.testing.assert_allclose(result["motor_speed_rad_s"], np.array([100, 200, 300, 400]) * 2 * np.pi)
    np.testing.assert_allclose(result["motor_state_thrust_n"], [0.2, 1.6, 7.2, 1.6])
    np.testing.assert_allclose(result["applied_thrust_n"], [0.2, 1.6, 7.2, 1.6])
    np.testing.assert_allclose(result["motor_thrust_coefficient_n_per_rps2"], [2e-5, 4e-5, 8e-5, 1e-5])
    np.testing.assert_allclose(result["commanded_motor_speed_rps"], [100, 200, 300, 400])
    np.testing.assert_allclose(result["motor_wrench_b"], backend.allocation_matrix_b @ [0.2, 1.6, 7.2, 1.6])
    assert result["motor_speed_source"] == "direct_rps_state_integrated_with_native_motor_rate_and_rk4_or_euler"
    assert result["motor_speed_is_encoder_measurement"] is False
    assert actuator.mixing_calls == 0


def test_motor_speed_maps_multiple_actuator_groups_and_per_group_limits():
    backend = make_backend()
    backend.robot.actuators = {
        "first": make_actuator(np.array([2, 0]), [0.002, 0.001], [20, 25], bounds=(0.1, 1.0)),
        "second": make_actuator(slice(1, 4, 2), [0.003, 0.004], [10, 30], bounds=(0.2, 5.0)),
    }
    backend._initialize_speed_actuators()
    backend._extract_geometry()
    result = backend.telemetry()
    np.testing.assert_allclose(result["motor_speed_rps"], [10, 20, 30, 25])
    np.testing.assert_allclose(result["motor_thrust_coefficient_n_per_rps2"], [0.003, 0.002, 0.004, 0.001])
    params = backend.motor_parameters()
    np.testing.assert_allclose(params["thrust_min_n"], [0.2, 0.1, 0.2, 0.1])
    np.testing.assert_allclose(params["thrust_max_n"], [5, 1, 5, 1])
    command = [0, 100, 1, 100]
    backend.apply_motor_speeds(command, Wrench.zero())
    result = backend.telemetry()
    np.testing.assert_allclose(result["requested_motor_speed_rps"], command)
    np.testing.assert_allclose(
        result["commanded_motor_speed_rps"], [0, np.sqrt(1 / 0.002), np.sqrt(0.2 / 0.004), np.sqrt(1 / 0.001)]
    )
    for actuator in backend.robot.actuators.values():
        assert actuator.mixing_calls == 1
        assert actuator.rate_calls == 4
    assert result["actuators"]["first"]["thruster_names"] == ["back_right_prop", "front_right_prop"]
    assert result["actuators"]["second"]["thruster_indices"] == [1, 3]


def test_motor_telemetry_reset_direct_speeds_clears_submission_and_supports_zero():
    backend = make_backend()
    backend.apply_motor_speeds(np.full(4, 300.0), Wrench([1, 2, 3], [0.1, 0.2, 0.3]))
    assert backend.telemetry()["has_submitted_wrench"] is True
    actuator = backend.robot.actuators["thrusters"]
    actuator.thrust_const[:] = [[1e-5, 2e-5, 3e-5, 4e-5]]
    backend.robot.data.default_thruster_rps[:] = [[400.0, 0.0, 200.0, 300.0]]
    backend.reset(initial_position_w=[4, 5, 0.251])
    result = backend.telemetry()
    np.testing.assert_allclose(result["motor_speed_rps"], [0, 200, 300, 400])
    assert result["has_submitted_wrench"] is False
    np.testing.assert_allclose(result["applied_force_b_n"], 0)
    np.testing.assert_allclose(result["applied_torque_b_nm"], 0)
    np.testing.assert_allclose(backend.robot.written_pose[0, :3], [4, 5, 0.251])
    assert actuator.mixing_calls == 1
    assert result["applied_thrust_n"][0] == 0


def test_applied_force_and_torque_report_diagnostic_sum_of_prop_and_external():
    backend = make_backend()
    external = Wrench([1, 2, 3], [0.1, 0.2, 0.3])
    backend.apply_motor_speeds([200, 300, 400, 500], external)
    result = backend.telemetry()
    np.testing.assert_allclose(np.array(result["motor_wrench_b"]) + external.vector, result["total_wrench_b"])
    np.testing.assert_allclose(result["applied_force_b_n"], result["total_wrench_b"][:3])
    np.testing.assert_allclose(result["applied_torque_b_nm"], result["total_wrench_b"][3:])
    assert "not_applied_to_root" in result["motor_wrench_source"]
    assert "excludes_gravity_contact_and_native_damping" in result["applied_wrench_source"]


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_motor_speed_rejects_nonpositive_or_nonfinite_sampled_kf(value):
    backend = make_backend()
    backend.robot.actuators["thrusters"].thrust_const[0, 2] = value
    with pytest.raises(ValueError, match=r"thrust_const|coefficient"):
        backend.telemetry()


@pytest.mark.parametrize("selection", [[0, 0, 2, 3], [0, 1, 2, 4], [0, 1, 2, 0.5]])
def test_motor_speed_rejects_bad_actuator_mapping(selection):
    backend = make_backend()
    backend.robot.actuators["thrusters"].thruster_indices = selection
    with pytest.raises(ValueError, match="thruster indices"):
        backend.telemetry()


def test_motor_speed_rejects_incomplete_or_overlapping_groups():
    backend = make_backend()
    actuator = backend.robot.actuators["thrusters"]
    incomplete = types.SimpleNamespace(
        thruster_indices=[0, 1],
        cfg=actuator.cfg,
        tau_inc_s=actuator.tau_inc_s[:, :2],
        tau_dec_s=actuator.tau_dec_s[:, :2],
        thrust_const=actuator.thrust_const[:, :2],
        curr_thrust=actuator.curr_thrust[:, :2],
    )
    backend.robot.actuators = {"incomplete": incomplete}
    with pytest.raises(ValueError, match="cover every"):
        backend.telemetry()
    backend.robot.actuators["overlapping"] = incomplete
    with pytest.raises(ValueError, match="overlaps"):
        backend.telemetry()


def test_native_acceleration_weights_resolved_masses_in_public_body_order():
    backend = make_backend()
    # Public order is FR, base, BL, FL, BR (backend order differs).
    backend.robot.data.body_com_lin_acc_w = np.array(
        [[[1.0, 0.0, -9.81], [2.0, 0.0, -9.81], [3.0, 0.0, -9.81], [4.0, 0.0, -9.81], [5.0, 0.0, -9.81]]]
    )
    backend.apply_motor_speeds(np.full(4, 300.0), Wrench.zero())
    backend.robot.data._sim_timestamp = 0.005
    result = backend.telemetry()
    np.testing.assert_allclose(result["linear_acceleration_com_w_m_s2"], [5.3 / 2.4, 0, -9.81])
    assert (
        result["linear_acceleration_source"]
        == "physx_mass_weighted_body_com_lin_acc_w_world_coordinate_acceleration_including_gravity"
    )


def test_native_acceleration_unavailable_before_step_after_reset_or_without_interface():
    backend = make_backend()
    # The presence of stale/nonzero data must not make it valid before stepping.
    backend.robot.data.body_com_lin_acc_w = np.ones((1, 5, 3)) * 999
    assert backend.telemetry()["linear_acceleration_com_w_m_s2"] is None
    backend.apply_motor_speeds(np.full(4, 300.0), Wrench.zero())
    assert backend.telemetry()["linear_acceleration_com_w_m_s2"] is None
    backend.robot.data._sim_timestamp = 0.005
    assert backend.telemetry()["linear_acceleration_com_w_m_s2"] is not None
    backend.reset()
    assert backend.telemetry()["linear_acceleration_com_w_m_s2"] is None
    backend.apply_motor_speeds(np.full(4, 300.0), Wrench.zero())
    backend.robot.data._sim_timestamp = 0.010
    del backend.robot.data.body_com_lin_acc_w
    result = backend.telemetry()
    assert result["linear_acceleration_com_w_m_s2"] is None
    assert result["linear_acceleration_source"] == "unavailable_native_body_com_acceleration_interface"


def test_native_acceleration_property_is_read_once_and_invalid_values_are_unavailable():
    backend = make_backend()
    original_data = backend.robot.data

    class LazyData:
        def __init__(self):
            self.__dict__.update(vars(original_data))
            self.reads = 0

        @property
        def body_com_lin_acc_w(self):
            self.reads += 1
            return np.full((1, 5, 3), np.nan)

    backend.apply_motor_speeds(np.full(4, 300.0), Wrench.zero())
    backend.robot.data = LazyData()
    backend.robot.data._sim_timestamp = 0.005
    result = backend.telemetry()
    assert result["linear_acceleration_com_w_m_s2"] is None
    assert backend.robot.data.reads == 1
    assert result["linear_acceleration_source"] == "unavailable_invalid_native_body_acceleration_or_mass"


@pytest.mark.parametrize("scheme", ["rk4", "euler"])
@pytest.mark.parametrize("discrete", [False, True])
def test_speed_dynamics_reuse_native_rise_fall_and_integration(scheme, discrete):
    motor = make_actuator([0], [1e-5], scheme=scheme, discrete=discrete)
    adapter = NativeRpsActuator(motor, np.zeros((1, 1)))
    adapter.step([[300.0]])
    step_fraction = motor.cfg.dt / (0.06 + motor.cfg.dt if discrete else 0.06)
    gain = (
        step_fraction
        if scheme == "euler"
        else step_fraction - step_fraction**2 / 2 + step_fraction**3 / 6 - step_fraction**4 / 24
    )
    np.testing.assert_allclose(adapter.rps, [[300 * gain]])
    np.testing.assert_allclose(motor.last_tau, 0.06)
    rising = adapter.rps.copy()
    assert 0 < float(motor.curr_thrust[0, 0]) < 0.1  # no invented running-min force during spin-up
    adapter.step([[0.0]])
    np.testing.assert_allclose(motor.last_tau, 0.005)
    assert np.all(adapter.rps >= 0) and np.all(adapter.rps < rising)
    np.testing.assert_allclose(motor.curr_thrust, motor.thrust_const * adapter.rps**2)
    assert motor.mixing_calls == 2
    assert motor.rate_calls == (8 if scheme == "rk4" else 2)
    assert adapter.target_rps[0, 0] == 0


def test_speed_limits_rate_limit_and_zero_shutdown_are_distinct():
    motor = make_actuator([0, 1, 2], [1e-5, 2e-5, 4e-5], rate=10.0)
    adapter = NativeRpsActuator(motor, np.zeros((1, 3)))
    adapter.step([[0.0, 1.0, 100000.0]])
    np.testing.assert_allclose(adapter.target_rps, [[0, np.sqrt(0.1 / 2e-5), np.sqrt(10 / 4e-5)]])
    np.testing.assert_allclose(adapter.rps, [[0, 0.05, 0.05]])
    assert motor.curr_thrust[0, 0] == 0
    with pytest.raises(ValueError, match="nonnegative"):
        adapter.step([[-1, 0, 0]])
    with pytest.raises(ValueError):
        adapter.step([[np.nan, 0, 0]])


def test_thrust_to_rps_is_pure_per_motor_and_preserves_off():
    kf = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    result = thrust_to_motor_speeds([0, 0.01, 2, 100], kf, np.full(4, 0.1), np.full(4, 10.0))
    np.testing.assert_allclose(result, np.sqrt([0, 0.1 / 2e-5, 2 / 3e-5, 10 / 4e-5]))
    np.testing.assert_allclose(kf, [1e-5, 2e-5, 3e-5, 4e-5])
    with pytest.raises(ValueError, match="nonnegative"):
        thrust_to_motor_speeds([-1, 0, 0, 0], kf, np.zeros(4), np.ones(4))


def test_ground_position_recomputes_stale_extent_and_only_initializes_on_reset():
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/lmf2")
    base = UsdGeom.Xform.Define(stage, "/lmf2/base_link")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    box = UsdGeom.Cube.Define(stage, "/lmf2/base_link/box")
    box.CreateSizeAttr(1.0)
    box.CreateExtentAttr([Gf.Vec3f(-0.25), Gf.Vec3f(0.25)])  # stale, matches actual ARL pitfall
    box.AddScaleOp().Set(Gf.Vec3f(0.5))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    # Large visual box must not count toward ground support.
    UsdGeom.Cube.Define(stage, "/lmf2/base_link/visual").CreateSizeAttr(100.0)
    backend = make_backend()
    result = backend.ground_start_position(0.0, 0.001, stage=stage)
    np.testing.assert_allclose(result, [4, 5, 0.251])
    assert not hasattr(backend.robot, "written_pose")
    backend.reset(initial_position_w=result)
    np.testing.assert_allclose(backend.robot.written_pose[0, :3], result)
    # Scale is physical, so it must not be removed when forming body coordinates.
    base.AddScaleOp().Set(Gf.Vec3f(2.0))
    result = backend.ground_start_position(0.1, 0.001, stage=stage)
    np.testing.assert_allclose(result, [4, 5, 0.601])
    angle = np.pi / 6
    backend.config["vehicle"]["initial_state"]["quaternion_wxyz"] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    result = backend.ground_start_position(0.0, 0.0, stage=stage)
    assert result[2] == pytest.approx(0.5 * (np.sin(angle) + np.cos(angle)))


def test_individual_link_forces_recover_geometry_wrench_with_nonzero_rotor_com():
    backend = make_backend()
    # Public BL prop has nonzero COM and its local frame is tilted. The force
    # still acts at its geometric origin, not at the shifted prop COM.
    backend.robot.data.body_com_pos_b[0, 2] = [0.025, -0.012, 0.008]
    angle = 0.2
    backend.robot.data.body_link_quat_w[0, 2] = [np.sin(angle / 2), 0, 0, np.cos(angle / 2)]
    backend._extract_geometry()
    thrusts = np.array([1.0, 2.0, 3.0, 4.0])
    external = Wrench([1, 2, 3], [0.1, 0.2, 0.3])
    force, torque, points = backend._propeller_wrench_arrays(thrusts, external)
    recovered_force, recovered_torque = np.zeros(3), np.zeros(3)
    for public_id, backend_id in enumerate(backend.robot.data.body_ordering.user_to_backend):
        rotation = backend._link_rot_b[public_id]
        f = rotation @ force[0, backend_id]
        t = rotation @ torque[0, backend_id]
        point = backend._link_pos_b[public_id] + rotation @ points[0, backend_id]
        recovered_force += f
        recovered_torque += t + np.cross(point - backend._mass.com_b, f)
    np.testing.assert_allclose(
        np.r_[recovered_force, recovered_torque], backend.allocation_matrix_b @ thrusts + external.vector, atol=2e-7
    )
    np.testing.assert_allclose(points[0, 2], 0)  # Actual prop origin, despite shifted COM.


def test_actual_arl_ground_collision_instance_proxy_extent():
    Usd = pytest.importorskip("pxr.Usd")
    asset = Path(__file__).resolve().parents[1] / "assets/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd"
    if not asset.is_file():
        pytest.skip("ARL USD asset unavailable")
    stage = Usd.Stage.Open(str(asset))
    corners, paths = _collision_corners_b(stage, "/lmf2")
    assert paths == ["/lmf2/base_link/collisions/mesh_0/box"]
    assert stage.GetPrimAtPath(paths[0]).IsInstanceProxy()
    np.testing.assert_allclose(corners.min(0), [-0.25] * 3)
    np.testing.assert_allclose(corners.max(0), [0.25] * 3)


def test_ground_launch_audits_physics_plane_height_not_visual_mesh():
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import Gf, UsdGeom, UsdPhysics

    from isaac_drone.sim.isaaclab.backend import _ground_plane_height_m

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/Ground")
    visual = UsdGeom.Cube.Define(stage, "/Ground/visual")
    visual.AddTranslateOp().Set(Gf.Vec3d(0, 0, 999))
    plane = UsdGeom.Plane.Define(stage, "/Ground/physicsPlane")
    plane.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    height = plane.AddTranslateOp()
    height.Set(Gf.Vec3d(0, 0, 0.125))
    actual, path = _ground_plane_height_m(stage, "/Ground")
    assert actual == pytest.approx(0.125)
    assert path == "/Ground/physicsPlane"
    plane.AddRotateXOp().Set(10.0)
    with pytest.raises(ValueError, match="horizontal physical"):
        _ground_plane_height_m(stage, "/Ground")


@pytest.mark.skipif(UPSTREAM_SOURCE is None, reason="Isaac Lab checkout not found (set ISAACLAB_PATH)")
def test_transcription_matches_upstream_source():
    upstream = native_thruster.load_upstream_methods(UPSTREAM_SOURCE)
    rng = np.random.default_rng(0)
    for name in ("rk4", "euler"):
        fake = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005), max_rate=np.full((1, 4), 50.0))
        error, tau = rng.normal(0, 200, (1, 4)), rng.uniform(0.004, 0.09, (1, 4))
        results = []
        for methods in (upstream, native_thruster.TRANSCRIBED_METHODS):
            bound = {key: types.MethodType(func, fake) for key, func in methods.items()}
            fake.motor_model_rate = bound["motor_model_rate"]
            mixing = bound["discrete_mixing_factor"](tau)
            results.append(
                (
                    mixing,
                    bound["continuous_mixing_factor"](tau),
                    bound["rk4_integration"](error, mixing),
                    bound["motor_model_rate"](error, mixing),
                )
            )
        for expected, actual in zip(*results):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0, err_msg=name)
