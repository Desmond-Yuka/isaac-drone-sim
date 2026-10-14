"""Physics/frame checks for the external wrench models, with synthetic fixtures only."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_drone.core.types import VehicleState
from isaac_drone.effects import (
    BodyDrag,
    ConstantWrench,
    EffectsPipeline,
    Gust,
    OUWrench,
    UniformGustWind,
    build_effects,
)

DISABLED = {
    "wind": {"enabled": False},
    "aerodynamic": {"enabled": False},
    "disturbances": [],
    "ou_wrench": {"enabled": False},
}


def state(*, time=0.0, position=(0, 0, 0), quat=(1, 0, 0, 0), velocity=(0, 0, 0), omega=(0, 0, 0)):
    return VehicleState(time, np.array(position), np.array(quat), np.array(velocity), np.array(omega))


def wind(velocity=(0, 0, 0)):
    return UniformGustWind(velocity)


def drag(*, linear=None, quadratic=(0, 0, 0), angular=None, point=(0, 0, 0)):
    return BodyDrag(
        linear_drag_b_kg_s=np.eye(3) if linear is None else linear,
        quadratic_drag_b_kg_m=quadratic,
        angular_linear_drag_b_nm_s=np.zeros((3, 3)) if angular is None else angular,
        center_of_pressure_b_m=point,
    )


def ou(**overrides):
    values = dict(
        force_mean_w_n=(0, 0, 0),
        force_std_w_n=(0, 0, 0),
        force_time_constant_s=1.0,
        torque_mean_b_nm=(0, 0, 0),
        torque_std_b_nm=(0, 0, 0),
        torque_time_constant_s=1.0,
    )
    values.update(overrides)
    return OUWrench(**values)


class TestAerodynamics:
    def test_relative_wind_zero_when_vehicle_moves_with_air(self):
        result = drag(quadratic=(1, 2, 3)).evaluate(state(velocity=(2, -3, 1)), 0.01, wind((2, -3, 1)))
        np.testing.assert_allclose(result.vector, np.zeros(6), atol=1e-14)

    def test_rotated_world_wind_and_anisotropic_drag(self):
        half = np.sqrt(0.5)
        result = drag(linear=np.diag([2, 3, 4])).evaluate(state(quat=(half, 0, 0, half)), 0.01, wind((2, 0, 0)))
        # +world X is -body Y after a +90 degree yaw; force follows the wind.
        np.testing.assert_allclose(result.force_b, [0, -6, 0], atol=1e-12)

    def test_rotation_at_offset_point_produces_drag_and_lever_moment(self):
        result = drag(point=(1, 0, 0)).evaluate(state(omega=(0, 0, 2)), 0.01, wind())
        np.testing.assert_allclose(result.force_b, [0, -2, 0])
        np.testing.assert_allclose(result.torque_b, [0, 0, -2])

    def test_full_matrix_and_lever_arm_preserve_dissipation(self):
        d = np.array([[3, 1.0, 0], [-0.5, 2, 0.2], [0, 0.2, 4]])
        model = drag(linear=d, quadratic=(0.1, 0.2, 0.3), angular=np.eye(3), point=(0.7, -0.2, 0.3))
        rng = np.random.default_rng(62)
        for _ in range(20):
            velocity, omega = rng.normal(size=(2, 3))
            result = model.evaluate(state(velocity=velocity, omega=omega), 0.01, wind())
            assert result.force_b @ velocity + result.torque_b @ omega <= 1e-12

    def test_non_dissipative_or_unidentified_coefficients_rejected(self):
        with pytest.raises(ValueError, match="positive-semidefinite"):
            drag(linear=np.diag([-1, 1, 1]))
        with pytest.raises(ValueError, match="nonnegative"):
            drag(quadratic=(-1, 0, 0))
        with pytest.raises(ValueError, match="explicit WindField"):
            drag().evaluate(state(), 0.01)

    def test_shear_is_sampled_at_actual_aerodynamic_point(self):
        flow = UniformGustWind(
            (0, 0, 0), spatial_gradient_per_s=[[0, 0, 0], [2, 0, 0], [0, 0, 0]], reference_position_w_m=(0, 0, 0)
        )
        result = drag(point=(1, 0, 0)).evaluate(state(), 0.01, flow)
        np.testing.assert_allclose(result.force_b, [0, 2, 0])
        np.testing.assert_allclose(result.torque_b, [0, 0, 2])

    def test_smooth_gust_and_reproducible_turbulence_reset(self):
        gust = Gust(1, 2, np.array([4, 0, 0]))
        for time in [0, 1, 3, 4]:
            np.testing.assert_allclose(gust.velocity_w(time), [0, 0, 0])
        np.testing.assert_allclose(gust.velocity_w(2), [4, 0, 0])
        flow = UniformGustWind((0, 0, 0), turbulence_time_constant_s=[1, 2, 3], turbulence_std_w_m_s=[2, 2, 2])

        def run():
            flow.reset(314)
            values = []
            for time in [0, 0.1, 0.2]:
                flow.advance(time, 0.1)
                values.append(flow.velocity_w(np.zeros(3), time))
            return values

        np.testing.assert_array_equal(run(), run())
        with pytest.raises(ValueError, match="once"):
            flow.advance(0.2, 0.1)


class TestDisturbances:
    def test_world_force_rotated_with_body_application_point(self):
        half = np.sqrt(0.5)
        effect = ConstantWrench(
            force_n=[0, 2, 0],
            torque_nm=[0, 0, 3],
            frame="world",
            application_point_b_m=[0, 1, 0],
            start_time_s=1,
            end_time_s=2,
        )
        result = effect.evaluate(state(time=1, quat=(half, 0, 0, half)), 0.01)
        np.testing.assert_allclose(result.force_b, [2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result.torque_b, [0, 0, 1], atol=1e-12)
        np.testing.assert_array_equal(effect.evaluate(state(time=2), 0.01).vector, np.zeros(6))

    def test_world_application_point_uses_world_com(self):
        effect = ConstantWrench(
            force_n=[0, 0, 4],
            torque_nm=[0, 0, 0],
            frame="world",
            application_point_w_m=[11, 0, 0],
            start_time_s=0,
            end_time_s=None,
        )
        result = effect.evaluate(state(position=[10, 0, 0]), 0.01)
        np.testing.assert_allclose(result.torque_b, [0, -4, 0])

    def test_pipeline_sum_reset_and_time_validation(self):
        effect = ConstantWrench(force_n=[1, 2, 3], torque_nm=[3, 2, 1], frame="body", start_time_s=0, end_time_s=None)
        pipeline = EffectsPipeline([effect, effect])
        np.testing.assert_allclose(pipeline.evaluate(state(), 0.01).vector, [2, 4, 6, 6, 4, 2])
        with pytest.raises(ValueError, match="once"):
            pipeline.evaluate(state(), 0.01)
        pipeline.reset(12)
        pipeline.evaluate(state(), 0.01)

    def test_disabled_effects_and_missing_parameters(self):
        values = {key: dict(value) if isinstance(value, dict) else list(value) for key, value in DISABLED.items()}
        pipeline = build_effects(values)
        assert pipeline.wind is None and pipeline.effects == ()
        np.testing.assert_array_equal(pipeline.evaluate(state(), 0.01).vector, np.zeros(6))
        values["aerodynamic"]["enabled"] = True
        with pytest.raises(ValueError, match="explicit wind"):
            build_effects(values)
        values["aerodynamic"]["enabled"] = False
        values["wind"] = {"enabled": True, "model": "uniform_gusts", "gusts": [], "turbulence": {"enabled": False}}
        with pytest.raises(ValueError, match="velocity_w_m_s"):
            build_effects(values)

    @pytest.mark.parametrize("bad_dt", [0, -1, float("nan"), float("inf"), True, None])
    def test_bad_step_rejected(self, bad_dt):
        with pytest.raises(ValueError):
            EffectsPipeline().evaluate(state(), bad_dt)


class TestOUWrench:
    def test_mean_only_is_world_force_and_body_torque(self):
        half = np.sqrt(0.5)
        effect = ou(force_mean_w_n=(2, 0, 0), torque_mean_b_nm=(0, 0, 3))
        result = effect.evaluate(state(quat=(half, 0, 0, half)), 0.005)
        np.testing.assert_allclose(result.force_b, [0, -2, 0], atol=1e-12)
        np.testing.assert_allclose(result.torque_b, [0, 0, 3])

    def test_stationary_statistics_and_correlation_time(self):
        # 200k steps of 5 ms with tau = 50 ms: about 10k independent samples, tolerances are ~4 sigma.
        dt, tau, std = 0.005, 0.05, np.array([1.0, 2.0, 0.0])
        effect = ou(force_std_w_n=std, force_time_constant_s=tau, torque_std_b_nm=(0.5, 0.5, 0.5))
        effect.reset(7)
        samples = np.array([effect.evaluate(state(time=k * dt), dt).vector for k in range(200_000)])
        force, torque = samples[:, :3], samples[:, 3:]
        np.testing.assert_allclose(force.mean(axis=0), 0, atol=0.08)
        np.testing.assert_allclose(force.std(axis=0), std, atol=0.06)
        np.testing.assert_allclose(torque.std(axis=0), 0.5, atol=0.03)
        lag = round(tau / dt)
        correlation = np.corrcoef(force[:-lag, 0], force[lag:, 0])[0, 1]
        assert correlation == pytest.approx(np.exp(-1), abs=0.04)

    def test_seeded_reset_reproduces_and_window_gates_output(self):
        effect = ou(force_std_w_n=(1, 1, 1), start_time_s=1.0, end_time_s=2.0)

        def run():
            effect.reset(3)
            return np.array([effect.evaluate(state(time=t), 0.5).vector for t in (0.0, 0.5, 1.0, 1.5, 2.0)])

        first, second = run(), run()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[[0, 1, 4]], 0)
        assert np.all(first[[2, 3], :3] != 0)

    def test_config_section_builds_and_validates(self):
        values = {
            **{key: dict(value) for key, value in DISABLED.items() if isinstance(value, dict)},
            "disturbances": [],
            "ou_wrench": {
                "enabled": True,
                "force_mean_w_n": [0, 0, -1],
                "force_std_w_n": [0.5, 0.5, 0],
                "force_time_constant_s": 0.5,
                "torque_mean_b_nm": [0, 0, 0],
                "torque_std_b_nm": [0, 0, 0],
                "torque_time_constant_s": 1.0,
                "start_time_s": 0.0,
                "end_time_s": None,
            },
        }
        pipeline = build_effects(values)
        assert len(pipeline.effects) == 1 and isinstance(pipeline.effects[0], OUWrench)
        for key, bad in [
            ("force_std_w_n", [-1, 0, 0]),
            ("torque_time_constant_s", 0),
            ("end_time_s", 0.0),
            ("force_mean_w_n", [True, 0, 0]),
        ]:
            broken = {**values, "ou_wrench": {**values["ou_wrench"], key: bad}}
            with pytest.raises(ValueError):
                build_effects(broken)
        with pytest.raises(ValueError, match="unsupported keys"):
            build_effects({**values, "ou_wrench": {**values["ou_wrench"], "force_stdev_w_n": [0, 0, 0]}})
        values["ou_wrench"]["enabled"] = False
        values["ou_wrench"]["force_std_w_n"] = None
        assert build_effects(values).effects == ()
        values["ou_wrench"]["enabled"] = True
        with pytest.raises(ValueError, match="force_std_w_n"):
            build_effects(values)


def test_configuration_typos_are_not_silently_ignored():
    with pytest.raises(ValueError, match="unsupported keys"):
        build_effects({**DISABLED, "wind": {"enabled": False, "velocty_w_m_s": [0, 0, 0]}})


@pytest.mark.parametrize(
    "bad_wind",
    [
        {"enabled": False, "turbulence": {"enabled": False, "stationary_stdev_w_m_s": None}},
        {"enabled": False, "turbulence": {"enabled": "false"}},
        {"enabled": False, "gusts": [{"start_time": 1}]},
        {"enabled": False, "gusts": "none"},
        {"enabled": False, "velocity_w_m_s": [True, 0, 0]},
        {"enabled": False, "velocity_w_m_s": ["0", 0, 0]},
    ],
)
def test_disabled_wind_still_checks_nested_schema_and_numeric_types(bad_wind):
    with pytest.raises(ValueError):
        build_effects({**DISABLED, "wind": bad_wind})


def test_disabled_disturbance_still_checks_optional_parameters():
    with pytest.raises(ValueError):
        build_effects({**DISABLED, "disturbances": [{"enabled": False, "force_n": [False, 0, 0]}]})
