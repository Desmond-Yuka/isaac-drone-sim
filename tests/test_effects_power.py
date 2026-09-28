"""Physics/frame/circuit checks use synthetic fixtures, never ARL calibration."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_drone.core.types import VehicleState
from isaac_drone.effects import BodyDrag, ConstantWrench, EffectsPipeline, Gust, UniformGustWind, build_effects
from isaac_drone.power import (
    BatteryLimitError,
    ModelValidityError,
    ThrustBounds,
    build_battery,
    build_power,
    validate_power_config,
)


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


def config():
    # Synthetic one-cell pack; these numbers are test values, not ARL parameters.
    return dict(
        enabled=True,
        model="equivalent_circuit",
        current_source="external_measurement",
        calibration_id="synthetic-unit-test-only",
        capacity_ah=1.0,
        initial_soc=0.8,
        ocv_soc_voltage_table=[[0.0, 3.0], [0.5, 3.5], [1.0, 4.0]],
        series_resistance_ohm=0.05,
        rc_branches=[],
        charge_coulombic_efficiency=1.0,
        discharge_coulombic_efficiency=1.0,
        cutoff_voltage_v=2.5,
        maximum_voltage_v=4.3,
        max_discharge_current_a=20.0,
        max_charge_current_a=10.0,
        calibration_temperature_c=25.0,
        valid_temperature_range_c=[20.0, 30.0],
        initial_temperature_c=None,
    )


def battery(**overrides):
    values = config()
    values.update(overrides)
    return build_battery(values)


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
        # Include non-diagonal dissipative and skew components, and nonzero CoP.
        d = np.array([[3, 1.0, 0], [-0.5, 2, 0.2], [0, 0.2, 4]])
        model = drag(linear=d, quadratic=(0.1, 0.2, 0.3), angular=np.eye(3), point=(0.7, -0.2, 0.3))
        rng = np.random.default_rng(62)
        for _ in range(20):
            velocity, omega = rng.normal(size=(2, 3))
            result = model.evaluate(state(velocity=velocity, omega=omega), 0.01, wind())
            mechanical_power = result.force_b @ velocity + result.torque_b @ omega
            assert mechanical_power <= 1e-12

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
        values = dict(wind={"enabled": False}, aerodynamic={"enabled": False}, disturbances=[])
        pipeline = build_effects(values)
        assert pipeline.wind is None
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


class TestBattery:
    def test_soc_terminal_voltage_and_port_energy(self):
        pack = battery()
        result = pack.step(2, 180, temperature_c=25)
        assert result.soc == pytest.approx(0.7)
        assert result.terminal_voltage_v == pytest.approx(3.6)
        # Terminal starts at 3.7 V and ends at 3.6 V under the 2 A load.
        assert result.energy_delivered_j == pytest.approx(2 * 180 * (3.7 + 3.6) / 2)
        assert result.ohmic_heat_j == pytest.approx(2**2 * 0.05 * 180)
        assert result.charge_throughput_c == pytest.approx(360)

    def test_ocv_energy_integrates_across_table_knots(self):
        pack = battery(initial_soc=0.9, ocv_soc_voltage_table=[[0, 3], [0.3, 3.1], [0.5, 3.8], [1, 4.0]])
        result = pack.step(1, 2520, temperature_c=25)
        # Independently integrate the three trapezoids from SOC .2 to .9.
        ocv_area = (3 + 0.2 / 3 + 3.1) / 2 * 0.1 + (3.1 + 3.8) / 2 * 0.2 + (3.8 + 3.96) / 2 * 0.4
        expected_energy = 3600 * ocv_area - 0.05 * 2520
        assert result.soc == pytest.approx(0.2)
        assert result.energy_delivered_j == pytest.approx(expected_energy)

    def test_rc_polarization_and_relaxation(self):
        pack = battery(rc_branches=[dict(resistance_ohm=0.1, capacitance_f=100, initial_polarization_v=0.0)])
        loaded = pack.step(2, 10, temperature_c=25)
        expected_polarization = 0.2 * (1 - np.exp(-1))
        assert loaded.polarization_voltage_v[0] == pytest.approx(expected_polarization)
        relaxed = pack.step(0, 10, temperature_c=25)
        assert relaxed.polarization_voltage_v[0] == pytest.approx(expected_polarization * np.exp(-1))
        assert relaxed.soc == loaded.soc
        assert relaxed.energy_delivered_j == loaded.energy_delivered_j
        assert relaxed.ohmic_heat_j > loaded.ohmic_heat_j

    def test_rc_energy_conservation_against_stored_and_dissipated_energy(self):
        pack = battery(rc_branches=[dict(resistance_ohm=0.1, capacitance_f=100, initial_polarization_v=0.07)])
        result = pack.step(2, 8, temperature_c=25)
        source_energy = 2 * 8 * ((3.8 + result.open_circuit_voltage_v) / 2)
        initial_cap_energy = 0.5 * 100 * 0.07**2
        final_cap_energy = 0.5 * 100 * result.polarization_voltage_v[0] ** 2
        assert source_energy + initial_cap_energy == pytest.approx(
            result.energy_delivered_j + result.ohmic_heat_j + final_cap_energy, rel=1e-12
        )

    def test_small_dt_rc_heat_remains_positive_and_accurate(self):
        pack = battery(
            series_resistance_ohm=0, rc_branches=[dict(resistance_ohm=1, capacitance_f=1000, initial_polarization_v=0)]
        )
        result = pack.step(1, 1e-5, temperature_c=25)
        leading_order_heat = (1e-5) ** 3 / (3 * 1000**2)
        assert result.ohmic_heat_j == pytest.approx(leading_order_heat, rel=1e-7, abs=0)
        assert result.ohmic_heat_j > 0

    def test_charge_sign_and_nonunit_coulombic_efficiencies(self):
        pack = battery(charge_coulombic_efficiency=0.9, discharge_coulombic_efficiency=0.8)
        charged = pack.step(-1, 360, temperature_c=25)
        assert charged.soc == pytest.approx(0.89)
        assert charged.energy_delivered_j < 0
        discharged = pack.step(1, 360, temperature_c=25)
        assert discharged.soc == pytest.approx(0.765)

    def test_limit_failures_do_not_mutate_state(self):
        pack = battery(initial_soc=0.01)
        old = pack.state
        for current, dt, temp in [(1, 360, 25), (21, 1, 25), (-11, 1, 25), (1, 1, 19), (1, 0, 25)]:
            with pytest.raises(ValueError):
                pack.step(current, dt, temperature_c=temp)
            assert pack.state is old
        assert pack.step(1, 36, temperature_c=25).soc == pytest.approx(0)

    def test_undervoltage_is_reported_without_clipping(self):
        pack = battery(series_resistance_ohm=0.1, cutoff_voltage_v=3.0)
        result = pack.step(10, 1, temperature_c=25)
        assert result.terminal_voltage_v < 3.0
        assert "undervoltage" in result.flags

    def test_reset_restores_initial_rc_and_unknown_observations(self):
        pack = battery(rc_branches=[dict(resistance_ohm=0.1, capacitance_f=10, initial_polarization_v=0.03)])
        initial = pack.state
        pack.step(1, 1, temperature_c=25)
        assert pack.reset() == initial
        assert pack.state.terminal_voltage_v is None
        assert pack.state.temperature_c is None
        assert pack.state.energy_delivered_j == 0

    def test_missing_temperature_is_unknown_not_persisted(self):
        pack = battery(initial_temperature_c=25)
        assert pack.state.temperature_c == 25
        result = pack.step(1, 1)
        assert result.temperature_c is None
        assert "temperature_unobserved" in result.flags

    @pytest.mark.parametrize(
        "override",
        [
            {"capacity_ah": 0},
            {"initial_soc": 1.1},
            {"series_resistance_ohm": -1},
            {"ocv_soc_voltage_table": [[0.1, 3], [1, 4]]},
            {"ocv_soc_voltage_table": [[0, 4], [1, 3]]},
            {"ocv_soc_voltage_table": [[0, 3], [0, 3.1], [1, 4]]},
            {"rc_branches": None},
            {"calibration_id": ""},
            {"charge_coulombic_efficiency": 1.1},
            {"valid_temperature_range_c": [30, 20]},
        ],
    )
    def test_invalid_parameter_sets_rejected(self, override):
        with pytest.raises(ValueError):
            battery(**override)


class SyntheticEnvelope:
    """Only a test stub; no real voltage/thrust calibration is supplied."""

    def reset(self):
        pass

    def thrust_bounds_n(self, battery_state, vehicle_state, telemetry):
        return ThrustBounds(np.zeros(4), np.full(4, telemetry["synthetic_max_thrust_n"]))


class SyntheticLoad:
    def reset(self):
        pass

    def current_a(self, vehicle_state, telemetry, battery_state):
        return telemetry["synthetic_current_a"]


class TestPowerCoupling:
    def test_disabled_is_absent_and_enabled_requires_calibrated_envelope(self):
        assert build_power({"enabled": False}) is None
        validate_power_config(config())
        with pytest.raises(ValueError, match="ActuatorEnvelope"):
            build_power(config())
        values = config()
        values["current_source"] = "load_model"
        with pytest.raises(ValueError, match="LoadModel"):
            build_power(values, actuator_envelope=SyntheticEnvelope())

    def test_operating_point_and_measurements_are_required(self):
        system = build_power(config(), actuator_envelope=SyntheticEnvelope())
        with pytest.raises(ModelValidityError):
            system.thrust_bounds_n(state(), {})
        system.initialize(1, temperature_c=25)
        bounds = system.thrust_bounds_n(state(), {"synthetic_max_thrust_n": 2})
        np.testing.assert_array_equal(bounds.upper_n, [2, 2, 2, 2])
        with pytest.raises(ValueError, match="measured_current_a"):
            system.update(state(), {}, 0.01, temperature_c=25)
        result = system.update(state(), {}, 0.01, measured_current_a=2, temperature_c=25)
        assert result.current_a == 2
        system.reset()
        with pytest.raises(ModelValidityError):
            system.thrust_bounds_n(state(), {})

    def test_voltage_limits_prevent_actuation_and_load_plugin_is_used(self):
        values = config()
        values.update(current_source="load_model", cutoff_voltage_v=3.5)
        system = build_power(values, load_model=SyntheticLoad(), actuator_envelope=SyntheticEnvelope())
        system.initialize(0, temperature_c=25)
        result = system.update(state(), {"synthetic_current_a": 10}, 0.01, temperature_c=25)
        assert "undervoltage" in result.flags
        with pytest.raises(BatteryLimitError):
            system.thrust_bounds_n(state(), {})

    @pytest.mark.parametrize("lower,upper", [([], []), ([0], [-1]), ([2], [1]), ([0, 0], [1]), ([0], [np.inf])])
    def test_bad_actuator_bounds_rejected(self, lower, upper):
        with pytest.raises(ValueError):
            ThrustBounds(lower, upper)


def test_configuration_typos_are_not_silently_ignored():
    with pytest.raises(ValueError, match="unsupported keys"):
        build_power({"enabled": False, "capacitty_ah": 1})
    with pytest.raises(ValueError, match="unsupported keys"):
        build_effects(
            {
                "wind": {"enabled": False, "velocty_w_m_s": [0, 0, 0]},
                "aerodynamic": {"enabled": False},
                "disturbances": [],
            }
        )


@pytest.mark.parametrize(
    "bad_power",
    [
        {"enabled": False, "rc_branches": [{"resistence_ohm": 1}]},
        {"enabled": False, "rc_branches": "unknown"},
        {"enabled": False, "capacity_ah": True},
        {"enabled": False, "capacity_ah": "1.0"},
        {"enabled": False, "ocv_soc_voltage_table": [[0, "3"], [1, "4"]]},
        {"enabled": False, "initial_temperature_c": float("nan")},
    ],
)
def test_disabled_battery_still_checks_nested_schema_and_numeric_types(bad_power):
    with pytest.raises(ValueError):
        validate_power_config(bad_power)


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
        build_effects({"wind": bad_wind, "aerodynamic": {"enabled": False}, "disturbances": []})


def test_disabled_disturbance_still_checks_optional_parameters():
    with pytest.raises(ValueError):
        build_effects(
            {
                "wind": {"enabled": False},
                "aerodynamic": {"enabled": False},
                "disturbances": [{"enabled": False, "force_n": [False, 0, 0]}],
            }
        )
