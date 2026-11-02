"""Wind field modes, instant start, step gusts and wind on the rotors, with synthetic fixtures only."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_drone.core.types import VehicleState
from isaac_drone.effects import Gust, OUWrench, RotorGeometry, RotorWind, UniformGustWind, build_effects

DISABLED = {
    "wind": {"enabled": False},
    "aerodynamic": {"enabled": False},
    "disturbances": [],
    "ou_wrench": {"enabled": False},
}
ROTOR_WIND = {
    "enabled": True,
    "axial": {"enabled": True, "thrust_gain_per_m_s": 0.05},
    "inplane_drag": {"enabled": True, "drag_n_per_m_s": 0.5},
}
HOVER_RPS = np.array([400.0, 420.0, 440.0, 460.0])
TILT_30_ABOUT_Y = (np.cos(np.pi / 12), 0.0, np.sin(np.pi / 12), 0.0)  # thrust axis tilted 30 deg toward +x


def state(*, time=0.0, quat=(1, 0, 0, 0)):
    return VehicleState(time, np.zeros(3), np.array(quat), np.zeros(3), np.zeros(3))


def samples(flow, steps, dt=0.005):
    values = []
    for index in range(steps):
        flow.advance(index * dt, dt)
        values.append(flow.velocity_w(np.zeros(3), index * dt))
    return np.array(values)


def wind_config(**overrides):
    config = {
        "enabled": True,
        "model": "uniform_gusts",
        "velocity_w_m_s": [0.0, 0.0, 0.0],
        "spatial_gradient_per_s": None,
        "reference_position_w_m": None,
        "start_time_s": 0.0,
        "end_time_s": None,
        "gusts": [],
        "turbulence": {
            "enabled": False,
            "mode": "axes",
            "direction_w": None,
            "time_constant_s": None,
            "stationary_std_w_m_s": None,
        },
    }
    config.update(overrides)
    return config


def geometry():
    positions = [(-0.1, 0.1, 0.0), (-0.1, -0.1, 0.0), (0.1, 0.1, 0.0), (0.1, -0.1, 0.0)]
    return RotorGeometry(positions, np.zeros(3), np.tile([0.0, 0.0, 1.0], (4, 1)), HOVER_RPS)


class TestWindField:
    def test_direction_turbulence_moves_only_along_that_direction(self):
        flow = UniformGustWind(
            (0, 0, 0), turbulence_time_constant_s=0.05, turbulence_std_w_m_s=1.5, turbulence_direction_w=(0, 0, 2)
        )
        flow.reset(11)
        values = samples(flow, 100_000)  # ~10k independent samples; tolerance ~4 sigma
        np.testing.assert_array_equal(values[:, :2], 0)
        assert values[:, 2].std() == pytest.approx(1.5, abs=0.06)
        diagonal = UniformGustWind(
            (0, 0, 0), turbulence_time_constant_s=1.0, turbulence_std_w_m_s=1.0, turbulence_direction_w=(1, 1, 0)
        )
        diagonal.reset(2)
        values = samples(diagonal, 50)
        np.testing.assert_allclose(values[:, 0], values[:, 1])
        np.testing.assert_array_equal(values[:, 2], 0)

    def test_axes_mode_with_zero_deviations_randomizes_a_single_axis(self):
        flow = UniformGustWind((1, 0, 0), turbulence_time_constant_s=(1, 1, 1), turbulence_std_w_m_s=(0, 0.8, 0))
        flow.reset(5)
        values = samples(flow, 200)
        np.testing.assert_array_equal(values[:, 0], 1)
        np.testing.assert_array_equal(values[:, 2], 0)
        assert np.ptp(values[:, 1]) > 0

    def test_turbulence_starts_at_full_strength(self):
        # Across seeds the very first sample already has the stationary spread: no build-up from zero.
        flow = UniformGustWind((0, 0, 0), turbulence_time_constant_s=(10, 10, 10), turbulence_std_w_m_s=(2, 2, 2))
        first = []
        for seed in range(4000):
            flow.reset(seed)
            flow.advance(0.0, 0.005)
            first.append(flow.velocity_w(np.zeros(3), 0.0))
        np.testing.assert_allclose(np.std(first, axis=0), 2.0, atol=0.08)

    def test_window_gives_still_air_then_the_full_wind_at_once(self):
        flow = UniformGustWind(
            (3, 0, 0),
            turbulence_time_constant_s=(1, 1, 1),
            turbulence_std_w_m_s=(1, 1, 1),
            start_time_s=1.0,
            end_time_s=2.0,
        )
        flow.reset(9)
        dt, values = 0.005, {}
        for index in range(500):
            flow.advance(index * dt, dt)
            values[index] = flow.velocity_w(np.zeros(3), index * dt)
        np.testing.assert_array_equal(values[199], 0)  # t = 0.995 s
        assert np.linalg.norm(values[200]) > 0 and not np.allclose(values[200], [3, 0, 0])  # mean and OU at 1.0 s
        np.testing.assert_array_equal(values[400], 0)  # t = 2.0 s: the window is half-open
        assert flow.components_w(np.zeros(3), 1.5)[0].tolist() == [3, 0, 0]

    def test_step_gust_appears_at_once_and_may_last_to_the_end(self):
        lasting = Gust(1.0, None, np.array([0, 0, -2]), "step")
        np.testing.assert_array_equal(lasting.velocity_w(0.999), 0)
        np.testing.assert_array_equal(lasting.velocity_w(1.0), [0, 0, -2])
        np.testing.assert_array_equal(lasting.velocity_w(500.0), [0, 0, -2])
        pulse = Gust(1.0, 2.0, np.array([0, 1, 0]), "step")
        np.testing.assert_array_equal(pulse.velocity_w(2.999), [0, 1, 0])
        np.testing.assert_array_equal(pulse.velocity_w(3.0), 0)
        np.testing.assert_array_equal(Gust(1.0, 2.0, np.array([0, 1, 0])).velocity_w(1.0), 0)  # 1-cosine rises from 0
        with pytest.raises(ValueError, match="only a step gust"):
            Gust(1.0, None, np.array([0, 1, 0]))
        with pytest.raises(ValueError, match="shape"):
            Gust(1.0, 1.0, np.array([0, 1, 0]), "square")

    def test_config_builds_each_turbulence_mode_and_rejects_mixed_settings(self):
        direction = {
            "enabled": True,
            "mode": "direction",
            "direction_w": [0, 0, 1],
            "time_constant_s": 1.0,
            "stationary_std_w_m_s": 0.5,
        }
        pipeline = build_effects({**DISABLED, "wind": wind_config(turbulence=direction, start_time_s=2.0)})
        np.testing.assert_allclose(pipeline.wind.direction_w, [0, 0, 1])
        assert pipeline.wind.start_s == 2.0
        axes = {
            "enabled": True,
            "mode": "axes",
            "direction_w": None,
            "time_constant_s": [1, 1, 1],
            "stationary_std_w_m_s": [0.5, 0, 0],
        }
        assert build_effects({**DISABLED, "wind": wind_config(turbulence=axes)}).wind.direction_w is None
        for bad in (
            {**direction, "stationary_std_w_m_s": [0.5, 0.5, 0.5]},
            {**direction, "direction_w": [0, 0, 0]},
            {**direction, "direction_w": None},
            {**axes, "direction_w": [1, 0, 0]},
            {**axes, "mode": "diagonal"},
        ):
            with pytest.raises(ValueError):
                build_effects({**DISABLED, "wind": wind_config(turbulence=bad)})
        gust = {"start_time_s": 1.0, "delta_velocity_w_m_s": [0, 0, -2], "shape": "step"}
        with pytest.raises(ValueError, match="explicit duration_s"):
            build_effects({**DISABLED, "wind": wind_config(gusts=[gust])})
        lasting = build_effects({**DISABLED, "wind": wind_config(gusts=[{**gust, "duration_s": None}])})
        assert lasting.wind.gusts[0].duration_s is None
        with pytest.raises(ValueError, match="shape"):
            build_effects({**DISABLED, "wind": wind_config(gusts=[{**gust, "duration_s": 1.0, "shape": "square"}])})
        with pytest.raises(ValueError, match="end_time_s"):
            build_effects({**DISABLED, "wind": wind_config(start_time_s=2.0, end_time_s=1.0)})


class TestRotorWind:
    def test_axial_offsets_scale_with_hover_speed_and_the_wind_along_the_rotor_axis(self):
        model = RotorWind(axial_thrust_gain_per_m_s=0.05)
        model.bind_rotor_geometry(geometry())
        offsets = model.rotor_speed_offsets(state(), UniformGustWind((0, 0, 2)))
        np.testing.assert_allclose(offsets, 0.5 * 0.05 * HOVER_RPS * 2)
        # Tilted 30 deg toward +x: half of a 2 m/s horizontal wind passes through the disks.
        offsets = model.rotor_speed_offsets(state(quat=TILT_30_ABOUT_Y), UniformGustWind((2, 0, 0)))
        np.testing.assert_allclose(offsets, 0.5 * 0.05 * HOVER_RPS * 1.0, rtol=1e-12)
        np.testing.assert_allclose(model.diagnostics()["rotor_axial_wind_m_s"], 1.0)

    def test_drag_lies_in_the_disk_plane_along_the_wind(self):
        model = RotorWind(inplane_drag_n_per_m_s=0.5)
        result = model.evaluate(state(), 0.005, UniformGustWind((2, 0, 1)))
        np.testing.assert_allclose(result.force_b, [1, 0, 0])
        np.testing.assert_array_equal(result.torque_b, 0)
        tilted = model.evaluate(state(quat=TILT_30_ABOUT_Y), 0.005, UniformGustWind((2, 0, 0)))
        np.testing.assert_allclose(tilted.force_b, [0.5 * 2 * np.cos(np.pi / 6), 0, 0], atol=1e-12)
        assert model.rotor_speed_offsets(state(), UniformGustWind((0, 0, 1))) is None  # axial channel off

    def test_still_air_unbound_geometry_and_bad_coefficients(self):
        model = RotorWind(axial_thrust_gain_per_m_s=0.05, inplane_drag_n_per_m_s=0.5)
        with pytest.raises(RuntimeError, match="bind"):
            model.rotor_speed_offsets(state(), UniformGustWind((0, 0, 1)))
        model.bind_rotor_geometry(geometry())
        still = UniformGustWind((0, 0, 0))
        np.testing.assert_array_equal(model.evaluate(state(), 0.005, still).vector, 0)
        np.testing.assert_array_equal(model.rotor_speed_offsets(state(), still), 0)
        with pytest.raises(ValueError, match="at least one channel"):
            RotorWind()
        with pytest.raises(ValueError, match="nonnegative"):
            RotorWind(inplane_drag_n_per_m_s=-1)
        with pytest.raises(ValueError, match="explicit WindField"):
            model.evaluate(state(), 0.005)

    def test_pipeline_wires_rotor_wind_after_the_other_effects(self):
        ou = {
            "enabled": True,
            "force_mean_w_n": [0, 0, 0],
            "force_std_w_n": [1, 1, 1],
            "force_time_constant_s": 1.0,
            "torque_mean_b_nm": [0, 0, 0],
            "torque_std_b_nm": [0, 0, 0],
            "torque_time_constant_s": 1.0,
            "start_time_s": 0.0,
            "end_time_s": None,
        }
        updraft = wind_config(velocity_w_m_s=[0, 0, 1])
        with_rotor = build_effects({**DISABLED, "wind": updraft, "ou_wrench": ou, "rotor_wind": ROTOR_WIND})
        without = build_effects({**DISABLED, "wind": updraft, "ou_wrench": ou})
        assert isinstance(with_rotor.effects[0], OUWrench) and isinstance(with_rotor.effects[-1], RotorWind)
        assert with_rotor.needs_rotor_geometry and not without.needs_rotor_geometry
        with_rotor.bind_rotor_geometry(geometry())
        for pipeline in (with_rotor, without):
            pipeline.reset(4)
            pipeline.evaluate(state(), 0.005)
        # Appending rotor wind last leaves the OU force's random stream unchanged.
        np.testing.assert_array_equal(
            with_rotor.diagnostics(state())["force_ou_w_n"], without.diagnostics(state())["force_ou_w_n"]
        )
        offsets = with_rotor.rotor_speed_offsets(state())
        np.testing.assert_allclose(offsets, 0.5 * 0.05 * HOVER_RPS)
        assert without.rotor_speed_offsets(state()) is None
        diagnostics = with_rotor.diagnostics(state())
        np.testing.assert_allclose(diagnostics["wind_w_m_s"], [0, 0, 1])
        np.testing.assert_allclose(diagnostics["wind_mean_w_m_s"], [0, 0, 1])
        np.testing.assert_allclose(diagnostics["rotor_speed_offset_rps"], offsets)
        np.testing.assert_allclose(diagnostics["force_wind_drag_b_n"], 0)
        with_rotor.reset(4)
        with pytest.raises(ValueError, match="evaluate"):
            with_rotor.rotor_speed_offsets(state())

    def test_rotor_wind_config_needs_wind_and_an_enabled_channel(self):
        updraft = wind_config(velocity_w_m_s=[0, 0, 1])
        with pytest.raises(ValueError, match="requires an enabled wind"):
            build_effects({**DISABLED, "rotor_wind": ROTOR_WIND})
        off = {
            "enabled": True,
            "axial": {"enabled": False, "thrust_gain_per_m_s": None},
            "inplane_drag": {"enabled": False, "drag_n_per_m_s": None},
        }
        with pytest.raises(ValueError, match="at least one enabled channel"):
            build_effects({**DISABLED, "wind": updraft, "rotor_wind": off})
        missing = {**ROTOR_WIND, "axial": {"enabled": True, "thrust_gain_per_m_s": None}}
        with pytest.raises(ValueError, match="thrust_gain_per_m_s"):
            build_effects({**DISABLED, "wind": updraft, "rotor_wind": missing})
        with pytest.raises(ValueError, match="unsupported keys"):
            build_effects({**DISABLED, "wind": updraft, "rotor_wind": {**ROTOR_WIND, "gain": 1}})
        assert build_effects({**DISABLED, "rotor_wind": {**ROTOR_WIND, "enabled": False}}).effects == ()
