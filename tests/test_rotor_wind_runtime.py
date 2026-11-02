"""Wind on the rotors through the closed loop on the synthetic plant: telemetry, flight effect and figures."""

from pathlib import Path

import numpy as np
import pytest

from isaac_drone.analysis.catalog import settings_line
from isaac_drone.config import load_config
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.sim.synthetic import SyntheticBackend, run_simulation

ROOT = Path(__file__).resolve().parents[1]
BASE = ["effects.ou_wrench.enabled=false", "completion=null"]
ROTOR_WIND = [
    "effects.wind.enabled=true",
    "effects.rotor_wind.enabled=true",
    "effects.rotor_wind.axial.enabled=true",
    "effects.rotor_wind.axial.thrust_gain_per_m_s=0.05",
    "effects.rotor_wind.inplane_drag.enabled=true",
    "effects.rotor_wind.inplane_drag.drag_n_per_m_s=0.5",
]
UPDRAFT_AT_HALF_SECOND = (
    "effects.wind.gusts=[{start_time_s: 0.5, duration_s: null, delta_velocity_w_m_s: [0, 0, 2], shape: step}]"
)


def fly(*overrides, steps=200):
    config = load_config(ROOT / "configs/arl_robot_1.yaml", overrides=[*BASE, *overrides])
    plant = SyntheticBackend(config)
    loop = MotionControlLoop(config, plant)
    loop.reset()
    records = []
    for _ in range(steps):
        loop.prepare_step()
        plant.step()
        records.append(loop.finish_step())
    return plant, records


def test_still_air_on_the_rotors_leaves_the_flight_unchanged():
    _, reference = fly()
    _, still = fly(*ROTOR_WIND, "effects.wind.velocity_w_m_s=[0, 0, 0]")
    for plain, windless in zip(reference, still):
        np.testing.assert_array_equal(plain["basic"]["position_w_m"], windless["basic"]["position_w_m"])
    np.testing.assert_array_equal(still[-1]["basic"]["rotor_speed_offset_rps"], 0)
    np.testing.assert_array_equal(still[-1]["basic"]["wind_w_m_s"], 0)
    assert reference[-1]["basic"]["rotor_speed_offset_rps"] is None
    assert reference[-1]["basic"]["wind_w_m_s"] is None


def test_a_sudden_updraft_speeds_every_rotor_by_the_axial_gain_and_lifts_the_vehicle():
    _, reference = fly(*ROTOR_WIND, "effects.wind.velocity_w_m_s=[0, 0, 0]")
    plant, records = fly(*ROTOR_WIND, "effects.wind.velocity_w_m_s=[0, 0, 0]", UPDRAFT_AT_HALF_SECOND)
    before, after = records[99]["basic"], records[100]["basic"]  # step 100 starts at t = 0.5 s
    np.testing.assert_array_equal(before["rotor_speed_offset_rps"], 0)
    np.testing.assert_allclose(after["wind_w_m_s"], [0, 0, 2])
    np.testing.assert_allclose(after["rotor_axial_wind_m_s"], 2, atol=1e-3)  # nearly level at hover
    hover = np.sqrt(plant.mass.mass_kg * 9.81 / 4 / plant.native.thrust_const[0])
    offsets = np.asarray(after["rotor_speed_offset_rps"])
    np.testing.assert_allclose(offsets, 0.5 * 0.05 * hover * np.asarray(after["rotor_axial_wind_m_s"]))
    np.testing.assert_allclose(after["rotor_speed_effective_rps"], np.asarray(after["motor_speed_rps"]) + offsets)
    np.testing.assert_allclose(
        after["motor_thrust_applied_n"],
        plant.native.thrust_const[0] * np.asarray(after["rotor_speed_effective_rps"]) ** 2,
    )
    # A vertical wind has an in-plane part only through the small hover tilt.
    assert np.linalg.norm(after["force_wind_drag_b_n"]) < 0.05
    # About 5 % more thrust per rotor: the vehicle rises until the position loop catches it.
    assert records[-1]["basic"]["position_w_m"][2] > reference[-1]["basic"]["position_w_m"][2] + 1e-3


def test_example_config_validates_and_describes_itself():
    line = settings_line(load_config(ROOT / "configs/helix_rotor_wind.yaml"))
    assert "wind 0 m/s + OU along (1,0,0) 1 m/s τ 1 s from 2 s" in line
    assert "rotor wind: axial 0.05/(m/s) → rotor speed, in-plane drag 0.5 N/(m/s)" in line
    assert "OU force" not in line


def test_disturbance_figures_follow_the_configured_models(tmp_path):
    pytest.importorskip("matplotlib")
    windy = load_config(
        ROOT / "configs/arl_robot_1.yaml",
        overrides=[*BASE, *ROTOR_WIND, "effects.wind.velocity_w_m_s=[1, 0, 0.5]", "simulation.duration_s=0.05"],
    )
    run = run_simulation(windy, figures=True, log=lambda *_: None, log_root=tmp_path / "wind").run_dir
    names = {path.name for path in (run / "plots").glob("*.png")}
    assert {"wind.png", "rotor_wind.png", "wind_drag.png"} <= names
    assert not names & {"ou_force.png", "ou_torque.png"}
    default = load_config(
        ROOT / "configs/arl_robot_1.yaml", overrides=["completion=null", "simulation.duration_s=0.05"]
    )
    run = run_simulation(default, figures=True, log=lambda *_: None, log_root=tmp_path / "ou").run_dir
    names = {path.name for path in (run / "plots").glob("*.png")}
    assert "ou_force.png" in names  # the default OU force; its torque is zero, so no torque figure
    assert not names & {"ou_torque.png", "wind.png", "rotor_wind.png", "wind_drag.png"}
