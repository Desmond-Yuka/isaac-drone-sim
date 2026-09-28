"""Synthetic CPU plant physics checks (the closed-loop regression lives in test_golden_regression)."""
from pathlib import Path

import numpy as np
import pytest

from isaac_drone.config import load_config
from isaac_drone.core.types import Wrench
from isaac_drone.sim.synthetic import SyntheticBackend

ROOT = Path(__file__).resolve().parents[1]


def make_plant():
    config = load_config(ROOT / "configs/helix.yaml")
    plant = SyntheticBackend(config)
    launch = config["vehicle"]["launch"]
    plant.reset(plant.ground_start_position(launch["ground_z_m"], launch["clearance_m"]))
    return plant


def test_synthetic_fixture_keeps_full_inertia_com_and_unequal_motor_lag():
    plant = make_plant()
    assert np.count_nonzero(plant.mass.com_b) == 3
    assert np.count_nonzero(plant.mass.inertia_com_b-np.diag(np.diag(plant.mass.inertia_com_b))) == 6
    assert len(set(plant.native.thrust_const[0])) == 4
    plant.apply_motor_speeds(np.full(4, 400.), Wrench.zero())
    assert np.all(plant.motor.rps[0] > 0)
    assert np.all(plant.motor.rps[0] < 400)
    assert len(set(plant.motor.rps[0])) == 4
    np.testing.assert_allclose(plant.native.curr_thrust, plant.native.thrust_const*plant.motor.rps**2)
    with pytest.raises(RuntimeError, match="already prepared"):
        plant.apply_motor_speeds(np.full(4, 400.), Wrench.zero())


def test_zero_motors_fall_to_contact_and_cannot_follow_target():
    plant = make_plant()
    original_xy = plant.position[:2].copy()
    for _ in range(50):
        plant.apply_motor_speeds(np.zeros(4), Wrench.zero())
        plant.step()
    np.testing.assert_allclose(plant.position[:2], original_xy)
    assert plant.position[2] == pytest.approx(plant.contact_floor_z)
    np.testing.assert_allclose(plant.velocity, 0)
    np.testing.assert_allclose(plant.motor.rps, 0)
    assert plant.contact_impulse_n_s == pytest.approx(plant.mass.mass_kg*9.81*plant.dt)


def test_rigid_body_freefall_and_torque_follow_newton_euler():
    plant = make_plant()
    plant.position[2] = 10.
    start = plant.position.copy()
    plant.apply_motor_speeds(np.zeros(4), Wrench([0, 0, 0], [.03, -.02, .01]))
    expected_alpha = np.linalg.solve(plant.mass.inertia_com_b, [.03, -.02, .01])
    plant.step()
    np.testing.assert_allclose(plant.position, start + .5*plant.gravity*plant.dt**2, atol=1e-14)
    np.testing.assert_allclose(plant.velocity, plant.gravity*plant.dt, atol=1e-14)
    np.testing.assert_allclose(plant.omega/plant.dt, expected_alpha, atol=1e-4)
    assert np.linalg.norm(plant.quaternion) == pytest.approx(1.)
    assert not np.allclose(plant.quaternion, [1, 0, 0, 0])
