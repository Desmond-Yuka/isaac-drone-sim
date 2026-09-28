"""Closed-loop numerical regression against traces recorded before the modular refactor.

tests/data/golden_synthetic.npz was produced at commit 7a458b4 by the original
MotionControlLoop flying the synthetic CPU plant: the minimum-time helix config
for 20 s and the hold config (airborne start at 200 rps) for 5 s. Rows are
subsampled every ``<name>_stride`` physics steps (row k = state after step
stride*(k+1)). Refactors must reproduce them to round-off; a deliberate change
of flight behaviour must regenerate the file and say so in its commit.
"""
from pathlib import Path

import numpy as np
import pytest

from isaac_drone.config import load_config
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.sim.synthetic import SyntheticBackend

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = np.load(Path(__file__).parent / "data" / "golden_synthetic.npz")
CONFIGS = {"helix": ROOT / "configs/arl_robot_1_helix.yaml", "hold": ROOT / "configs/arl_robot_1.yaml"}
TOLERANCE = {"rtol": 1e-9, "atol": 1e-9}


def fly(name, steps):
    config = load_config(CONFIGS[name])
    plant = SyntheticBackend(config)
    loop = MotionControlLoop(config, plant)
    loop.reset()
    stride = int(GOLDEN[f"{name}_stride"])
    rows = {key: [] for key in ("pos", "quat", "rps", "wrench", "target")}
    errors, record = [], None
    for index in range(steps):
        loop.prepare_step()
        plant.step()
        record = loop.finish_step()
        basic = record["basic"]
        errors.append(basic["position_error_norm_m"])
        if (index + 1) % stride == 0:
            for key, value in (("pos", basic["position_w_m"]), ("quat", basic["quaternion_wxyz"]),
                               ("rps", basic["motor_speed_rps"]), ("wrench", record["requested_wrench_b"]),
                               ("target", basic["target_position_w_m"])):
                rows[key].append(value)
    return {key: np.asarray(value) for key, value in rows.items()}, np.asarray(errors), record


def assert_matches_golden(name, rows):
    count = len(rows["pos"])
    for key, actual in rows.items():
        np.testing.assert_allclose(actual, GOLDEN[f"{name}_{key}"][:count], **TOLERANCE, err_msg=f"{name}.{key}")


def test_helix_prefix_through_spin_up_takeoff_and_early_helix():
    rows, _, record = fly("helix", 800)  # 4 s: spin-up, full takeoff, start of the helix
    assert_matches_golden("helix", rows)
    assert record["mission_phase"] not in ("delay", "takeoff", "hold")


def test_hold_from_airborne_start():
    rows, errors, _ = fly("hold", 1000)
    assert_matches_golden("hold", rows)
    assert errors.max() == pytest.approx(float(GOLDEN["hold_max_error"]), rel=1e-9)


@pytest.mark.slow
def test_full_minimum_time_helix_mission():
    rows, errors, record = fly("helix", 4000)
    assert_matches_golden("helix", rows)
    assert errors.max() == pytest.approx(float(GOLDEN["helix_max_error"]), rel=1e-9)
    assert np.sqrt(np.mean(errors**2)) == pytest.approx(float(GOLDEN["helix_rms_error"]), rel=1e-9)
    assert record["mission"]["achieved"] and not record["mission"]["timed_out"]
