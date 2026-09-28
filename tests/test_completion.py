"""Measured hover acceptance must not be confused with a completed reference."""

from dataclasses import replace

import numpy as np
import pytest

from isaac_drone.config import HELIX_CONFIG, load_config, validate_config
from isaac_drone.core.types import TrajectorySetpoint, VehicleState
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.sim.synthetic import SyntheticBackend
from isaac_drone.trajectories.completion import CompletionMonitor


def sample(time, **kwargs):
    return replace(VehicleState(time, np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3)), **kwargs)


def monitor():
    return CompletionMonitor(load_config(HELIX_CONFIG)["completion"])


def test_reference_finishing_alone_is_not_success_and_reset_discards_dwell():
    m, target = monitor(), TrajectorySetpoint(np.zeros(3))
    assert not m.update(sample(29), target, "helix")["achieved"]
    assert not m.update(sample(30), target, "hold")["achieved"]
    assert not m.update(sample(31.99), target, "hold")["achieved"]
    assert m.update(sample(32), target, "hold")["achieved"]
    m.reset()
    assert not m.update(sample(40), target, "hold")["achieved"]


@pytest.mark.parametrize(
    "change",
    [
        {"position_w": np.array([0.11, 0, 0])},
        {"linear_velocity_w": np.array([0, 0.11, 0])},
        {"angular_velocity_b": np.array([0, 0, 0.11])},
        {"quaternion_wxyz": np.array([np.cos(0.11), 0, 0, np.sin(0.11)])},
    ],
)
def test_each_physical_error_breaks_continuous_dwell_and_revokes_success(change):
    m, target = monitor(), TrajectorySetpoint(np.zeros(3))
    m.update(sample(30), target, "hold")
    assert m.update(sample(32), target, "hold")["achieved"]
    status = m.update(sample(33, **change), target, "hold")
    assert status["ever_achieved"] and not status["achieved"]
    assert status["continuous_dwell_s"] == 0
    assert not m.update(sample(34), target, "hold")["achieved"]
    assert m.update(sample(36), target, "hold")["achieved"]


def test_timeout_is_not_overwritten_by_late_success_and_yaw_is_wrapped():
    m, target = monitor(), TrajectorySetpoint(np.zeros(3), yaw_rad=6 * np.pi)
    assert m.update(sample(30), target, "hold")["within_tolerances"]
    m.update(sample(31, position_w=np.ones(3)), target, "hold")
    assert m.update(sample(40), target, "hold")["timed_out"]
    status = m.update(sample(42), target, "hold")
    assert status["timed_out"] and not status["achieved"]
    with pytest.raises(ValueError, match="increase"):
        m.update(sample(42), target, "hold")


def test_helix_preset_requires_a_stopped_ground_start_and_a_complete_mission():
    cfg = load_config(HELIX_CONFIG)
    assert cfg["vehicle"]["launch"]["from_ground"]
    assert cfg["trajectory"]["kind"] == "helix"
    assert not any(cfg["vehicle"]["initial_state"]["rps"].values())
    for section, key, value in [("launch", "spin_up_s", 0.0), ("initial_state", "lin_vel", [0, 0, 0.1])]:
        bad = load_config(HELIX_CONFIG)
        bad["vehicle"][section][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    # Mission length is known only after (minimum-time) planning, so reset checks the duration.
    cfg["simulation"]["duration_s"] = 12.0
    with pytest.raises(ValueError, match="dwell"):
        MotionControlLoop(cfg, SyntheticBackend(cfg)).reset()


@pytest.mark.parametrize("field,value", [("dwell_time_s", 0), ("yaw_tolerance_rad", 4.0), ("max_hold_time_s", 1.0)])
def test_invalid_completion_criteria_are_rejected(field, value):
    cfg = load_config(HELIX_CONFIG)
    cfg["completion"][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
