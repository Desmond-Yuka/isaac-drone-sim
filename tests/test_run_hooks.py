"""Run hooks against a real loop (synthetic plant), where trajectory timing matters."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_drone.config import load_config
from isaac_drone.runtime.hooks import PathOverlayHook
from isaac_drone.runtime.loop import MotionControlLoop
from isaac_drone.sim.synthetic import SyntheticBackend
from isaac_drone.viz.overlay import PathTrail

ROOT = Path(__file__).resolve().parents[1]


def test_path_overlay_setpoint_holds_start_pose_during_spin_up():
    config = load_config(ROOT / "configs/helix.yaml")
    backend = SyntheticBackend(config)
    loop = MotionControlLoop(config, backend)
    loop.reset()
    start_time = loop.trajectory.start_time_s
    assert start_time > 0
    # The raw trajectory is undefined before its start; the overlay must not sample it there.
    with pytest.raises(ValueError, match="precedes its start time"):
        loop.trajectory.sample(0.0)

    setpoints = []
    hook = PathOverlayHook(SimpleNamespace(update=lambda points, setpoint: setpoints.append(setpoint)), PathTrail())
    hook.start(SimpleNamespace(loop=loop))
    hook.refresh()
    while loop.time_s < start_time / 2:
        loop.prepare_step()
        backend.step()
        hook.after_step(None, loop.finish_step())
    hook.refresh()

    hold = loop.trajectory.sample(start_time).position_w
    assert len(setpoints) == 2
    for setpoint in setpoints:
        np.testing.assert_allclose(setpoint, hold)
