"""Exercise the real standalone scheduling/logging with fake physics and RGB I/O."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import isaac_drone.backends.isaaclab as backend_module
import isaac_drone.backends.video as video_backend
from isaac_drone.config import DEFAULT_RECORDING, load_config
import isaac_drone.playback as playback_module
import isaac_drone.recording as recording_module
import isaac_drone.runtime as runtime_module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    entry_path = Path(__file__).resolve().parents[1] / "scripts/standalone/run_arl.py"
    spec = importlib.util.spec_from_file_location("recording_test_run_arl", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    state = types.SimpleNamespace(physics_steps=0, rendered=0, rig_reads=[], rigs=[],
                                  writers=[], recorders=[], pacer_frames=0, lifecycle=[],
                                  displayed=False, interrupt=None)

    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__path__ = []
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        if "." in name:
            parent, field = name.rsplit(".", 1)
            monkeypatch.setattr(sys.modules[parent], field, result, raising=False)
        return result

    def actual_position(step):
        return np.array([step * 0.01, step * -0.02, 2.0 + step * 0.03])

    class Simulation:
        def __init__(self, config):
            self.is_rendering = state.displayed

        def reset(self):
            state.lifecycle.append("sim.reset")

        def is_stopped(self):
            return False

        def is_playing(self):
            return True

        def step(self, *, render):
            assert render is False
            state.physics_steps += 1

        def render(self):
            state.rendered += 1

    class Loop:
        def __init__(self, config, backend):
            self.is_helix, self.mission, self.feasibility = False, None, None
            self.step_index, self.time_s = 0, 0.0
            self.dt_s = config["simulation"]["dt"]
            # Deliberately far from truth: camera tracking must not use this.
            self.trajectory = types.SimpleNamespace(
                sample=lambda time: types.SimpleNamespace(position_w=np.array([100.0, 100.0, 100.0])))

        def reset(self):
            state.lifecycle.append("loop.reset")

        def prepare_step(self):
            if state.interrupt is not None and self.step_index == 20:
                raise state.interrupt("interrupted flight")

        def finish_step(self):
            self.step_index += 1
            self.time_s = self.step_index * self.dt_s
            assert self.step_index == state.physics_steps
            return {"step": self.step_index,
                    "post_step_state": {"position_w": actual_position(self.step_index)}}

    class Rig:
        def __init__(self, sim, width, height, views):
            assert state.lifecycle[:2] == ["sim.reset", "loop.reset"]
            self.width, self.height, self.closed = width, height, False
            state.lifecycle.append("rig.open")
            state.rigs.append(self)

        def read(self, poses):
            assert not self.closed
            state.rig_reads.append((state.physics_steps, copy.deepcopy(poses)))
            return {name: np.full((self.height, self.width, 3), len(state.rig_reads), dtype=np.uint8)
                    for name in poses}

        def close(self):
            self.closed = True
            state.lifecycle.append("rig.close")

    class Writer:
        def __init__(self, path, **settings):
            self.path, self.settings, self.frames, self.closed = path, settings, [], False
            state.writers.append(self)

        def write(self, frame):
            assert not self.closed
            self.frames.append(frame.copy())

        def close(self):
            self.closed = True
            state.lifecycle.append("writer.close")

    class SkippingPacer:
        def __init__(self, speed):
            pass

        def start(self, time):
            pass

        def frame(self, time, draw):
            state.pacer_frames += 1  # Deliberately skip every display draw.

        def summary(self):
            return {"skipped_frames": state.pacer_frames}

    real_recorder = recording_module.MultiViewRecorder

    def recorder_factory(directory, config):
        recorder = real_recorder(directory, config, writer_factory=Writer)
        state.recorders.append(recorder)
        return recorder

    module("torch", manual_seed=lambda seed: None,
           cuda=types.SimpleNamespace(is_available=lambda: False))
    module("isaaclab")
    module("isaaclab.sim", SimulationContext=Simulation)
    module("isaaclab_contrib")
    module("isaaclab_contrib.assets", Multirotor=lambda cfg: types.SimpleNamespace(update=lambda dt: None))
    module("isaac_drone.plots", plot_run=lambda path: [path / "plots" / "test.png"])
    monkeypatch.setattr(backend_module, "ARLBackend", lambda *args: types.SimpleNamespace(
        read_state=lambda time: types.SimpleNamespace(position_w=actual_position(0)), telemetry=lambda: {}))
    monkeypatch.setattr(backend_module, "make_robot_cfg", lambda *args: None)
    monkeypatch.setattr(backend_module, "make_simulation_cfg", lambda config: config)
    monkeypatch.setattr(backend_module, "apply_usd_overrides", lambda *args: None)
    monkeypatch.setattr(runtime_module, "MotionControlLoop", Loop)
    monkeypatch.setattr(video_backend, "IsaacCameraRig", Rig)
    monkeypatch.setattr(recording_module, "MultiViewRecorder", recorder_factory)
    monkeypatch.setattr(playback_module, "RealTimePacer", SkippingPacer)
    monkeypatch.setattr(entry, "asset_hashes", lambda path: {})

    config = load_config()
    config["simulation"].update(dt=0.005, duration_s=1.0, render_interval=4)
    config["scene"]["ground"]["enabled"] = False
    config["scene"]["light"]["enabled"] = False
    config["logging"]["directory"] = str(tmp_path / "runs")
    config["recording"] = {**DEFAULT_RECORDING, "enabled": True, "fps": 60, "width": 4, "height": 2,
                           "cameras": ["overview", "follow", "top"]}
    state.config, state.entry, state.actual_position = config, entry, actual_position
    state.run = lambda: entry.run_simulation(types.SimpleNamespace(is_running=lambda: True), config,
                                           tmp_path / "robot.usd", path_overlay=False)

    def records():
        path = next((tmp_path / "runs").glob("*/telemetry.jsonl"))
        return [json.loads(line) for line in path.read_text().splitlines()]

    state.records = records
    return state


def test_disabled_recording_does_not_import_or_construct_video_resources(runtime, monkeypatch):
    runtime.config["recording"]["enabled"] = False
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name not in {"isaac_drone.recording", "isaac_drone.recording_views", "isaac_drone.backends.video"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert runtime.run() == 0
    assert runtime.physics_steps == 200
    assert not runtime.rigs and not runtime.writers
    assert runtime.records()[-1]["video"] is None


@pytest.mark.parametrize("displayed", [False, True])
def test_three_views_record_60_synchronized_frames_even_when_display_skips(runtime, displayed):
    runtime.displayed = displayed
    assert runtime.run() == 0
    assert runtime.physics_steps == 200
    assert len(runtime.rig_reads) == 60
    assert runtime.rig_reads[0][0] == 0
    assert runtime.rig_reads[-1][0] == 197
    assert len(runtime.writers) == 4
    assert all(len(writer.frames) == 60 and writer.closed for writer in runtime.writers)
    assert runtime.rigs[0].closed
    assert runtime.lifecycle.index("rig.close") < runtime.lifecycle.index("writer.close")
    first_overview = runtime.rig_reads[0][1]["overview"]
    for step, poses in runtime.rig_reads:
        assert set(poses) == {"overview", "follow", "top"}
        np.testing.assert_allclose(poses["follow"]["target"], runtime.actual_position(step))
        np.testing.assert_allclose(poses["top"]["target"], runtime.actual_position(step))
        np.testing.assert_allclose(poses["overview"]["eye"], first_overview["eye"])
    events = runtime.records()
    assert events[-1]["event"] == "finished" and events[-1]["success"] is True
    assert events[-1]["video"]["frames"] == 60
    assert events[-1]["video"]["closed"] is True
    assert runtime.rendered == 0
    assert runtime.pacer_frames == (50 if displayed else 0)


@pytest.mark.parametrize("duration, steps, sample_times", [
    (0.035, 7, [0.0, 0.02, 0.035]),
    (0.001, 1, [0.0]),
])
def test_final_physics_state_fills_due_slots_without_extra_trailing_frame(runtime, duration, steps, sample_times):
    runtime.config["simulation"]["duration_s"] = duration
    assert runtime.run() == 0
    assert runtime.physics_steps == steps
    assert len(runtime.rig_reads) == len(sample_times)
    assert all(len(writer.frames) == len(sample_times) and writer.closed for writer in runtime.writers)
    index_path = runtime.recorders[0].path / "video_frames.jsonl"
    index = [json.loads(line) for line in index_path.read_text().splitlines()]
    np.testing.assert_allclose([frame["simulation_time_s"] for frame in index], sample_times)
    np.testing.assert_allclose([frame["video_time_s"] for frame in index], np.arange(len(sample_times)) / 60)
    assert all(frame["video_time_s"] < steps * 0.005 for frame in index)
    assert runtime.records()[-1]["video"]["frames"] == len(sample_times)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_exception_or_ctrl_c_closes_every_video_and_records_aborted(runtime, failure):
    runtime.interrupt = failure
    with pytest.raises(failure, match="interrupted flight"):
        runtime.run()
    assert runtime.physics_steps == 20
    assert all(writer.closed for writer in runtime.writers)
    assert runtime.rigs[0].closed
    assert runtime.lifecycle.index("rig.close") < runtime.lifecycle.index("writer.close")
    event = runtime.records()[-1]
    assert event["event"] == "aborted"
    assert event["video"]["closed"] is True
    assert event["video"]["frames"] == len(runtime.rig_reads)
