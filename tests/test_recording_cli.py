"""CLI recording selection and lazy simulator/encoder startup without Isaac Sim."""
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from isaac_drone.config import ConfigurationError, DEFAULT_RECORDING, load_config


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "standalone"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"recording_cli_test_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(monkeypatch):
    # The entry point prepends the repository path; leave the test process unchanged.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return load_script("run_arl")


@pytest.fixture
def recording_yaml(tmp_path):
    cfg = load_config()
    cfg["recording"]["enabled"] = True
    path = tmp_path / "recording.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_default_validation_reports_recording_disabled(cli, capsys):
    assert cli.main(["--validate"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["configuration"] == "valid"
    assert output["recording"] == DEFAULT_RECORDING
    assert output["recording"]["enabled"] is False


def test_record_video_and_format_overrides_are_reported(cli, capsys):
    assert cli.main(["--validate", "--record-video", "--video-fps", "100",
                     "--video-width", "640", "--video-height", "480",
                     "--video-cameras", "top", "follow"]) == 0
    assert json.loads(capsys.readouterr().out)["recording"] == {
        "enabled": True, "fps": 100, "width": 640, "height": 480,
        "cameras": ["top", "follow"],
    }


def test_no_record_video_overrides_enabled_yaml(cli, recording_yaml, capsys):
    assert cli.main(["--config", str(recording_yaml), "--validate", "--no-record-video"]) == 0
    recording = json.loads(capsys.readouterr().out)["recording"]
    assert recording == DEFAULT_RECORDING
    assert load_config(recording_yaml)["recording"]["enabled"] is True


def test_format_options_do_not_implicitly_enable_recording(cli, capsys):
    assert cli.main(["--validate", "--video-fps", "100", "--video-cameras", "follow"]) == 0
    recording = json.loads(capsys.readouterr().out)["recording"]
    assert recording["enabled"] is False
    assert recording["fps"] == 100
    assert recording["cameras"] == ["follow"]


@pytest.mark.parametrize("extra", [[], ["--record-video"]])
def test_validate_never_imports_simulator_or_video_encoder(cli, monkeypatch, capsys, extra):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"isaaclab", "isaacsim", "omni", "pxr", "imageio_ffmpeg"}
        assert name not in {"isaac_drone.recording", "isaac_drone.backends.video"}
        return real_import(name, *args, **kwargs)

    def no_process(*args, **kwargs):
        pytest.fail("Validation must not launch an encoder or simulator process")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    assert cli.main(["--validate", *extra]) == 0
    assert json.loads(capsys.readouterr().out)["configuration"] == "valid"


@pytest.mark.parametrize("extra,match", [
    (["--video-fps", "0"], "recording.fps"),
    (["--video-fps", "201"], "recording.fps.*physics frequency"),
    (["--video-cameras", "follow", "follow"], "recording.cameras"),
])
def test_invalid_video_overrides_are_rejected(cli, extra, match):
    with pytest.raises(ConfigurationError, match=match):
        cli.main(["--validate", "--record-video", *extra])


@pytest.fixture
def fake_launch(cli, monkeypatch):
    events = []
    state = SimpleNamespace(events=events)

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--livestream", type=int, default=-1)
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--enable_cameras", action="store_true")

        def __init__(self, args):
            events.append("launch")
            state.args = args
            self.app = SimpleNamespace(close=lambda: events.append("close"))
            state.app = self.app

    isaaclab = ModuleType("isaaclab")
    isaaclab_app = ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = AppLauncher
    isaaclab.app = isaaclab_app
    recording = ModuleType("isaac_drone.recording")
    recording.ensure_video_dependencies = lambda: events.append("dependencies")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setitem(sys.modules, "isaac_drone.recording", recording)
    monkeypatch.delenv("LIVESTREAM", raising=False)

    def run(app, config, asset_path, **kwargs):
        events.append("run")
        assert app is state.app
        state.config = config
        state.options = kwargs
        return 17

    monkeypatch.setattr(cli, "run_simulation", run)
    state.recording = recording
    return state


@pytest.mark.parametrize("yaml_enabled,extra,should_record,inspect_only", [
    (False, [], False, False),
    (False, ["--record-video"], True, False),
    (True, [], True, False),
    (True, ["--no-record-video"], False, False),
    (False, ["--record-video", "--inspect"], False, True),
    (True, ["--inspect"], False, True),
])
def test_encoder_and_camera_enablement_only_for_recorded_flight(
        cli, fake_launch, recording_yaml, yaml_enabled, extra, should_record, inspect_only):
    config_args = ["--config", str(recording_yaml)] if yaml_enabled else []
    assert cli.main([*config_args, *extra, "--headless"]) == 17
    assert fake_launch.events == (["dependencies"] if should_record else []) + ["launch", "run", "close"]
    assert fake_launch.args.enable_cameras is should_record
    assert fake_launch.options["inspect_only"] is inspect_only
    if not inspect_only:
        assert fake_launch.config["recording"]["enabled"] is should_record


def test_encoder_preflight_failure_prevents_simulator_launch(cli, fake_launch):
    def unavailable():
        fake_launch.events.append("dependencies")
        raise RuntimeError("test encoder unavailable")

    fake_launch.recording.ensure_video_dependencies = unavailable
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        cli.main(["--record-video"])
    assert fake_launch.events == ["dependencies"]


@pytest.mark.parametrize("entry", ["helix_ascent", "spiral_ascent"])
def test_helix_and_legacy_entry_points_inherit_recording_flags(cli, monkeypatch, capsys, entry):
    monkeypatch.setitem(sys.modules, "run_arl", cli)
    legacy = load_script("spiral_ascent")
    monkeypatch.setitem(sys.modules, "spiral_ascent", legacy)
    wrapper = legacy if entry == "spiral_ascent" else load_script(entry)
    assert wrapper.helix_main(["--validate", "--record-video", "--video-fps", "100",
                               "--video-cameras", "follow"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["trajectory"] == "helix"
    assert output["recording"]["enabled"] is True
    assert output["recording"]["fps"] == 100
    assert output["recording"]["cameras"] == ["follow"]
