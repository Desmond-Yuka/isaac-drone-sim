"""Run directory names, the run clock, sweep point names and the per-run video copy."""

import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from isaac_drone.config import DEFAULT_CONFIG, HELIX_CONFIG, ConfigurationError, load_config
from isaac_drone.runtime.hooks import VideoHook
from isaac_drone.sim.synthetic import run_simulation
from isaac_drone.telemetry.naming import (
    VIDEO_DIR,
    available_path,
    create_unique_directory,
    run_directory_name,
    safe_name,
    stamp,
)

# 2026-10-15 00:07:27 UTC is 2026-10-14 20:07 in New York (EDT, UTC-4).
STARTED = datetime(2026, 10, 15, 0, 7, 27, tzinfo=timezone.utc)


def test_stamp_uses_the_new_york_clock_in_summer_and_winter():
    assert stamp(STARTED) == "1014-2007"
    assert stamp(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)) == "0115-0700"  # EST, UTC-5


@pytest.mark.parametrize(
    "config, metadata, expected",
    [
        (
            {"logging": {"run_name": "helix"}, "recording": {"enabled": True}},
            {"backend_kind": "isaaclab"},
            "1014-2007_helix_isaac_video",
        ),
        (
            {"logging": {"run_name": "helix"}, "recording": {"enabled": False}},
            {"backend_kind": "synthetic"},
            "1014-2007_helix_synth",
        ),
        ({"logging": {}, "trajectory": {"kind": "hold"}}, {}, "1014-2007_hold"),
        ({"logging": {"flush_every_n_records": 1}}, {}, "1014-2007_run"),
        ({"logging": {"run_name": "my test/../x"}}, {"backend_kind": "other"}, "1014-2007_my-test-..-x"),
    ],
)
def test_directory_name_is_time_experiment_backend_and_video_marker(config, metadata, expected):
    assert run_directory_name(config, metadata, STARTED) == expected


def test_safe_name_keeps_shell_friendly_characters_only():
    assert safe_name("helix_moderate-2.yaml") == "helix_moderate-2.yaml"
    assert safe_name("../") == "run"


def test_taken_names_get_a_numeric_suffix(tmp_path):
    assert available_path(tmp_path, "x") == tmp_path / "x"
    assert not (tmp_path / "x").exists()
    created = [create_unique_directory(tmp_path / "runs", "1014-2007_helix_synth") for _ in range(3)]
    assert [path.name for path in created] == [
        "1014-2007_helix_synth",
        "1014-2007_helix_synth_2",
        "1014-2007_helix_synth_3",
    ]
    assert all(path.is_dir() for path in created)
    assert available_path(tmp_path / "runs", "1014-2007_helix_synth").name == "1014-2007_helix_synth_4"


def test_run_name_defaults_to_the_yaml_file_name_and_can_be_overridden():
    assert load_config(DEFAULT_CONFIG)["logging"]["run_name"] == "arl_robot_1"
    assert load_config(HELIX_CONFIG)["logging"]["run_name"] == "helix"
    assert load_config(HELIX_CONFIG, overrides=["logging.run_name=baseline"])["logging"]["run_name"] == "baseline"
    for bad in ('""', "5", "[a]"):
        with pytest.raises(ConfigurationError, match="run_name"):
            load_config(HELIX_CONFIG, overrides=[f"logging.run_name={bad}"])


def test_runs_are_named_by_start_time_config_and_backend(tmp_path):
    config = load_config(DEFAULT_CONFIG, overrides=["simulation.duration_s=0.05"])
    result = run_simulation(config, figures=False, log=lambda *_: None, log_root=tmp_path)
    assert re.fullmatch(r"\d{4}-\d{4}_arl_robot_1_synth(_\d+)?", result.run_dir.name)
    started = json.loads((result.run_dir / "metadata.json").read_text())["provenance"]["started_utc"]
    assert result.run_dir.name.startswith(stamp(datetime.fromisoformat(started)))
    named = run_simulation(config, figures=False, log=lambda *_: None, log_root=tmp_path, directory_name="07")
    assert named.run_dir == tmp_path / "07"


def video_hook(tmp_path, summary, logs):
    run_dir = tmp_path / "runs" / "1014-2007_helix_isaac_video"
    run_dir.mkdir(parents=True)
    hook = VideoHook({}, None, None, None, log=logs.append)
    hook.video = SimpleNamespace(summary=lambda: summary)
    return hook, SimpleNamespace(run_dir=run_dir)


def summary(**changes):
    return {"failed": False, "frames": 60, "videos": {"combined": {"filename": "combined.mp4"}}, **changes}


def test_finished_video_is_copied_to_the_shared_video_folder(tmp_path):
    logs = []
    hook, result = video_hook(tmp_path, summary(), logs)
    (result.run_dir / "combined.mp4").write_bytes(b"mp4 bytes")
    hook.finalize(result)
    copy = tmp_path / "runs" / VIDEO_DIR / "1014-2007_helix_isaac_video.mp4"
    assert copy.read_bytes() == b"mp4 bytes"
    assert (result.run_dir / "combined.mp4").is_file()
    assert logs == [f"Video copy: {copy}"]


@pytest.mark.parametrize("changes", [{"failed": True}, {"frames": 0}])
def test_failed_or_empty_recordings_are_not_copied(tmp_path, changes):
    logs = []
    hook, result = video_hook(tmp_path, summary(**changes), logs)
    (result.run_dir / "combined.mp4").write_bytes(b"partial")
    hook.finalize(result)
    assert not (tmp_path / "runs" / VIDEO_DIR).exists() and logs == []


def test_a_failed_copy_is_reported_and_does_not_raise(tmp_path):
    logs = []
    hook, result = video_hook(tmp_path, summary(), logs)
    hook.finalize(result)  # combined.mp4 is missing
    assert len(logs) == 1 and logs[0].startswith("Video not copied")


def test_recording_disabled_has_no_video_to_copy(tmp_path):
    logs = []
    hook = VideoHook({}, None, None, None, log=logs.append)
    hook.finalize(SimpleNamespace(run_dir=tmp_path / "run"))
    assert logs == [] and not (tmp_path / VIDEO_DIR).exists()
