"""Optional video recording stays strict without breaking existing YAML files."""
from copy import deepcopy

import pytest
import yaml

from isaac_drone.config import (
    ConfigurationError,
    DEFAULT_CONFIG,
    DEFAULT_RECORDING,
    HELIX_CONFIG,
    load_config,
    validate_config,
)


@pytest.mark.parametrize("path", [DEFAULT_CONFIG, HELIX_CONFIG])
def test_presets_leave_recording_disabled(path):
    recording = load_config(path)["recording"]
    assert recording == DEFAULT_RECORDING
    assert recording["fps"] == 60


def test_legacy_config_loads_with_independent_recording_defaults(tmp_path):
    cfg = load_config()
    del cfg["recording"]
    # A disabled recorder must not constrain the old simulation frequency.
    cfg["simulation"]["dt"] = 0.1
    original = deepcopy(cfg)
    validate_config(cfg)
    assert cfg == original
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["recording"] == DEFAULT_RECORDING
    loaded["recording"]["enabled"] = True
    loaded["recording"]["cameras"].clear()
    assert load_config(path)["recording"] == DEFAULT_RECORDING
    assert not DEFAULT_RECORDING["enabled"]
    assert DEFAULT_RECORDING["cameras"] == ["overview", "follow", "top"]


@pytest.mark.parametrize("field,value", [
    ("enabled", "false"),
    ("enabled", 0),
    ("enabled", None),
    ("fps", True),
    ("fps", 0),
    ("fps", -1),
    ("fps", 60.0),
    ("fps", "60"),
    ("width", False),
    ("width", 0),
    ("width", 1280.0),
    ("width", 1279),
    ("height", -2),
    ("height", 720.0),
    ("height", 719),
    ("cameras", None),
    ("cameras", "follow"),
    ("cameras", []),
    ("cameras", ["follow", "follow"]),
    ("cameras", ["follow", "unknown"]),
    ("cameras", [True]),
    ("cameras", [["follow"]]),
])
def test_invalid_recording_values_fail_even_when_disabled(field, value):
    cfg = load_config()
    cfg["recording"][field] = value
    with pytest.raises(ConfigurationError, match=f"recording.{field}"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [None, False, [], {}, {"enabled": True}])
def test_explicit_recording_section_requires_complete_mapping(value):
    cfg = load_config()
    cfg["recording"] = value
    with pytest.raises(ConfigurationError, match="recording"):
        validate_config(cfg)


def test_unknown_recording_fields_fail():
    cfg = load_config()
    cfg["recording"]["enable"] = True
    with pytest.raises(ConfigurationError, match="unknown=.*enable"):
        validate_config(cfg)


@pytest.mark.parametrize("fps", [30, 60, 200])
def test_enabled_recording_accepts_nondivisor_and_physics_frequency(fps):
    cfg = load_config()
    cfg["simulation"]["dt"] = 0.005
    cfg["recording"].update(enabled=True, fps=fps)
    validate_config(cfg)


def test_enabled_recording_rejects_rate_above_physics_frequency():
    cfg = load_config()
    cfg["simulation"]["dt"] = 0.005
    cfg["recording"].update(enabled=True, fps=201)
    with pytest.raises(ConfigurationError, match="recording.fps.*physics frequency"):
        validate_config(cfg)


def test_disabled_recording_does_not_constrain_physics_frequency():
    cfg = load_config()
    cfg["simulation"]["dt"] = 0.1
    validate_config(cfg)


@pytest.mark.parametrize("cameras", [["follow"], ["top", "overview"], ["top", "overview", "follow"]])
def test_recording_accepts_camera_subsets_in_requested_order(cameras):
    cfg = load_config()
    cfg["recording"].update(enabled=True, cameras=cameras)
    validate_config(cfg)
    assert cfg["recording"]["cameras"] == cameras
