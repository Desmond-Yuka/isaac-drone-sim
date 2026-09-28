"""YAML composition (extends), kind-aware merging and dotted overrides."""

import pytest

from isaac_drone.config import (
    DEFAULT_CONFIG,
    HELIX_CONFIG,
    ConfigurationError,
    apply_override,
    load_config,
    load_yaml_tree,
    merge,
)


def test_merge_recurses_replaces_lists_and_switches_kind_wholesale():
    base = {"a": {"x": 1, "y": [1, 2]}, "plugin": {"kind": "one", "p": 1, "q": 2}}
    child = {"a": {"y": [3]}, "plugin": {"kind": "two", "r": 3}}
    merged = merge(base, child)
    assert merged == {"a": {"x": 1, "y": [3]}, "plugin": {"kind": "two", "r": 3}}
    assert merge(base, {"plugin": {"q": 5}})["plugin"] == {"kind": "one", "p": 1, "q": 5}
    assert base["plugin"]["q"] == 2


def test_helix_config_extends_base_and_only_states_differences():
    base, helix = load_config(DEFAULT_CONFIG), load_config(HELIX_CONFIG)
    assert helix["scene"] == base["scene"] and helix["effects"] == base["effects"]
    assert helix["controller"]["attitude_kp"] == base["controller"]["attitude_kp"]
    assert helix["controller"]["position_kp"] == [12.0, 12.0, 8.0]
    assert base["trajectory"]["kind"] == "hold" and helix["trajectory"]["kind"] == "helix"
    assert "position_w_m" not in helix["trajectory"]
    assert "extends" not in helix


def test_extends_resolves_relative_paths_and_rejects_cycles(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "child.yaml").write_text(f"extends: {HELIX_CONFIG}\nsimulation:\n  seed: 7\n")
    (tmp_path / "grandchild.yaml").write_text("extends: sub/child.yaml\ncompletion: null\n")
    config = load_config(tmp_path / "grandchild.yaml")
    assert config["simulation"]["seed"] == 7 and config["completion"] is None
    assert config["trajectory"]["kind"] == "helix"
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigurationError, match="Circular"):
        load_yaml_tree(tmp_path / "a.yaml")


def test_overrides_parse_yaml_values_and_validate_the_result():
    config = load_config(
        HELIX_CONFIG,
        overrides=["controller.position_kp=[10, 10, 6]", "trajectory.turns=2.5", "limits.max_acceleration_m_s2=null"],
    )
    assert config["controller"]["position_kp"] == [10, 10, 6]
    assert config["trajectory"]["turns"] == 2.5 and config["limits"]["max_acceleration_m_s2"] is None
    switched = load_config(HELIX_CONFIG, overrides=["trajectory={kind: hold}", "completion=null"])
    assert switched["trajectory"] == {"kind": "hold"}
    with pytest.raises(ConfigurationError):
        load_config(HELIX_CONFIG, overrides=["controller.position_kp=[1, 2]"])
    for bad in ("controller.position_kp", "nosuch.section=1", "=3", "controller..kp=1"):
        with pytest.raises(ConfigurationError):
            apply_override({"controller": {}}, bad)
