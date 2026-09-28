"""Experiment configuration: strict YAML with ``extends`` composition and dotted overrides.

Importing this package imports no simulator. See docs/configuration.md.
"""
from isaac_drone.core.validation import ConfigurationError

from .loader import apply_override, load_yaml_tree, merge
from .schema import (
    CONFIG_DIR,
    DEFAULT_CONFIG,
    DEFAULT_RECORDING,
    HELIX_CONFIG,
    REPO_ROOT,
    SCHEMA_VERSION,
    assert_flight_ready,
    load_config,
    resolve_asset,
    validate_config,
)

__all__ = ["CONFIG_DIR", "DEFAULT_CONFIG", "DEFAULT_RECORDING", "HELIX_CONFIG", "REPO_ROOT", "SCHEMA_VERSION",
           "ConfigurationError", "apply_override", "assert_flight_ready", "load_config", "load_yaml_tree", "merge",
           "resolve_asset", "validate_config"]
