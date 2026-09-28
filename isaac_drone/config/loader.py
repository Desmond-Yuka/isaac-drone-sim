"""Strict YAML loading with ``extends`` composition and dotted command-line overrides.

* Duplicate keys, non-string keys and YAML errors are rejected.
* ``extends: <path>`` (relative to the file) loads a parent first. Child
  mappings merge recursively into the parent; lists and scalars replace. A
  child mapping whose ``kind`` differs from the parent's replaces it entirely,
  so switching ``trajectory``/``controller`` kinds never inherits stale params.
* ``KEY.PATH=VALUE`` overrides parse VALUE as YAML (``[1, 2, 3]``, ``null``,
  ``true``, ``{kind: hold}``) and replace the addressed value.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml

from isaac_drone.core.validation import ConfigurationError


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("YAML keys must be strings")
        if key in result:
            raise ConfigurationError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def parse_yaml(text: str, source: str = "<string>"):
    try:
        return yaml.load(text, Loader=_UniqueLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"{source}: {error}") from error


def merge(base, child):
    """Recursive merge of ``child`` onto ``base`` (see module docstring); inputs are not modified."""
    if not isinstance(base, Mapping) or not isinstance(child, Mapping):
        return deepcopy(child)
    if "kind" in child and "kind" in base and child["kind"] != base["kind"]:
        return deepcopy(child)
    result = deepcopy(dict(base))
    for key, value in child.items():
        result[key] = merge(result[key], value) if key in result else deepcopy(value)
    return result


def load_yaml_tree(path, _chain=()) -> dict:
    """Load one file and its ``extends`` ancestors into a single mapping."""
    path = Path(path).expanduser().resolve()
    if path in _chain:
        raise ConfigurationError(f"Circular extends: {' -> '.join(str(item) for item in (*_chain, path))}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"Cannot read configuration {path}: {error}") from error
    data = parse_yaml(text, str(path))
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"{path} must contain a YAML mapping")
    data = dict(data)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent:
        raise ConfigurationError(f"{path}: extends must be a relative or absolute file path")
    return merge(load_yaml_tree(path.parent / parent, (*_chain, path)), data)


def apply_override(config: dict, assignment: str) -> None:
    """Apply one ``a.b.c=value`` override in place."""
    if "=" not in assignment:
        raise ConfigurationError(f"Override {assignment!r} must look like key.path=value")
    key, text = assignment.split("=", 1)
    parts = key.strip().split(".")
    if not all(parts):
        raise ConfigurationError(f"Override key {key!r} is not a dotted path")
    value = parse_yaml(text, f"--set {key}") if text.strip() else None
    target = config
    for index, part in enumerate(parts[:-1]):
        if not isinstance(target, dict) or part not in target:
            raise ConfigurationError(f"Override {key!r}: {'.'.join(parts[:index + 1])} does not exist")
        target = target[part]
    if not isinstance(target, dict):
        raise ConfigurationError(f"Override {key!r}: {'.'.join(parts[:-1])} is not a mapping")
    target[parts[-1]] = value
