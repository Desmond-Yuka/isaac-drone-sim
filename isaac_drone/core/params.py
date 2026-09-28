"""Typed plugin parameters: parse a YAML mapping into a frozen dataclass, strictly.

Every trajectory/controller plugin declares its parameters as a dataclass. The
parser rejects unknown and missing keys, booleans used as numbers, strings
used as numbers, NaN/Inf and wrongly sized vectors; range checks belong in
the dataclass's ``__post_init__`` (raise ``ValueError``). Supported field
types: ``float``, ``int``, ``bool``, ``str``, ``Literal[...]``, fixed-length
``tuple[float, ...]`` (e.g. ``Vec3``), variable ``tuple[X, ...]`` and ``X | None``.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from collections.abc import Mapping
from numbers import Integral, Real

from .validation import ConfigurationError

Vec3 = tuple[float, float, float]


def _convert(annotation, value, path):
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        options = [item for item in arguments if item is not type(None)]
        if value is None:
            if type(None) in arguments:
                return None
            raise ConfigurationError(f"{path} must not be null")
        if len(options) != 1:
            raise ConfigurationError(f"{path}: unsupported union annotation {annotation}")
        return _convert(options[0], value, path)
    if value is None:
        raise ConfigurationError(f"{path} must not be null")
    if origin is typing.Literal:
        if value not in arguments:
            raise ConfigurationError(f"{path} must be one of {list(arguments)}; got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigurationError(f"{path} must be a list")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_convert(arguments[0], item, f"{path}[{i}]") for i, item in enumerate(value))
        if len(value) != len(arguments):
            raise ConfigurationError(f"{path} must have {len(arguments)} entries; got {len(value)}")
        return tuple(_convert(kind, item, f"{path}[{i}]") for i, (kind, item) in enumerate(zip(arguments, value)))
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigurationError(f"{path} must be true or false")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ConfigurationError(f"{path} must be an integer")
        return int(value)
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ConfigurationError(f"{path} must be a finite number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"{path} must be a nonempty string")
        return value
    raise ConfigurationError(f"{path}: unsupported parameter annotation {annotation}")


def parse_params(cls, mapping, path: str):
    """Build ``cls`` (a dataclass) from ``mapping``; errors name the full config path."""
    if not isinstance(mapping, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls) if field.init}
    required = {
        name
        for name, field in fields.items()
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }
    unknown, missing = set(mapping) - set(fields), required - set(mapping)
    if unknown or missing:
        raise ConfigurationError(f"{path}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    values = {name: _convert(hints[name], value, f"{path}.{name}") for name, value in mapping.items()}
    try:
        return cls(**values)
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}: {error}") from error


def params_to_dict(params) -> dict:
    """Plain YAML/JSON-ready mapping of a parameter dataclass (tuples become lists)."""

    def plain(value):
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    return {field.name: plain(getattr(params, field.name)) for field in dataclasses.fields(params)}
