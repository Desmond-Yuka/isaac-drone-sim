"""Named plugin registries: select an implementation with ``kind:`` in YAML.

A registry maps a kind name to a factory and its parameter dataclass::

    @CONTROLLERS.register("my_controller", params=MyParams)
    def build(params: MyParams, context) -> Controller: ...

``parse(section, path)`` validates ``{kind: ..., **params}`` into
``(kind, params)``; ``build(section, path, **context)`` also constructs it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .params import parse_params
from .validation import ConfigurationError


@dataclass(frozen=True)
class Entry:
    name: str
    factory: Callable
    params: type
    description: str


class Registry:
    def __init__(self, what: str):
        self.what = what
        self._entries: dict[str, Entry] = {}

    def register(self, name: str, *, params: type, description: str = ""):
        def decorator(factory):
            if name in self._entries:
                raise ValueError(f"Duplicate {self.what} kind: {name}")
            summary = description or next(iter((factory.__doc__ or "").strip().splitlines()), "")
            self._entries[name] = Entry(name, factory, params, summary)
            return factory
        return decorator

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entry(self, name: str) -> Entry:
        try:
            return self._entries[name]
        except KeyError:
            raise ConfigurationError(f"Unknown {self.what} kind {name!r}; available: {self.names()}") from None

    def parse(self, section: Mapping, path: str):
        if not isinstance(section, Mapping) or "kind" not in section:
            raise ConfigurationError(f"{path} must be a mapping with a 'kind' ({'/'.join(self.names())})")
        entry = self.entry(section["kind"])
        params = parse_params(entry.params, {key: value for key, value in section.items() if key != "kind"},
                              f"{path}[{entry.name}]")
        return entry.name, params

    def build(self, section: Mapping, path: str, **context):
        name, params = self.parse(section, path)
        return self._entries[name].factory(params, **context)
