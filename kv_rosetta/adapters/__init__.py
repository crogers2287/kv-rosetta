from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Adapter, AdapterError

if TYPE_CHECKING:
    from typing import Type

_REGISTRY: dict[str, "Type[Adapter]"] = {}


def register(name: str, cls: "Type[Adapter]") -> None:
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        raise AdapterError(f"{cls!r} is not a subclass of Adapter")
    if name in _REGISTRY:
        raise AdapterError(f"adapter name already registered: {name!r}")
    _REGISTRY[name] = cls


def get_adapter(name: str) -> "Type[Adapter]":
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise AdapterError(f"unknown adapter {name!r}; known adapters: {known}")


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
