"""
Phase registry for the SeedDataGen pipeline.

Phase classes self-register by applying the @register decorator.  The runner
discovers all phases by importing every phase_*.py module, then looks them up
by name via get_phase().

Usage in a phase file:
    from SeedDataGen.registry import register
    from SeedDataGen.base_phase import Phase

    @register
    class MyPhase(Phase):
        name = "my_phase"
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SeedDataGen.base_phase import Phase

_registry: dict[str, type["Phase"]] = {}


def register(cls: type["Phase"]) -> type["Phase"]:
    """Decorator: add *cls* to the phase registry under cls.name."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Phase class {cls!r} must define a non-empty 'name' attribute.")
    if cls.name in _registry:
        raise ValueError(
            f"A phase named '{cls.name}' is already registered "
            f"({_registry[cls.name]!r}). Phase names must be unique."
        )
    _registry[cls.name] = cls
    return cls


def get_phase(name: str) -> type["Phase"]:
    """Return the phase class registered under *name*, or raise KeyError."""
    if name not in _registry:
        available = sorted(_registry.keys())
        raise KeyError(
            f"No phase named '{name}' is registered. "
            f"Available phases: {available}"
        )
    return _registry[name]


def list_phases() -> list[str]:
    """Return a sorted list of all registered phase names."""
    return sorted(_registry.keys())
