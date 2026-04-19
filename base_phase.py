"""
Base class and role system for SeedDataGen pipeline phases.

Every phase declares:
  - name: unique string key used in pipeline.yaml and the registry
  - role: PhaseRole enum value
  - input_schema: expected Pydantic row type (None for generators)
  - output_schema: emitted Pydantic row type

Compatibility between consecutive phases is checked in two layers:
  1. Role transition — governed by COMPATIBLE_TRANSITIONS
  2. Schema subclass  — prev.output_schema must be a subclass of self.input_schema
"""

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Set, Tuple, Type

from pydantic import BaseModel


class PhaseRole(str, Enum):
    GENERATOR = "generator"
    EDITOR = "editor"
    FILTER = "filter"
    JUDGE = "judge"
    DEDUP = "dedup"


COMPATIBLE_TRANSITIONS: dict[PhaseRole, Set[PhaseRole]] = {
    PhaseRole.GENERATOR: {PhaseRole.EDITOR, PhaseRole.FILTER, PhaseRole.JUDGE, PhaseRole.DEDUP},
    PhaseRole.EDITOR: {PhaseRole.EDITOR, PhaseRole.FILTER, PhaseRole.JUDGE, PhaseRole.DEDUP},
    PhaseRole.FILTER: {PhaseRole.EDITOR, PhaseRole.FILTER, PhaseRole.JUDGE, PhaseRole.DEDUP},
    PhaseRole.JUDGE: {PhaseRole.FILTER, PhaseRole.DEDUP},
    PhaseRole.DEDUP: {PhaseRole.EDITOR, PhaseRole.FILTER},
}


class Phase(ABC):
    """Abstract base class for all pipeline phases."""

    name: str
    role: PhaseRole
    input_schema: Optional[Type[BaseModel]]  # None for GENERATOR phases
    output_schema: Type[BaseModel]

    def check_compatible_with(self, prev: "Phase", force: bool = False) -> None:
        """
        Validate that this phase can follow *prev*.

        Layer 1: role transition (bypassable with force=True, emits a warning).
        Layer 2: schema subclass (never bypassable).

        Raises TypeError with a descriptive message on failure.
        """
        allowed_roles = COMPATIBLE_TRANSITIONS.get(prev.role, set())
        if self.role not in allowed_roles:
            msg = (
                f"Role transition '{prev.role}' → '{self.role}' is not allowed "
                f"('{prev.name}' → '{self.name}'). "
                f"Allowed next roles after '{prev.role}': "
                f"{sorted(r.value for r in allowed_roles)}."
            )
            if force:
                warnings.warn(f"[force=True] {msg}", stacklevel=2)
            else:
                raise TypeError(msg)

        if self.input_schema is not None:
            if not issubclass(prev.output_schema, self.input_schema):
                raise TypeError(
                    f"Schema mismatch: '{prev.name}' produces "
                    f"'{prev.output_schema.__name__}' but '{self.name}' expects "
                    f"'{self.input_schema.__name__}' (or a subclass). "
                    f"This cannot be bypassed with force=True."
                )

    def describe_prompts(self) -> List[Tuple[str, str]]:
        return []

    @abstractmethod
    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        """Execute the phase, reading from *input_file* and writing to *output_file*."""
        ...
