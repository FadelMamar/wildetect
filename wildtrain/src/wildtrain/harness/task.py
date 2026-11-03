"""Task protocol for the training harness.

A task owns one name and one ``run`` method. Phase 1 tasks are wrappers around
the existing trainers. They do not own a new training loop.
"""

from typing import Protocol, runtime_checkable

from ..shared.schemas.base import BaseConfig
from .result import FitResult


@runtime_checkable
class Task(Protocol):
    """Plugin interface registered by name."""

    name: str

    def run(self, config: BaseConfig, debug: bool = False) -> FitResult:
        """Train with an existing trainer and return the best checkpoint."""
        ...
