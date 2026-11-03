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

    def evaluate(self, config: BaseConfig, weights: str | None = None, debug: bool = False):
        """Score a checkpoint with the shared metric factory."""
        ...

    def visualize(self, config: BaseConfig) -> None:
        """Send predictions to the task's visualizer."""
        ...

    def register(self, config: BaseConfig) -> None:
        """Register the task's model with the census metadata keys."""
        ...
