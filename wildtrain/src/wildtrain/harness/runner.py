"""Dispatch an experiment config through the task registry."""

from .config import ExperimentConfig
from .registry import TaskRegistry
from .result import FitResult
from .tasks import ClassificationTask, DetectionTask


def default_registry() -> TaskRegistry:
    """Registry that contains the two current training tasks."""
    registry = TaskRegistry()
    registry.register(ClassificationTask())
    registry.register(DetectionTask())
    return registry


def run_experiment(
    config: ExperimentConfig,
    debug: bool = False,
    registry: TaskRegistry | None = None,
) -> FitResult:
    """Run the task named by ``config.task``."""
    active = registry if registry is not None else default_registry()
    task = active.get(config.task)
    return task.run(config, debug=debug or config.debug)
