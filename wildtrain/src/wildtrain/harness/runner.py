"""Dispatch an experiment config to the registered task."""

from .config import ClassificationExperiment, DetectionExperiment
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
    config: ClassificationExperiment | DetectionExperiment,
    debug: bool = False,
    registry: TaskRegistry | None = None,
) -> FitResult:
    """Run the task named by ``config.task``."""
    active = registry if registry is not None else default_registry()
    task = active.get(config.task)
    return task.run(config, debug=debug)
