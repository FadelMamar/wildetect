"""Training harness skeleton.

Import concrete objects from the submodules. The package init stays light so
config and registry tests do not import the trainers.
"""

__all__ = [
    "ClassificationExperiment",
    "DetectionExperiment",
    "ExperimentConfig",
    "FitResult",
    "Task",
    "TaskRegistry",
    "default_registry",
    "load_experiment",
    "run_experiment",
]


def __getattr__(name: str):
    if name in {
        "ClassificationExperiment",
        "DetectionExperiment",
        "ExperimentConfig",
        "load_experiment",
    }:
        from . import config as config_module

        return getattr(config_module, name)
    if name == "FitResult":
        from .result import FitResult

        return FitResult
    if name == "Task":
        from .task import Task

        return Task
    if name == "TaskRegistry":
        from .registry import TaskRegistry

        return TaskRegistry
    if name in {"default_registry", "run_experiment"}:
        from . import runner as runner_module

        return getattr(runner_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
