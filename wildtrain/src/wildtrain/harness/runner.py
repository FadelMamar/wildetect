"""Dispatch an experiment config through the task registry."""

from .config import ExperimentConfig
from .registry import TaskRegistry
from .result import FitResult
from .tasks import ClassificationTask, DetectionTask


def apply_experiment_seed(config: ExperimentConfig) -> None:
    """Seed the task before training or a sweep trial.

    Classification uses Lightning's ``seed_everything``. Detection sets the
    Python, NumPy, and Torch seeds here. Ultralytics receives the same value
    as ``train.seed`` from ``DetectionExperiment.trainer_config``.
    """
    seed = int(config.seed)
    if config.task == "classification":
        import lightning as L

        L.seed_everything(seed, workers=True)
        return
    from ..utils.misc import set_seed

    set_seed(seed)


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
    apply_experiment_seed(config)
    active = registry if registry is not None else default_registry()
    task = active.get(config.task)
    result = task.run(config, debug=debug or config.debug)
    if config.eval is not None and config.eval.enabled:
        report = task.evaluate(config, weights=result.best_path, debug=debug)
        merged = dict(result.metrics or {})
        merged["eval"] = report
        result.metrics = merged
    return result
