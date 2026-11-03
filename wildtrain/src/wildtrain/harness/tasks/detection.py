"""Detection wrapper around the existing Ultralytics trainer."""

from ...trainers.detection_trainer import UltralyticsDetectionTrainer
from ..config import ExperimentConfig
from ..result import FitResult


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


class DetectionTask:
    """Run ``UltralyticsDetectionTrainer`` and report its best checkpoint.

    The trainer stores Ultralytics fitness on ``best_fitness`` and leaves
    ``best_model_score`` unset. The fitness value is the score reported here.
    """

    name = "detection"

    def run(self, config, debug: bool = False) -> FitResult:
        trainer_config = config
        if isinstance(config, ExperimentConfig):
            if config.task != self.name:
                raise ValueError(
                    f"{self.name} task received a {config.task} experiment"
                )
            trainer_config = config.trainer_config()
            debug = debug or config.run.debug
        trainer = UltralyticsDetectionTrainer(trainer_config)
        trainer.run(debug=debug)
        score = trainer.best_model_score
        if score is None:
            score = trainer.best_fitness
        metrics = getattr(trainer, "metrics", None)
        return FitResult(
            best_path=trainer.best_model_path,
            best_score=_as_float(score),
            score_name="fitness",
            metrics=dict(metrics) if isinstance(metrics, dict) else None,
        )
