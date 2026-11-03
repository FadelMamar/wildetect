"""Detection wrapper around the existing Ultralytics trainer."""

from ...shared.schemas.detection import DetectionConfig
from ...trainers.detection_trainer import UltralyticsDetectionTrainer
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

    def run(self, config: DetectionConfig, debug: bool = False) -> FitResult:
        trainer = UltralyticsDetectionTrainer(config)
        trainer.run(debug=debug)
        score = trainer.best_model_score
        if score is None:
            score = trainer.best_fitness
        return FitResult(
            best_path=trainer.best_model_path,
            best_score=_as_float(score),
            score_name="fitness",
        )
