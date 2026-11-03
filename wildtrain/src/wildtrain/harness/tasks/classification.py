"""Classification wrapper around the existing Lightning trainer."""

from ...shared.schemas.classification import ClassificationConfig
from ...trainers.classification_trainer import ClassifierTrainer
from ..result import FitResult


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


class ClassificationTask:
    """Run ``ClassifierTrainer`` and report its best checkpoint."""

    name = "classification"

    def run(self, config: ClassificationConfig, debug: bool = False) -> FitResult:
        trainer = ClassifierTrainer(config)
        trainer.run(debug=debug)
        return FitResult(
            best_path=trainer.best_model_path,
            best_score=_as_float(trainer.best_model_score),
            score_name="val_f1score",
        )
