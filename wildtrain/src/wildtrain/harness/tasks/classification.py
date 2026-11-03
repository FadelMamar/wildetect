"""Classification wrapper around the existing Lightning trainer."""

from ...trainers.classification_trainer import ClassifierTrainer
from ..config import ExperimentConfig
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

    def run(self, config, debug: bool = False) -> FitResult:
        trainer_config = config
        if isinstance(config, ExperimentConfig):
            if config.task != self.name:
                raise ValueError(
                    f"{self.name} task received a {config.task} experiment"
                )
            trainer_config = config.trainer_config()
            debug = debug or config.debug
        trainer = ClassifierTrainer(trainer_config)
        trainer.run(debug=debug)
        return FitResult(
            best_path=trainer.best_model_path,
            best_score=_as_float(trainer.best_model_score),
            score_name="val_f1score",
        )
