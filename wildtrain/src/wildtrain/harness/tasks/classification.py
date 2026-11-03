"""Classification wrapper around the existing Lightning trainer."""

from ...trainers.classification_trainer import ClassifierTrainer
from ..config import ExperimentConfig
from ..result import FitResult


def _classification_eval_parts(config, weights: str | None):
    from ..config import ClassificationExperiment

    if isinstance(config, ClassificationExperiment):
        checkpoint = weights or config.model.weights
        if not checkpoint:
            raise ValueError("classification eval needs a checkpoint")
        policy = config.data.label_policy
        return (
            checkpoint,
            config.data.root,
            policy,
            config.eval.split if config.eval else "val",
            config.eval.device if config.eval else "cpu",
            config.eval.batch_size if config.eval else config.data.batch_size,
            _used_transforms(config.data.transforms),
            config.eval.metrics if config.eval else [],
        )
    checkpoint = weights or config.weights
    if not checkpoint:
        raise ValueError("classification eval needs weights")
    return (
        checkpoint,
        config.data.root,
        config.data.label_policy,
        config.eval.split,
        config.eval.device,
        config.eval.batch_size,
        config.data.transforms,
        config.eval.metrics,
    )


def _used_transforms(transforms):
    if transforms is None:
        return None
    if not getattr(transforms, "train", None) and not getattr(transforms, "val", None):
        return None
    return transforms


def _filter_metrics(report: dict, names: list) -> dict:
    if not names:
        return report
    selected = {}
    for key, value in report.items():
        if key in names or any(key.startswith(f"{name}_") for name in names):
            selected[key] = value
    return selected


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

    def evaluate(self, config, weights: str | None = None, debug: bool = False):
        from ...evaluators.classification import ClassificationEvaluator
        from ...shared.schemas.classification import (
            ClassificationEvalConfig,
            ClassificationEvalDatasetConfig,
        )
        from ...shared.schemas.common import SingleClassConfig
        if config.task != self.name:
            raise ValueError(f"{self.name} task received a {config.task} document")
        checkpoint, root, policy, split, device, batch_size, transforms, metrics = (
            _classification_eval_parts(config, weights)
        )
        mapped = SingleClassConfig(
            mode=policy.mode,
            enable=policy.mode == "collapse",
            background_class_name=policy.background_name,
            single_class_name=policy.positive_name,
            keep_classes=policy.keep,
            discard_classes=policy.discard,
        )
        payload = {
            "classifier": checkpoint,
            "split": split,
            "device": device,
            "batch_size": batch_size,
            "dataset": ClassificationEvalDatasetConfig(
                root_data_directory=root,
                label_policy=mapped,
                single_class=mapped,
            ),
        }
        if transforms is not None:
            payload["transforms"] = transforms
        eval_config = ClassificationEvalConfig(**payload)
        report = ClassificationEvaluator(config=eval_config).evaluate(debug=debug)
        return _filter_metrics(report, metrics)

    def visualize(self, config) -> None:
        from ...models.classifier import GenericClassifier
        from ...visualization import Visualizer

        if config.task != self.name:
            raise ValueError(f"{self.name} task received a {config.task} document")
        if config.mlflow_name and config.mlflow_alias:
            model = GenericClassifier.from_mlflow(
                name=config.mlflow_name,
                alias=config.mlflow_alias,
                mlflow_tracking_uri=config.mlflow_tracking_uri,
            )
        else:
            model = GenericClassifier.load_from_checkpoint(
                config.weights, map_location=config.device
            )
        visualizer = Visualizer(fiftyone_dataset_name=config.dataset_name)
        visualizer.add_predictions_from_classifier(
            model=model,
            prediction_field=config.prediction_field,
            batch_size=config.batch_size,
            debug=config.debug,
        )

    def register(self, config) -> None:
        from ...models.register import ModelRegistrar

        if config.task != self.name:
            raise ValueError(f"{self.name} task received a {config.task} document")
        registrar = ModelRegistrar(mlflow_tracking_uri=config.mlflow_tracking_uri)
        registrar.register_classifier(
            weights=config.weights,
            name=config.name,
            export_format=config.export_format,
            batch_size=config.batch,
        )
