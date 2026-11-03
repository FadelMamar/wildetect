"""Detection wrapper around the existing Ultralytics trainer."""

from pathlib import Path

from ...trainers.detection_trainer import UltralyticsDetectionTrainer
from ..config import ExperimentConfig
from ..result import FitResult


def _detection_weights(config) -> str | None:
    from ..config import DetectionExperiment

    if isinstance(config, DetectionExperiment):
        return config.model.weights
    return config.weights


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
            debug = debug or config.debug
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

    def evaluate(self, config, weights: str | None = None, debug: bool = False):
        from ...evaluators.ultralytics import UltralyticsEvaluator
        from ...shared.schemas.detection import (
            DetectionEvalConfig,
            DetectionEvalParamsConfig,
            DetectionMetricsConfig,
            DetectionWeightsConfig,
        )
        from ...shared.schemas.yolo import YoloDatasetConfig
        from ..config import DetectionExperiment
        from ..ops import resolve_checkpoint

        if config.task != self.name:
            raise ValueError(f"{self.name} task received a {config.task} document")
        localizer = resolve_checkpoint(config, weights or _detection_weights(config))
        from pathlib import Path as _Path
        eval_block = config.eval
        if isinstance(config, DetectionExperiment):
            data_cfg = config.data.data_cfg
            root = config.data.root
            discard = config.data.discard
            keep = config.data.keep
            force_merge = config.data.force_merge
            single_cls = (
                eval_block.single_cls
                if eval_block and eval_block.single_cls is not None
                else config.train.single_cls
            )
            imgsz = None
            if eval_block and eval_block.imgsz:
                imgsz = eval_block.imgsz
            imgsz = imgsz or config.train.imgsz
            classifier = eval_block.classifier if eval_block else None
        else:
            data_cfg = config.data.data_cfg
            root = config.data.root
            discard = config.data.discard
            keep = config.data.keep
            force_merge = config.data.force_merge
            single_cls = eval_block.single_cls
            if single_cls is None:
                single_cls = True
            imgsz = eval_block.imgsz or config.data.input_size
            classifier = eval_block.classifier
        if root is None and data_cfg:
            root = str(_Path(data_cfg).parent)
        split = eval_block.split if eval_block else "val"
        device = eval_block.device if eval_block else "cpu"
        batch_size = eval_block.batch_size if eval_block else 8
        iou = 0.5 if not eval_block or eval_block.iou is None else eval_block.iou
        conf = 0.1 if not eval_block or eval_block.conf is None else eval_block.conf
        results_dir = eval_block.results_dir if eval_block else "results/eval"
        class_agnostic = False if not eval_block else eval_block.class_agnostic
        eval_config = DetectionEvalConfig(
            weights=DetectionWeightsConfig(localizer=localizer, classifier=classifier),
            dataset=YoloDatasetConfig(
                data_cfg=data_cfg,
                root_data_directory=root,
                force_merge=force_merge,
                keep_classes=keep,
                discard_classes=discard,
            ),
            device=device,
            metrics=DetectionMetricsConfig(
                average="macro", class_agnostic=class_agnostic
            ),
            eval=DetectionEvalParamsConfig(
                imgsz=imgsz,
                split=split,
                iou=iou,
                single_cls=single_cls,
                batch_size=batch_size,
                conf=conf,
            ),
            results_dir=results_dir,
        )
        report = UltralyticsEvaluator(config=eval_config).evaluate(debug=debug)
        names = eval_block.metrics if eval_block else []
        if not names:
            return report
        return {key: value for key, value in report.items() if key in names}

    def visualize(self, config) -> None:
        from ...models.detector import Detector
        from ...shared.schemas.yolo import YoloInferenceConfig
        from ...visualization import Visualizer

        if config.task != self.name:
            raise ValueError(f"{self.name} task received a {config.task} document")
        if config.mlflow_name and config.mlflow_alias:
            detector = Detector.from_mlflow(
                name=config.mlflow_name,
                alias=config.mlflow_alias,
                mlflow_tracking_uri=config.mlflow_tracking_uri,
            )
        else:
            detector = Detector.from_config(
                localizer_config=YoloInferenceConfig(
                    weights=config.weights,
                    imgsz=config.imgsz,
                    device=config.device,
                ),
                classifier_ckpt=config.classifier_weights,
            )
        visualizer = Visualizer(
            fiftyone_dataset_name=config.dataset_name,
            label_studio_url=config.label_studio_url,
            label_studio_api_key=config.label_studio_api_key,
            label_studio_project_id=config.label_studio_project_id,
        )
        visualizer.add_predictions_from_detector(
            detector=detector,
            imgsz=config.imgsz,
            prediction_field=config.prediction_field,
            batch_size=config.batch_size,
            debug=config.debug,
            model_tag=config.model_tag,
        )

    def register(self, config) -> None:
        import tempfile

        import yaml

        from ...models.register import ModelRegistrar
        from ...shared.schemas.registration import DetectorRegistrationConfig

        if config.task != self.name:
            raise ValueError(f"{self.name} task received a {config.task} document")
        localizer = config.localizer
        # The localizer stays in export_format (pt keeps the .pt file).
        # GenericClassifier.export has no pt mode, so a two-stage model
        # uses torchscript unless classifier_export_format is set.
        classifier_format = config.classifier_export_format
        if classifier_format is None:
            classifier_format = (
                "torchscript"
                if config.classifier_weights and config.export_format == "pt"
                else config.export_format
            )
        document = {
            "localizer": {
                "yolo": {
                    "weights": localizer.weights,
                    "imgsz": localizer.imgsz,
                    "device": localizer.device,
                    "conf_thres": localizer.conf_thres,
                    "iou_thres": localizer.iou_thres,
                    "max_det": localizer.max_det,
                    "task": localizer.task,
                },
                "processing": {
                    "batch_size": config.batch,
                    "export_format": config.export_format,
                    "mlflow_tracking_uri": config.mlflow_tracking_uri,
                },
            },
            "classifier": {
                "weights": config.classifier_weights,
                "processing": {
                    "batch_size": config.batch,
                    "export_format": classifier_format,
                    "mlflow_tracking_uri": config.mlflow_tracking_uri,
                },
            },
            "processing": {
                "name": config.name,
                "batch_size": config.batch,
                "mlflow_tracking_uri": config.mlflow_tracking_uri,
                "export_format": config.export_format,
            },
        }
        validated = DetectorRegistrationConfig.model_validate(document)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(validated.model_dump(mode="json"), handle)
            path = Path(handle.name)
        try:
            ModelRegistrar(config.mlflow_tracking_uri).register_detector(
                path, alias=config.alias
            )
        finally:
            path.unlink(missing_ok=True)
