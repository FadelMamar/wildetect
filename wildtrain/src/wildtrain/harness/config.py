"""Experiment document for classification and detection.

The YAML is the layout in the harness assessment: ``task``, ``name``, ``seed``,
``tracking``, ``data``, ``model``, ``train``, and optional ``checkpoint``,
``eval``, and ``sweep``. Trainers still receive ``ClassificationConfig`` or
``DetectionConfig``. ``trainer_config()`` is that mapping.

Flat training YAML and ``base_config`` sweep files are rejected with a message
that names the replacement fields.
"""

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from omegaconf import OmegaConf
from pydantic import Field, TypeAdapter, model_validator

from ..shared.schemas.base import BaseConfig
from ..shared.schemas.classification import (
    ClassificationConfig,
    ClassificationDatasetConfig,
)
from ..shared.schemas.common import (
    CurriculumConfig,
    DatasetStatsConfig,
    MLflowConfig,
    SingleClassConfig,
    TransformsConfig,
)
from ..shared.schemas.detection import DetectionConfig
from ..shared.schemas.yolo import (
    YoloCurriculumConfig,
    YoloCustomConfig,
    YoloDatasetConfig,
    YoloModelConfig,
    YoloPretrainingConfig,
    YoloTrainConfig,
)

TaskName = Literal["classification", "detection"]

_LEGACY_TRAINING = (
    "This file is the old flat training config and is no longer accepted. "
    "Use an experiment document with wildtrain run -c experiment.yaml. "
    "Replace dataset.root_data_directory with data.root, "
    "dataset.dataset_type with data.format (roi or yolo), "
    "dataset.single_class.single_class_name with data.label_policy.positive_name, "
    "dataset.single_class.background_class_name with "
    "data.label_policy.background_name, "
    "dataset.single_class.discard_classes with data.label_policy.discard, "
    "train.batch with train.batch_size, "
    "use_custom_yolo: true with model.backend: custom_yolo, "
    "and top-level mlflow with tracking.mlflow. "
    "wildtrain train classifier, wildtrain train detector, "
    "POST /training/classifier, and POST /training/detector have been removed."
)

_LEGACY_SWEEP = (
    "This file is the old sweep config (base_config and parameters) and is no "
    "longer accepted. Put the experiment in one file and add a sweep block "
    "whose parameters are dotted paths, for example train.lr0 or model.backbone. "
    "A detection objective must match an Ultralytics metrics key exactly, such as "
    "metrics/mAP50(B), metrics/mAP50-95(B), metrics/precision(B), "
    "metrics/recall(B), or fitness. "
    "Run it with wildtrain sweep -c experiment.yaml."
)

_LEGACY_NESTED = (
    "Nested classification: or detection: blocks are not the experiment schema. "
    "Use top-level data, model, and train. data.format is roi or yolo. "
    "data.label_policy.positive_name replaces single_class.single_class_name. "
    "Run it with wildtrain run -c experiment.yaml."
)


class DvcBlock(BaseConfig):
    """Recorded DVC flag. The shared tracker is not wired in this slice."""

    enabled: bool = Field(default=False, description="Request dataset tracking")


class TrackingBlock(BaseConfig):
    """Shared tracking header."""

    mlflow: MLflowConfig = Field(description="MLflow run settings")
    dvc: DvcBlock = Field(default_factory=DvcBlock)


class SweepBlock(BaseConfig):
    """Categorical search space. Paths are fields on this experiment document.

    Detection ``objective`` is looked up verbatim on the Ultralytics metrics
    dict. Common keys are ``metrics/precision(B)``, ``metrics/recall(B)``,
    ``metrics/mAP50(B)``, ``metrics/mAP50-95(B)``, and ``fitness``.
    ``fitness`` and classification ``f1_score`` use ``FitResult.best_score``.
    Names such as ``map_50`` are not translated.
    """

    n_trials: int = Field(gt=0, description="Number of Optuna trials")
    direction: Literal["maximize", "minimize"] = Field(default="maximize")
    objective: Optional[str] = Field(
        default=None,
        description="Exact score name. Detection must match an Ultralytics key.",
    )
    parameters: Dict[str, List[Any]] = Field(default_factory=dict)
    storage: Optional[str] = Field(
        default=None, description="Optuna storage URL. Unset keeps the study in memory."
    )
    seed: int = Field(default=0, description="Sampler seed")
    name: Optional[str] = Field(default=None, description="Study name")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds")


class LabelPolicyConfig(BaseConfig):
    """How ROI labels are mapped. Omitted policies stay identity."""

    mode: Literal["collapse", "identity"] = Field(
        default="identity",
        description="collapse maps every foreground class to positive_name",
    )
    positive_name: str = Field(default="wildlife", description="Collapsed class name")
    background_name: str = Field(default="background")
    discard: Optional[List[str]] = Field(default=None, description="Classes to drop")
    keep: Optional[List[str]] = Field(default=None, description="Classes to keep")


class EvalBlock(BaseConfig):
    """Evaluation settings.

    ``enabled`` makes ``wildtrain run`` evaluate after training. ``wildtrain
    evaluate -c`` always evaluates. ``metrics`` filters the report when set.
    Detection metrics must be the names produced by the supervision report.
    """

    enabled: bool = False
    split: Literal["train", "val", "test"] = "val"
    metrics: List[str] = Field(default_factory=list)
    device: str = "cpu"
    batch_size: int = Field(default=8, gt=0)
    imgsz: Optional[int] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    single_cls: Optional[bool] = None
    class_agnostic: bool = False
    results_dir: str = "results/eval"
    classifier: Optional[str] = None


class ClassificationDataConfig(BaseConfig):
    """Classification samples. ``format: roi`` is the product dataset."""

    format: Literal["roi", "crop"] = "roi"
    root: str = Field(description="Dataset root. Checked when training starts.")
    batch_size: int = Field(default=64, gt=0)
    num_workers: int = Field(default=0, ge=0)
    input_size: int = Field(default=384, gt=0)
    label_policy: LabelPolicyConfig = Field(default_factory=LabelPolicyConfig)
    stats: DatasetStatsConfig
    transforms: TransformsConfig = Field(default_factory=TransformsConfig)
    rebalance: bool = False
    curriculum: Optional[CurriculumConfig] = None
    crop_size: Optional[int] = None
    max_tn_crops: Optional[int] = None
    p_draw_annotations: Optional[float] = None
    compute_difficulties: Optional[bool] = None
    preserve_aspect_ratio: Optional[bool] = None


class ClassificationModelConfig(BaseConfig):
    """timm classifier."""

    backend: Literal["timm"] = "timm"
    backbone: str
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    freeze_backbone: bool = False
    weights: Optional[str] = None
    hidden_dim: int = Field(default=128, gt=0)
    num_layers: int = Field(default=2, gt=0)


class ClassificationTrainConfig(BaseConfig):
    """Shared training header plus the Lightning fields the trainer reads."""

    epochs: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    accelerator: str = "auto"
    precision: str = "32"
    lrf: float = Field(default=0.1, gt=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    num_workers: Optional[int] = Field(default=None, ge=0)
    val_check_interval: int = Field(default=1, ge=1)


class CheckpointConfig(BaseConfig):
    """Lightning checkpoint. ``dirpath`` is the run output directory."""

    monitor: str = "val_f1score"
    mode: Literal["min", "max"] = "max"
    patience: int = Field(default=10, gt=0)
    dirpath: str = "checkpoints"
    save_top_k: int = Field(default=1, ge=0)
    save_last: bool = True
    save_weights_only: bool = True
    filename: str = "best"
    min_delta: float = Field(default=0.001, ge=0.0)


class DetectionDataConfig(BaseConfig):
    """YOLO data pointer. The data yaml is read when training starts."""

    format: Literal["yolo"] = "yolo"
    data_cfg: Optional[str] = None
    root: Optional[str] = None
    force_merge: bool = False
    discard: Optional[List[str]] = None
    keep: Optional[List[str]] = None


class DetectionModelConfig(BaseConfig):
    """Ultralytics localizer, or the custom YOLO fork when selected."""

    backend: Literal["ultralytics", "custom_yolo"] = "ultralytics"
    weights: Optional[str] = None
    architecture: Optional[str] = None
    pretrained: bool = True
    custom: YoloCustomConfig = Field(default_factory=YoloCustomConfig)


class DetectionTrainConfig(BaseConfig):
    """Detection loop. ``batch_size`` is the public name of Ultralytics ``batch``."""

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    imgsz: int = Field(gt=0)
    lr0: float = Field(gt=0.0)
    lrf: float = Field(default=0.1, gt=0.0)
    single_cls: bool = True
    optimizer: str = "AdamW"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 1
    cos_lr: bool = True
    patience: int = 10
    iou: float = 0.65
    box: float = 3.5
    cls: float = 1.0
    dfl: float = 1.5
    device: str = "cpu"
    workers: int = 0
    degrees: float = 45.0
    mixup: float = 0.0
    cutmix: float = 0.5
    shear: float = 10.0
    copy_paste: float = 0.0
    erasing: float = 0.0
    scale: float = 0.2
    fliplr: float = 0.5
    flipud: float = 0.5
    hsv_h: float = 0.0
    hsv_s: float = 0.1
    hsv_v: float = 0.1
    translate: float = 0.2
    mosaic: float = 0.0
    multi_scale: bool = False
    perspective: float = 0.0
    deterministic: bool = False
    freeze: int = 9
    cache: bool = False


class ExperimentConfig(BaseConfig):
    """Shared header. The loaded object is a task subclass."""

    name: str
    seed: int = 0
    tracking: TrackingBlock
    sweep: Optional[SweepBlock] = None
    eval: Optional[EvalBlock] = None
    debug: bool = False

    def trainer_config(self):
        """Build the config object the existing trainer accepts."""
        raise NotImplementedError

    def _mlflow(self) -> MLflowConfig:
        mlflow = self.tracking.mlflow.model_copy(deep=True)
        if not mlflow.run_name:
            mlflow.run_name = self.name
        if not mlflow.experiment_name:
            mlflow.experiment_name = self.name
        return mlflow


class ClassificationExperiment(ExperimentConfig):
    """ROI (or crop) classifier experiment."""

    task: Literal["classification"]
    data: ClassificationDataConfig
    model: ClassificationModelConfig
    train: ClassificationTrainConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    def trainer_config(self) -> ClassificationConfig:
        policy = self.data.label_policy
        mapped_policy = SingleClassConfig(
            mode=policy.mode,
            enable=policy.mode == "collapse",
            background_class_name=policy.background_name,
            single_class_name=policy.positive_name,
            keep_classes=policy.keep,
            discard_classes=policy.discard,
        )
        dataset_payload: Dict[str, Any] = {
            "root_data_directory": self.data.root,
            "dataset_type": self.data.format,
            "input_size": self.data.input_size,
            "batch_size": self.data.batch_size,
            "num_workers": self.data.num_workers,
            "stats": self.data.stats,
            "rebalance": self.data.rebalance,
            "label_policy": mapped_policy,
            "single_class": mapped_policy,
        }
        transforms = _transforms_or_none(self.data.transforms)
        if transforms is not None:
            dataset_payload["transforms"] = transforms
        if self.data.curriculum is not None:
            dataset_payload["curriculum_config"] = self.data.curriculum
        for field in (
            "crop_size",
            "max_tn_crops",
            "p_draw_annotations",
            "compute_difficulties",
            "preserve_aspect_ratio",
        ):
            value = getattr(self.data, field)
            if value is not None:
                dataset_payload[field] = value
        workers = self.data.num_workers
        if self.train.num_workers is not None:
            workers = self.train.num_workers
        batch_size = self.data.batch_size
        if self.train.batch_size is not None:
            batch_size = self.train.batch_size
        return ClassificationConfig(
            dataset=ClassificationDatasetConfig.model_validate(dataset_payload),
            model={
                "backbone": self.model.backbone,
                "pretrained": self.model.pretrained,
                "backbone_source": self.model.backend,
                "dropout": self.model.dropout,
                "freeze_backbone": self.model.freeze_backbone,
                "input_size": self.data.input_size,
                "mean": list(self.data.stats.mean),
                "std": list(self.data.stats.std),
                "weights": self.model.weights,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
            },
            train={
                "batch_size": batch_size,
                "epochs": self.train.epochs,
                "lr": self.train.lr,
                "label_smoothing": self.train.label_smoothing,
                "weight_decay": self.train.weight_decay,
                "lrf": self.train.lrf,
                "precision": self.train.precision,
                "accelerator": self.train.accelerator,
                "num_workers": workers,
                "val_check_interval": self.train.val_check_interval,
            },
            checkpoint=self.checkpoint.model_dump(),
            mlflow=self._mlflow(),
        )


class DetectionExperiment(ExperimentConfig):
    """Class-agnostic localizer by default. ``single_cls: false`` stays optional."""

    task: Literal["detection"]
    project: str = "wildtrain"
    data: DetectionDataConfig
    model: DetectionModelConfig = Field(default_factory=DetectionModelConfig)
    train: DetectionTrainConfig
    curriculum: YoloCurriculumConfig = Field(default_factory=YoloCurriculumConfig)
    pretraining: YoloPretrainingConfig = Field(default_factory=YoloPretrainingConfig)

    @model_validator(mode="after")
    def _yolo_source(self) -> "DetectionExperiment":
        if not self.data.data_cfg and not self.data.root:
            raise ValueError("detection data needs data_cfg or root")
        return self

    def trainer_config(self) -> DetectionConfig:
        train_payload = self.train.model_dump()
        train_payload["batch"] = train_payload.pop("batch_size")
        train_payload["seed"] = self.seed
        return DetectionConfig(
            dataset=YoloDatasetConfig(
                data_cfg=self.data.data_cfg,
                root_data_directory=self.data.root,
                force_merge=self.data.force_merge,
                keep_classes=self.data.keep,
                discard_classes=self.data.discard,
            ),
            curriculum=self.curriculum,
            pretraining=self.pretraining,
            model=YoloModelConfig(
                pretrained=self.model.pretrained,
                weights=self.model.weights,
                architecture_file=self.model.architecture,
            ),
            name=self.name,
            project=self.project,
            mlflow=self._mlflow(),
            use_custom_yolo=self.model.backend == "custom_yolo",
            custom_yolo_kwargs=self.model.custom,
            train=YoloTrainConfig.model_validate(train_payload),
        )


AnyExperiment = Annotated[
    Union[ClassificationExperiment, DetectionExperiment],
    Field(discriminator="task"),
]
_EXPERIMENT_ADAPTER = TypeAdapter(AnyExperiment)


def load_experiment(path: Path | str) -> ClassificationExperiment | DetectionExperiment:
    """Load an experiment YAML. Old training and sweep files raise ``ValueError``."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {config_path}")
    loaded = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment file must be a mapping: {config_path}")
    return experiment_from_mapping(payload)


def experiment_from_mapping(
    payload: Mapping[str, Any],
) -> ClassificationExperiment | DetectionExperiment:
    """Validate one experiment document."""
    data = dict(payload)
    legacy = _legacy_message(data)
    if legacy is not None:
        raise ValueError(legacy)
    return _EXPERIMENT_ADAPTER.validate_python(data)


def _transforms_or_none(transforms: TransformsConfig):
    """Empty transform lists mean the datamodule keeps its own default."""
    if not transforms.train and not transforms.val:
        return None
    return transforms


def _legacy_message(payload: Mapping[str, Any]) -> Optional[str]:
    if "base_config" in payload and "parameters" in payload:
        return _LEGACY_SWEEP
    if isinstance(payload.get("classification"), dict) or isinstance(
        payload.get("detection"), dict
    ):
        return _LEGACY_NESTED
    dataset = payload.get("dataset")
    if isinstance(dataset, dict) and _old_dataset(dataset):
        return _LEGACY_TRAINING
    if "use_custom_yolo" in payload:
        return _LEGACY_TRAINING
    if "mlflow" in payload and "tracking" not in payload:
        return _LEGACY_TRAINING
    return None


def _old_dataset(dataset: Mapping[str, Any]) -> bool:
    old_keys = {
        "root_data_directory",
        "dataset_type",
        "data_cfg",
        "single_class",
        "discard_classes",
        "keep_classes",
    }
    return any(key in dataset for key in old_keys)
