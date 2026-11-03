"""Eval, visualization, and registration documents.

These share the experiment field names (``task``, ``data.format``,
``label_policy``). Old eval, visualization, and registration YAML is rejected
with the fields to change.
"""

from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from omegaconf import OmegaConf
from pydantic import Field, model_validator

from ..shared.schemas.base import BaseConfig
from .config import EvalBlock, LabelPolicyConfig, TrackingBlock

_LEGACY_EVAL = (
    "This file is the old evaluation config and is no longer accepted. "
    "Use task, weights, data.format, and eval. "
    "Classification: data.format: roi, data.root, data.label_policy.positive_name "
    "(was dataset.single_class.single_class_name), and weights "
    "(was classifier). Detection: data.format: yolo, data.data_cfg or data.root, "
    "data.discard (was dataset.discard_classes), and weights "
    "(was weights.localizer). Run wildtrain evaluate -c eval.yaml."
)

_LEGACY_VIZ = (
    "This file is the old visualization config and is no longer accepted. "
    "Use task, dataset_name, weights, and prediction_field. "
    "Detection imgsz replaces localizer.imgsz. "
    "Run wildtrain visualize -c visualize.yaml."
)

_LEGACY_REGISTER = (
    "This file is the old registration config and is no longer accepted. "
    "Use task, name, batch, imgsz, and mlflow_tracking_uri. "
    "A detector sets localizer.weights and optional classifier_weights. "
    "A classifier sets weights. Run wildtrain register -c register.yaml."
)


class EvalDataConfig(BaseConfig):
    """Dataset pointer for a standalone evaluation file."""

    format: Literal["roi", "crop", "yolo"]
    root: Optional[str] = None
    data_cfg: Optional[str] = None
    input_size: int = Field(default=384, gt=0)
    batch_size: int = Field(default=8, gt=0)
    num_workers: int = 0
    label_policy: LabelPolicyConfig = Field(default_factory=LabelPolicyConfig)
    transforms: Optional[Any] = None
    discard: Optional[list[str]] = None
    keep: Optional[list[str]] = None
    force_merge: bool = False


class EvalDocument(BaseConfig):
    """Standalone evaluation file. An experiment with ``train`` is also accepted."""

    task: Literal["classification", "detection"]
    weights: Optional[str] = None
    name: str = "eval"
    data: EvalDataConfig
    eval: EvalBlock = Field(default_factory=EvalBlock)
    tracking: Optional[TrackingBlock] = None

    @model_validator(mode="after")
    def _format_matches_task(self) -> "EvalDocument":
        if self.task == "classification" and self.data.format == "yolo":
            raise ValueError("classification eval data.format must be roi or crop")
        if self.task == "detection" and self.data.format != "yolo":
            raise ValueError("detection eval data.format must be yolo")
        if self.task == "detection" and not self.data.data_cfg and not self.data.root:
            raise ValueError("detection eval needs data.data_cfg or data.root")
        if self.task == "classification" and not self.data.root:
            raise ValueError("classification eval needs data.root")
        return self


class VisualizationDocument(BaseConfig):
    """FiftyOne prediction upload for one task."""

    task: Literal["classification", "detection"]
    dataset_name: str
    weights: Optional[str] = None
    prediction_field: str = "predictions"
    batch_size: int = Field(default=8, gt=0)
    device: str = "cpu"
    debug: bool = False
    imgsz: int = Field(default=640, gt=0)
    classifier_weights: Optional[str] = None
    mlflow_name: Optional[str] = None
    mlflow_alias: Optional[str] = None
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    label_studio_url: Optional[str] = None
    label_studio_api_key: Optional[str] = None
    label_studio_project_id: Optional[int] = None
    model_tag: str = "version-demo"


class LocalizerSpec(BaseConfig):
    """Localizer pointed at by a detector registration."""

    weights: str
    imgsz: int = Field(default=640, gt=0)
    device: str = "cpu"
    conf_thres: float = 0.2
    iou_thres: float = 0.3
    max_det: int = 300
    task: Literal["detect", "segment"] = "detect"


class RegistrationDocument(BaseConfig):
    """MLflow registration. Metadata keys stay the census set."""

    task: Literal["classification", "detection"]
    name: str
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    batch: int = Field(default=8, gt=0)
    imgsz: Optional[int] = None
    export_format: str = "pt"
    classifier_export_format: Optional[str] = None
    alias: str = "demo"
    weights: Optional[str] = None
    localizer: Optional[LocalizerSpec] = None
    classifier_weights: Optional[str] = None

    @model_validator(mode="after")
    def _weights_for_task(self) -> "RegistrationDocument":
        if self.task == "classification" and not self.weights:
            raise ValueError("classification registration needs weights")
        if self.task == "detection" and self.localizer is None:
            raise ValueError("detection registration needs localizer.weights")
        return self


def resolve_checkpoint(config, weights: str | None = None) -> str:
    """Checkpoint for evaluation. Never starts training.

    Uses the weights just produced by a train call, then ``model.weights`` or
    the eval document's ``weights``, then a checkpoint file that is already
    on disk. An empty experiment fails and tells the caller to train first
    or set the weights.
    """
    if weights:
        return str(weights)
    model = getattr(config, "model", None)
    model_weights = getattr(model, "weights", None) if model is not None else None
    if model_weights:
        return str(model_weights)
    document_weights = getattr(config, "weights", None)
    if document_weights:
        return str(document_weights)
    checkpoint = getattr(config, "checkpoint", None)
    if checkpoint is not None:
        directory = Path(checkpoint.dirpath)
        filename = str(getattr(checkpoint, "filename", "best"))
        for candidate in (directory / f"{filename}.ckpt", directory / filename):
            if candidate.is_file():
                return str(candidate)
    raise ValueError(
        "No checkpoint to evaluate. Train first with wildtrain run -c, "
        "or set model.weights. wildtrain evaluate does not train."
    )


def load_yaml(path: Path | str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a mapping: {config_path}")
    return payload


def load_eval_document(path: Path | str) -> EvalDocument:
    """Load a standalone eval file. Full experiments are not eval documents."""
    payload = load_yaml(path)
    message = _legacy_eval(payload)
    if message:
        raise ValueError(message)
    if "train" in payload:
        raise ValueError(
            "This is a training experiment. Evaluate it with wildtrain run "
            "when eval.enabled is true, or drop train and pass weights."
        )
    return EvalDocument.model_validate(payload)


def load_visualization_document(path: Path | str) -> VisualizationDocument:
    payload = load_yaml(path)
    message = _legacy_visualization(payload)
    if message:
        raise ValueError(message)
    return VisualizationDocument.model_validate(payload)


def load_registration_document(path: Path | str) -> RegistrationDocument:
    payload = load_yaml(path)
    message = _legacy_registration(payload)
    if message:
        raise ValueError(message)
    return RegistrationDocument.model_validate(payload)


def eval_from_mapping(payload: Mapping[str, Any]):
    """An experiment with train, or a standalone eval document."""
    message = _legacy_eval(payload)
    if message:
        raise ValueError(message)
    if "train" in payload and "data" in payload:
        from .config import experiment_from_mapping

        return experiment_from_mapping(payload)
    return EvalDocument.model_validate(payload)


def visualization_from_mapping(payload: Mapping[str, Any]) -> VisualizationDocument:
    message = _legacy_visualization(payload)
    if message:
        raise ValueError(message)
    return VisualizationDocument.model_validate(payload)


def validate_kind(path: Path | str, kind: str) -> str:
    """Validate experiment, eval, visualization, or registration."""
    from .config import _LEGACY_PIPELINE, load_experiment

    if kind == "pipeline":
        raise ValueError(_LEGACY_PIPELINE)
    loaders = {
        "experiment": load_experiment,
        "eval": load_eval_document,
        "visualization": load_visualization_document,
        "registration": load_registration_document,
    }
    if kind not in loaders:
        raise ValueError(
            f"Unknown config type {kind!r}. "
            "Use experiment, eval, visualization, or registration."
        )
    loaders[kind](path)
    return kind


def validate_harness_file(path: Path | str) -> str:
    """Validate an experiment, eval, visualization, or registration file.

    Returns the document kind.
    """
    from .config import _LEGACY_PIPELINE, _is_pipeline_payload, experiment_from_mapping

    payload = load_yaml(path)
    if _is_pipeline_payload(payload):
        raise ValueError(_LEGACY_PIPELINE)
    for checker in (_legacy_eval, _legacy_visualization, _legacy_registration):
        message = checker(payload)
        if message and _is_old_shape(payload):
            raise ValueError(message)
    if "train" in payload and "data" in payload:
        experiment_from_mapping(payload)
        return "experiment"
    if "dataset_name" in payload:
        VisualizationDocument.model_validate(payload)
        return "visualization"
    classifier_registration = (
        payload.get("task") == "classification"
        and "weights" in payload
        and "data" not in payload
    )
    if "localizer" in payload or classifier_registration:
        RegistrationDocument.model_validate(payload)
        return "registration"
    if "weights" in payload and "data" in payload:
        EvalDocument.model_validate(payload)
        return "eval"
    raise ValueError(
        "Unrecognized config. Expected an experiment (task, data, train), "
        "an eval file (task, weights, data, eval), a visualization file "
        "(task, dataset_name, weights), or a registration file "
        "(task, name, weights or localizer)."
    )


def _is_old_shape(payload: Mapping[str, Any]) -> bool:
    dataset = payload.get("dataset")
    if isinstance(dataset, dict) and (
        "root_data_directory" in dataset
        or "discard_classes" in dataset
        or "single_class" in dataset
    ):
        return True
    weights = payload.get("weights")
    if isinstance(weights, dict) and "localizer" in weights:
        return True
    if (
        "classifier" in payload
        and "dataset" in payload
        and "task" not in payload
    ):
        return True
    if "fiftyone" in payload or (
        "dataset_name" in payload and "task" not in payload
    ):
        return True
    localizer = payload.get("localizer")
    if isinstance(localizer, dict) and "yolo" in localizer:
        return True
    if "processing" in payload and "task" not in payload:
        return True
    return False


def _legacy_eval(payload: Mapping[str, Any]) -> Optional[str]:
    if not _is_old_shape(payload):
        return None
    if "classifier" in payload or (
        isinstance(payload.get("weights"), dict) and "localizer" in payload["weights"]
    ):
        return _LEGACY_EVAL
    dataset = payload.get("dataset")
    if (
        isinstance(dataset, dict)
        and "single_class" in dataset
        and "train" not in payload
    ):
        return _LEGACY_EVAL
    return None


def _legacy_visualization(payload: Mapping[str, Any]) -> Optional[str]:
    if "fiftyone" in payload or ("dataset_name" in payload and "task" not in payload):
        return _LEGACY_VIZ
    localizer = payload.get("localizer")
    if (
        isinstance(localizer, dict)
        and "imgsz" in localizer
        and "yolo" not in localizer
        and "task" not in payload
    ):
        return _LEGACY_VIZ
    return None


def _legacy_registration(payload: Mapping[str, Any]) -> Optional[str]:
    localizer = payload.get("localizer")
    if isinstance(localizer, dict) and "yolo" in localizer:
        return _LEGACY_REGISTER
    if "processing" in payload and "task" not in payload:
        return _LEGACY_REGISTER
    return None
