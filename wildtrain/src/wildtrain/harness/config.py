"""One experiment document for classification and detection.

Canonical YAML has shared ``tracking``, ``run``, and optional ``sweep`` blocks,
plus the task body (``classification`` or ``detection``). Those bodies are the
existing training configs, so trainers do not learn a second field layout.

Today's flat training YAML still loads. A file with no shared blocks is wrapped:
MLflow becomes ``tracking.mlflow``, and the run name, project, and output
directory are copied off the body. ``task`` may be omitted; it is inferred, or
taken from the caller (the ``train classifier`` / ``train detector`` adapters).

``trainer_config()`` copies the shared blocks back onto the body. Trainers keep
receiving ``ClassificationConfig`` or ``DetectionConfig``. On that copy,
``tracking.mlflow`` wins over a nested ``mlflow``, ``run.output_dir`` is the
classification checkpoint directory, and ``run.name`` / ``run.project`` are the
detection run name and Ultralytics project.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

from ..shared.schemas.base import BaseConfig
from ..shared.schemas.classification import ClassificationConfig
from ..shared.schemas.common import MLflowConfig
from ..shared.schemas.detection import DetectionConfig

TaskName = Literal["classification", "detection"]
_SHARED_KEYS = {"task", "tracking", "run", "sweep", "classification", "detection"}


class _TaskName(BaseModel):
    """Reject unknown task names with a pydantic validation error."""

    task: TaskName


class TrackingBlock(BaseConfig):
    """Shared experiment tracking. MLflow is the backend in use today."""

    mlflow: MLflowConfig = Field(description="MLflow run settings")


class RunBlock(BaseConfig):
    """Shared run identity and output location."""

    name: Optional[str] = Field(default=None, description="Run name")
    project: Optional[str] = Field(default=None, description="Project or experiment")
    output_dir: Optional[str] = Field(
        default=None, description="Checkpoint or run directory"
    )
    seed: int = Field(default=0, description="Recorded seed. Sweep uses sweep.seed.")
    debug: bool = Field(default=False, description="Short debug training pass")


class SweepBlock(BaseConfig):
    """Categorical search space shared by both tasks.

    ``parameters`` maps a dotted path to a list of choices. Paths are relative
    to the task body (``train.lr``, ``model.backbone``) unless they start with
    ``tracking``, ``run``, ``classification``, or ``detection``.
    """

    n_trials: int = Field(gt=0, description="Number of Optuna trials")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optuna study direction"
    )
    objective: Optional[str] = Field(
        default=None,
        description="Score name. f1_score and fitness use FitResult.best_score.",
    )
    parameters: Dict[str, List[Any]] = Field(
        default_factory=dict, description="Dotted path to categorical choices"
    )
    storage: Optional[str] = Field(
        default=None, description="Optuna storage URL. Unset keeps the study in memory."
    )
    seed: int = Field(default=0, description="Sampler seed")
    name: Optional[str] = Field(default=None, description="Study name")
    timeout: Optional[int] = Field(
        default=None, description="Optimization timeout in seconds"
    )


class ExperimentConfig(BaseConfig):
    """Training experiment discriminated by ``task``."""

    task: TaskName
    tracking: TrackingBlock
    run: RunBlock = Field(default_factory=RunBlock)
    sweep: Optional[SweepBlock] = None
    classification: Optional[ClassificationConfig] = None
    detection: Optional[DetectionConfig] = None

    @model_validator(mode="after")
    def _body_matches_task(self) -> "ExperimentConfig":
        if self.task == "classification":
            if self.classification is None:
                raise ValueError(
                    "classification experiments require a classification block"
                )
            if self.detection is not None:
                raise ValueError(
                    "classification experiments cannot include a detection block"
                )
        else:
            if self.detection is None:
                raise ValueError("detection experiments require a detection block")
            if self.classification is not None:
                raise ValueError(
                    "detection experiments cannot include a classification block"
                )
        return self

    @property
    def body(self) -> Union[ClassificationConfig, DetectionConfig]:
        """Task body the existing trainers already understand."""
        if self.task == "classification":
            assert self.classification is not None
            return self.classification
        assert self.detection is not None
        return self.detection

    @property
    def dataset(self):
        """Dataset block of the active task. Kept for existing callers."""
        return self.body.dataset

    @property
    def train(self):
        """Train block of the active task. Kept for existing callers."""
        return self.body.train

    def trainer_config(self) -> Union[ClassificationConfig, DetectionConfig]:
        """Copy shared tracking and run fields onto a task-body copy."""
        body = self.body.model_copy(deep=True)
        body.mlflow = self.tracking.mlflow.model_copy(deep=True)
        if self.task == "classification":
            if self.run.output_dir:
                body.checkpoint.dirpath = self.run.output_dir
            if self.run.name is not None:
                body.mlflow.run_name = self.run.name
            if self.run.project is not None:
                body.mlflow.experiment_name = self.run.project
        else:
            if self.run.name is not None:
                body.name = self.run.name
            if self.run.project is not None:
                body.project = self.run.project
        return body


def load_experiment(
    path: Path | str, task: str | None = None
) -> ExperimentConfig:
    """Load canonical or legacy experiment YAML.

    OmegaConf resolves interpolations before validation. ``task`` is used only
    when the file does not name one, which is how ``train classifier`` and
    ``train detector`` accept today's YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {config_path}")

    loaded = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment file must be a mapping: {config_path}")
    return experiment_from_mapping(payload, task=task)


def experiment_from_mapping(
    payload: Mapping[str, Any], task: str | None = None
) -> ExperimentConfig:
    """Build an experiment from a canonical document or a legacy flat config."""
    data = dict(payload)
    if _is_canonical(data):
        return _from_canonical(data, task)
    return _from_legacy(data, task)


def _is_canonical(payload: Mapping[str, Any]) -> bool:
    tracking = payload.get("tracking")
    if not isinstance(tracking, dict):
        return False
    return isinstance(payload.get("classification"), dict) or isinstance(
        payload.get("detection"), dict
    )


def _resolve_task(declared: Any, requested: str | None) -> str:
    if declared is not None and requested is not None and declared != requested:
        raise ValueError(
            f"Config task is {declared!r}, but the caller asked for {requested!r}"
        )
    task = declared if declared is not None else requested
    if task is None:
        raise ValueError(
            "task is required. Use classification or detection, "
            "or pass it from the train command."
        )
    _TaskName.model_validate({"task": task})
    return str(task)


def _infer_task(payload: Mapping[str, Any]) -> str:
    dataset = payload.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    model = payload.get("model")
    model = model if isinstance(model, dict) else {}
    train = payload.get("train")
    train = train if isinstance(train, dict) else {}
    if (
        "checkpoint" in payload
        or "label_policy" in payload
        or "label_policy" in dataset
        or "single_class" in dataset
        or dataset.get("dataset_type") in {"roi", "crop"}
        or "backbone" in model
    ):
        return "classification"
    if (
        "curriculum" in payload
        or "use_custom_yolo" in payload
        or "single_cls" in payload
        or "single_cls" in train
        or "architecture_file" in model
    ):
        return "detection"
    raise ValueError(
        "Cannot infer task. Set task: classification or task: detection."
    )


def _sweep_from(payload: Mapping[str, Any]) -> Optional[SweepBlock]:
    sweep = payload.get("sweep")
    if not sweep:
        return None
    return SweepBlock.model_validate(sweep)


def _run_from_classification(
    config: ClassificationConfig, mlflow: Optional[MLflowConfig] = None
) -> RunBlock:
    source = mlflow or config.mlflow
    return RunBlock(
        name=source.run_name,
        project=source.experiment_name,
        output_dir=config.checkpoint.dirpath,
    )


def _run_from_detection(config: DetectionConfig) -> RunBlock:
    return RunBlock(name=config.name, project=config.project, output_dir=config.project)


def _merge_run(explicit: Mapping[str, Any], fallback: RunBlock) -> RunBlock:
    payload = fallback.model_dump()
    for key, value in explicit.items():
        if value is not None:
            payload[key] = value
    return RunBlock.model_validate(payload)


def _from_legacy(payload: Mapping[str, Any], requested: str | None) -> ExperimentConfig:
    declared = payload.get("task")
    resolved = (
        _resolve_task(declared, requested)
        if declared is not None or requested is not None
        else _infer_task(payload)
    )
    _TaskName.model_validate({"task": resolved})
    body = {key: value for key, value in payload.items() if key not in _SHARED_KEYS}
    sweep = _sweep_from(payload)
    if resolved == "classification":
        config = ClassificationConfig.model_validate(body)
        return ExperimentConfig(
            task="classification",
            tracking=TrackingBlock(mlflow=config.mlflow),
            run=_run_from_classification(config),
            sweep=sweep,
            classification=config,
        )
    config = DetectionConfig.model_validate(body)
    return ExperimentConfig(
        task="detection",
        tracking=TrackingBlock(mlflow=config.mlflow),
        run=_run_from_detection(config),
        sweep=sweep,
        detection=config,
    )


def _named_task(payload: Mapping[str, Any], requested: str | None) -> str:
    declared = payload.get("task")
    has_classification = isinstance(payload.get("classification"), dict)
    has_detection = isinstance(payload.get("detection"), dict)
    if declared is None and has_classification and not has_detection:
        declared = "classification"
    elif declared is None and has_detection and not has_classification:
        declared = "detection"
    if has_classification and has_detection:
        raise ValueError(
            "An experiment can include only one of classification or detection"
        )
    return _resolve_task(declared, requested)


def _from_canonical(
    payload: Mapping[str, Any], requested: str | None
) -> ExperimentConfig:
    resolved = _named_task(payload, requested)
    if not isinstance(payload.get(resolved), dict):
        raise ValueError(f"{resolved} experiments require a {resolved} block")
    body_payload = dict(payload[resolved])
    tracking_payload = dict(payload.get("tracking") or {})
    run_payload = dict(payload.get("run") or {})
    mlflow_payload = tracking_payload.get("mlflow")
    if "mlflow" not in body_payload and isinstance(mlflow_payload, dict):
        body_payload["mlflow"] = mlflow_payload
    if resolved == "classification":
        return _canonical_classification(
            payload, body_payload, tracking_payload, run_payload
        )
    return _canonical_detection(payload, body_payload, tracking_payload, run_payload)


def _canonical_classification(payload, body_payload, tracking_payload, run_payload):
    checkpoint = dict(body_payload.get("checkpoint") or {})
    if "dirpath" not in checkpoint and run_payload.get("output_dir"):
        checkpoint["dirpath"] = run_payload["output_dir"]
        body_payload["checkpoint"] = checkpoint
    config = ClassificationConfig.model_validate(body_payload)
    tracking = _tracking(tracking_payload, config.mlflow)
    run = _merge_run(run_payload, _run_from_classification(config, tracking.mlflow))
    return ExperimentConfig(
        task="classification",
        tracking=tracking,
        run=run,
        sweep=_sweep_from(payload),
        classification=config,
    )


def _canonical_detection(payload, body_payload, tracking_payload, run_payload):
    if "name" not in body_payload and run_payload.get("name"):
        body_payload["name"] = run_payload["name"]
    if "project" not in body_payload:
        project = run_payload.get("project") or run_payload.get("output_dir")
        if project:
            body_payload["project"] = project
    config = DetectionConfig.model_validate(body_payload)
    tracking = _tracking(tracking_payload, config.mlflow)
    return ExperimentConfig(
        task="detection",
        tracking=tracking,
        run=_merge_run(run_payload, _run_from_detection(config)),
        sweep=_sweep_from(payload),
        detection=config,
    )


def _tracking(payload: Mapping[str, Any], fallback: MLflowConfig) -> TrackingBlock:
    mlflow_payload = payload.get("mlflow")
    if isinstance(mlflow_payload, dict):
        return TrackingBlock(mlflow=MLflowConfig.model_validate(mlflow_payload))
    return TrackingBlock(mlflow=fallback)
