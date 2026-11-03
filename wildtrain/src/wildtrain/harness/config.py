"""Experiment envelope: today's training config plus a task name.

Phase 1 does not introduce a new field layout. ``task: classification`` carries
a :class:`ClassificationConfig` body, and ``task: detection`` carries a
:class:`DetectionConfig` body.
"""

from pathlib import Path
from typing import Annotated, Literal, Union

from omegaconf import OmegaConf
from pydantic import Field, TypeAdapter

from ..shared.schemas.classification import ClassificationConfig
from ..shared.schemas.detection import DetectionConfig


class ClassificationExperiment(ClassificationConfig):
    """Classification training document with a required task discriminator."""

    task: Literal["classification"]


class DetectionExperiment(DetectionConfig):
    """Detection training document with a required task discriminator."""

    task: Literal["detection"]


ExperimentConfig = Annotated[
    Union[ClassificationExperiment, DetectionExperiment],
    Field(discriminator="task"),
]

_EXPERIMENT_ADAPTER = TypeAdapter(ExperimentConfig)


def load_experiment(path: Path | str) -> ClassificationExperiment | DetectionExperiment:
    """Load an experiment YAML, resolving OmegaConf interpolations first."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {config_path}")

    loaded = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment file must be a mapping: {config_path}")
    return _EXPERIMENT_ADAPTER.validate_python(payload)
