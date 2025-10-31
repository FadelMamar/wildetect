"""Request models for the WildTrain API."""

from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any
from pathlib import Path

from ...cli.models import (
    ClassificationConfig,
    DetectionConfig,
    ClassificationEvalConfig,
    DetectionEvalConfig,
    ClassificationPipelineConfig,
    DetectionPipelineConfig,
    ClassificationVisualizationConfig,
    DetectionVisualizationConfig
)
from .common import BaseRequest


class ClassificationTrainingRequest(BaseRequest):
    """Request model for classifier training."""
    config: Dict[str, Any] = Field(description="Training configuration")


class DetectionTrainingRequest(BaseRequest):
    """Request model for detector training."""
    config: Dict[str, Any] = Field(description="Training configuration")


class ClassificationEvalRequest(BaseRequest):
    """Request model for classifier evaluation."""
    config: ClassificationEvalConfig = Field(description="Evaluation configuration")
    debug: bool = Field(default=False, description="Debug mode")


class DetectionEvalRequest(BaseRequest):
    """Request model for detector evaluation."""
    config: DetectionEvalConfig = Field(description="Evaluation configuration")
    debug: bool = Field(default=False, description="Debug mode")
    model_type: str = Field(default="yolo", description="Model type")

class ClassificationPipelineRequest(BaseRequest):
    """Request model for classification pipeline."""
    config: ClassificationPipelineConfig = Field(description="Pipeline configuration")


class DetectionPipelineRequest(BaseRequest):
    """Request model for detection pipeline."""
    config: DetectionPipelineConfig = Field(description="Pipeline configuration")


class ClassificationVisualizationRequest(BaseRequest):
    """Request model for classifier visualization."""
    config: ClassificationVisualizationConfig = Field(description="Visualization configuration")


class DetectionVisualizationRequest(BaseRequest):
    """Request model for detector visualization."""
    config: DetectionVisualizationConfig = Field(description="Visualization configuration")


class DatasetStatsRequest(BaseRequest):
    """Request model for dataset statistics."""
    data_dir: Path = Field(description="Path to dataset directory")
    split: str = Field(default="train", description="Dataset split to analyze")
    output_file: Optional[Path] = Field(default=None, description="Output file path")


class ConfigValidationRequest(BaseRequest):
    """Request model for configuration validation."""
    config_path: Path = Field(description="Path to configuration file")
    config_type: str = Field(description="Configuration type")


class FileUploadRequest(BaseRequest):
    """Request model for file upload."""
    filename: str = Field(description="Name of the uploaded file")
    file_type: str = Field(description="Type of file (config, model, etc.)")


class JobStatusRequest(BaseRequest):
    """Request model for job status."""
    job_id: str = Field(description="Job identifier")


class JobCancelRequest(BaseRequest):
    """Request model for job cancellation."""
    job_id: str = Field(description="Job identifier to cancel")
