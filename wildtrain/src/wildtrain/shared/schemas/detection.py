"""Detection training and evaluation configuration models."""

from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field, field_validator

from .base import BaseConfig
from .common import MLflowConfig
from .yolo import (
    YoloDatasetConfig,
    YoloCurriculumConfig,
    YoloPretrainingConfig,
    YoloModelConfig,
    YoloCustomConfig,
    YoloTrainConfig,
)


class DetectionConfig(BaseConfig):
    """Complete detection configuration matching the YAML structure."""
    dataset: YoloDatasetConfig = Field(description="Dataset configuration")
    curriculum: YoloCurriculumConfig = Field(description="Curriculum learning configuration")
    pretraining: YoloPretrainingConfig = Field(description="Pretraining configuration")
    model: YoloModelConfig = Field(description="Model configuration")
    name: str = Field(description="Run name")
    project: str = Field(description="Project name")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    use_custom_yolo: bool = Field(default=False, description="Use custom YOLO")
    custom_yolo_kwargs: YoloCustomConfig = Field(description="Custom YOLO configuration")
    train: YoloTrainConfig = Field(description="Training configuration")


class DetectionWeightsConfig(BaseConfig):
    """Weights configuration for detection evaluation."""
    localizer: str = Field(description="str to the localizer weights file")
    classifier: Optional[str] = Field(default=None, description="str to the classifier weights file (optional)")
    
    @field_validator('localizer')
    @classmethod
    def validate_localizer_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Localizer weights file does not exist: {v}")
        return v
    
    @field_validator('classifier')
    @classmethod
    def validate_classifier_exists(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"Classifier weights file does not exist: {v}")
        return v


class DetectionMetricsConfig(BaseConfig):
    """Metrics configuration for detection evaluation."""
    average: Literal["macro", "micro", "weighted"] = Field(default="macro", description="Averaging method for metrics")
    class_agnostic: bool = Field(default=False, description="Whether to use class-agnostic evaluation")


class DetectionEvalParamsConfig(BaseConfig):
    """Evaluation parameters for detection."""
    imgsz: int = Field(default=640, description="Image size for evaluation")
    split: Literal["train", "val", "test"] = Field(default="val", description="Split to evaluate on")
    iou: float = Field(default=0.6, description="IoU threshold for evaluation")
    single_cls: bool = Field(default=True, description="Treat dataset as single-class")
    half: bool = Field(default=False, description="Use half precision")
    batch_size: int = Field(default=8, description="Batch size for DataLoader")
    num_workers: int = Field(default=0, description="Number of DataLoader workers")
    rect: bool = Field(default=False, description="Use rectangular batches")
    stride: int = Field(default=32, description="Model stride")
    task: Literal["detect", "classify", "segment"] = Field(default="detect", description="Task type")
    classes: Optional[List[int]] = Field(default=None, description="Optionally restrict to specific class indices")
    cache: bool = Field(default=False, description="Use cache for images/labels")
    multi_modal: bool = Field(default=False, description="Not using multi-modal data")
    conf: float = Field(default=0.1, description="Confidence threshold for evaluation")
    max_det: int = Field(default=300, description="Maximum detections per image")
    verbose: bool = Field(default=False, description="Verbosity level")
    augment: bool = Field(default=False, description="Use Test Time Augmentation")
    
    @field_validator('imgsz')
    @classmethod
    def validate_imgsz(cls, v):
        if v <= 0:
            raise ValueError(f"Image size must be positive, got: {v}")
        return v
    
    @field_validator('iou')
    @classmethod
    def validate_iou(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"IoU must be between 0.0 and 1.0, got: {v}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v
    
    @field_validator('num_workers')
    @classmethod
    def validate_num_workers(cls, v):
        if v < 0:
            raise ValueError(f"Number of workers must be non-negative, got: {v}")
        return v
    
    @field_validator('stride')
    @classmethod
    def validate_stride(cls, v):
        if v <= 0:
            raise ValueError(f"Stride must be positive, got: {v}")
        return v
    
    @field_validator('conf')
    @classmethod
    def validate_conf(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {v}")
        return v
    
    @field_validator('max_det')
    @classmethod
    def validate_max_det(cls, v):
        if v <= 0:
            raise ValueError(f"Max detections must be positive, got: {v}")
        return v


class DetectionEvalConfig(BaseConfig):
    """Detection evaluation configuration.
    
    This configuration is specifically for evaluation workflows and has a different
    structure compared to training configurations.
    
    """
    weights: DetectionWeightsConfig = Field(description="Model weights configuration")
    dataset: YoloDatasetConfig = Field(description="Dataset configuration")
    device: str = Field(default="cpu", description="Device to run evaluation on")
    metrics: DetectionMetricsConfig = Field(description="Evaluation metrics configuration")
    eval: DetectionEvalParamsConfig = Field(description="Evaluation parameters")   
    results_dir: str = Field(description="Results directory for pipeline outputs")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        assert (v == "cpu") or ("cuda" in v), f"Device must be one of ['cpu', 'cuda'], got: {v}"
        return v


# Update forward references
DetectionEvalConfig.model_rebuild()
DetectionWeightsConfig.model_rebuild()
DetectionMetricsConfig.model_rebuild()
DetectionEvalParamsConfig.model_rebuild()
