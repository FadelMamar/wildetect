"""Classification training and evaluation configuration models."""

from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field, field_validator

from .base import BaseConfig
from .common import (
    DatasetStatsConfig,
    TransformsConfig,
    CurriculumConfig,
    SingleClassConfig,
    MLflowConfig,
)


class ClassificationDatasetConfig(BaseConfig):
    """Dataset configuration."""
    root_data_directory: str = Field(description="Root data directory path")
    dataset_type: Literal["roi", "crop"] = Field(description="Dataset type")
    input_size: int = Field(gt=0, description="Input image size")
    batch_size: int = Field(gt=0, description="Batch size")
    num_workers: int = Field(default=0, ge=0, description="Number of workers")
    stats: DatasetStatsConfig = Field(description="Dataset statistics")
    transforms: TransformsConfig = Field(None,description="Data transforms")
    curriculum_config: Optional[CurriculumConfig] = Field(default=None, description="Curriculum configuration")
    single_class: Optional[SingleClassConfig] = Field(default=None, description="Single class configuration")
    rebalance: bool = Field(default=False, description="Enable dataset rebalancing")
    
    # Crop dataset specific fields
    crop_size: Optional[int] = Field(default=None, gt=0, description="Crop size for crop datasets")
    max_tn_crops: Optional[int] = Field(default=None, gt=0, description="Maximum true negative crops")
    p_draw_annotations: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Probability of drawing annotations")
    compute_difficulties: Optional[bool] = Field(default=None, description="Compute crop difficulties")
    preserve_aspect_ratio: Optional[bool] = Field(default=None, description="Preserve aspect ratio")
    
    @field_validator('root_data_directory')
    @classmethod
    def validate_data_directory(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Data directory does not exist: {v}")
        return v
    
    @field_validator('crop_size', 'max_tn_crops', 'p_draw_annotations', 'compute_difficulties', 'preserve_aspect_ratio')
    @classmethod
    def validate_crop_fields(cls, v, info):
        if hasattr(info.data, 'dataset_type') and info.data.dataset_type == 'crop':
            if v is None:
                raise ValueError(f"Field is required for crop dataset type")
        return v


class ClassifierModelConfig(BaseConfig):
    """Model configuration."""
    backbone: str = Field(description="Model backbone")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    backbone_source: str = Field(default="timm", description="Backbone source")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate")
    freeze_backbone: bool = Field(default=False, description="Freeze backbone")
    input_size: int = Field(gt=0, description="Model input size")
    mean: List[float] = Field(description="Normalization mean values")
    std: List[float] = Field(description="Normalization std values")
    weights: Optional[str] = Field(default=None, description="Model weights path")
    hidden_dim: int = Field(default=128, gt=0, description="Hidden dimension")
    num_layers: int = Field(default=2, gt=0, description="Number of layers")
    
    @field_validator('mean', 'std')
    @classmethod
    def validate_normalization(cls, v):
        if len(v) != 3:
            raise ValueError("Normalization values must have exactly 3 values (RGB)")
        return v


class ClassifierTrainingConfig(BaseConfig):
    """Training configuration."""
    batch_size: int = Field(gt=0, description="Training batch size")
    epochs: int = Field(gt=0, description="Number of training epochs")
    lr: float = Field(gt=0.0, description="Learning rate")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0, description="Label smoothing")
    weight_decay: float = Field(default=0.0001, ge=0.0, description="Weight decay")
    lrf: float = Field(default=0.1, gt=0.0, description="Learning rate factor")
    precision: str = Field(default="32", description="Training precision")
    accelerator: str = Field(default="auto", description="Training accelerator")
    num_workers: int = Field(default=0, ge=0, description="Number of workers")
    val_check_interval: int = Field(default=1, ge=1, description="Validation check interval")


class ClassifierCheckpointConfig(BaseConfig):
    """Checkpoint configuration."""
    monitor: str = Field(description="Metric to monitor")
    save_top_k: int = Field(default=1, ge=0, description="Number of best models to save")
    mode: Literal["min", "max"] = Field(description="Monitor mode")
    save_last: bool = Field(default=True, description="Save last checkpoint")
    dirpath: str = Field(description="Checkpoint directory")
    patience: int = Field(default=10, gt=0, description="Early stopping patience")
    save_weights_only: bool = Field(default=True, description="Save only weights")
    filename: str = Field(description="Checkpoint filename pattern")
    min_delta: float = Field(default=0.001, ge=0.0, description="Minimum improvement delta")


class ClassificationConfig(BaseConfig):
    """Complete classification configuration."""
    dataset: ClassificationDatasetConfig = Field(description="Dataset configuration")
    model: ClassifierModelConfig = Field(description="Model configuration")
    train: ClassifierTrainingConfig = Field(description="Training configuration")
    checkpoint: ClassifierCheckpointConfig = Field(description="Checkpoint configuration")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    
    @field_validator('model')
    @classmethod
    def validate_model_dataset_compatibility(cls, v, info):
        if hasattr(info.data, 'dataset'):
            dataset = info.data.dataset
            if v.input_size != dataset.input_size:
                raise ValueError(f"Model input_size ({v.input_size}) must match dataset input_size ({dataset.input_size})")
        return v


class ClassificationEvalDatasetConfig(BaseConfig):
    """Dataset configuration for classification evaluation."""
    root_data_directory: str = Field(description="Root directory containing the dataset")
    single_class: Optional[SingleClassConfig] = Field(default=None, description="Single class configuration")
    
    @field_validator('root_data_directory')
    @classmethod
    def validate_data_directory_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Dataset directory does not exist: {v}")
        return v


class ClassificationEvalConfig(BaseConfig):
    """Classification evaluation configuration.
    
    This configuration is specifically for evaluation workflows and has a simpler
    structure compared to training configurations.
    
    """
    classifier: str = Field(description="str to the classifier checkpoint (.ckpt) file")
    split: Literal["train", "val", "test"] = Field(default="val", description="Dataset split to evaluate on")
    device: str = Field(default="cpu", description="Device to run evaluation on (cpu, cuda)")
    batch_size: int = Field(default=4, description="Batch size for evaluation")
    dataset: ClassificationEvalDatasetConfig = Field(description="Dataset configuration for evaluation")
    transforms: TransformsConfig = Field(None,description="Data transforms")

    @field_validator('classifier')
    @classmethod
    def validate_checkpoint_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Classifier checkpoint does not exist: {v}")
        return v
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        valid_devices = ["cpu", "cuda", "cuda:0", "cuda:1"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}, got: {v}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v


# Update forward references
ClassificationEvalConfig.model_rebuild()
ClassificationEvalDatasetConfig.model_rebuild()
