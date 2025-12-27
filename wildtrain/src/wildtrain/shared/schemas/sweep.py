"""Hyperparameter sweep configuration models for classification and detection."""

from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field, field_validator

from .base import BaseConfig, SweepObjectiveTypes, SweepDirectionTypes


class SweepOutputConfig(BaseConfig):
    """Output configuration for hyperparameter sweep results."""
    directory: Optional[str] = Field(default=None, description="Output directory for results (defaults to results/sweeps/sweep_name)")
    save_results: bool = Field(default=True, description="Save detailed results")
    save_plots: bool = Field(default=True, description="Generate visualization plots")
    format: Literal["json", "csv", "both"] = Field(default="json", description="Output format")
    include_optimization_history: bool = Field(default=True, description="Include all trials in output")


class ClassificationSweepModelParametersConfig(BaseConfig):
    """Model parameters for hyperparameter sweep."""
    backbone: List[str] = Field(description="List of backbone model names to search")
    dropout: List[float] = Field(description="List of dropout values to search")
    
    @field_validator('backbone')
    @classmethod
    def validate_backbone_non_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError("backbone list cannot be empty")
        return v
    
    @field_validator('dropout')
    @classmethod
    def validate_dropout(cls, v):
        if not v or len(v) == 0:
            raise ValueError("dropout list cannot be empty")
        for value in v:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"dropout values must be between 0.0 and 1.0, got {value}")
        return v


class ClassificationSweepTrainParametersConfig(BaseConfig):
    """Training parameters for hyperparameter sweep."""
    lr: List[float] = Field(description="List of learning rates to search")
    lrf: List[float] = Field(description="List of learning rate factors to search")
    label_smoothing: List[float] = Field(description="List of label smoothing values to search")
    weight_decay: List[float] = Field(description="List of weight decay values to search")
    batch_size: List[int] = Field(description="List of batch sizes to search")
    epochs: List[int] = Field(gt=0, description="Number of training epochs")
    
    @field_validator('lr')
    @classmethod
    def validate_lr(cls, v):
        if not v or len(v) == 0:
            raise ValueError("lr list cannot be empty")
        for value in v:
            if value <= 0.0:
                raise ValueError(f"lr values must be greater than 0.0, got {value}")
        return v
    
    @field_validator('lrf')
    @classmethod
    def validate_lrf(cls, v):
        if not v or len(v) == 0:
            raise ValueError("lrf list cannot be empty")
        for value in v:
            if value <= 0.0:
                raise ValueError(f"lrf values must be greater than 0.0, got {value}")
        return v
    
    @field_validator('label_smoothing')
    @classmethod
    def validate_label_smoothing(cls, v):
        if not v or len(v) == 0:
            raise ValueError("label_smoothing list cannot be empty")
        for value in v:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"label_smoothing values must be between 0.0 and 1.0, got {value}")
        return v
    
    @field_validator('weight_decay')
    @classmethod
    def validate_weight_decay(cls, v):
        if not v or len(v) == 0:
            raise ValueError("weight_decay list cannot be empty")
        for value in v:
            if value < 0.0:
                raise ValueError(f"weight_decay values must be non-negative, got {value}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if not v or len(v) == 0:
            raise ValueError("batch_size list cannot be empty")
        for value in v:
            if value <= 0:
                raise ValueError(f"batch_size values must be greater than 0, got {value}")
        return v


class ClassificationSweepParametersConfig(BaseConfig):
    """Parameters configuration for hyperparameter sweep."""
    model: ClassificationSweepModelParametersConfig = Field(description="Model parameters to search")
    train: ClassificationSweepTrainParametersConfig = Field(description="Training parameters to search")


class ClassificationSweepConfig(BaseConfig):
    """Hyperparameter sweep configuration."""
    base_config: str = Field(description="Path to base training configuration file")
    parameters: ClassificationSweepParametersConfig = Field(description="Hyperparameter search space")
    sweep_name: str = Field(description="Name of the sweep experiment")
    n_trials: int = Field(gt=0, le=1000, description="Number of optimization trials")
    seed: int = Field(description="Random seed for reproducibility")    
    timeout: Optional[int] = Field(default=None, description="Maximum time for optimization in seconds")
    output: Optional[SweepOutputConfig] = Field(default=None, description="Output configuration (optional, uses defaults if not provided)")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ClassificationSweepConfig":
        return super().from_yaml(yaml_path)
    
    @field_validator('base_config')
    @classmethod
    def validate_base_config_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Base config file does not exist: {v}")
        return v
    
    @field_validator('sweep_name')
    @classmethod
    def validate_sweep_name(cls, v):
        if not v or not v.strip():
            raise ValueError("sweep_name cannot be empty")
        return v.strip()


class DetectionSweepModelParametersConfig(BaseConfig):
    """Model parameters for detection hyperparameter sweep."""
    architecture_file: Optional[List[str]] = Field(default=None, description="List of architecture files to search")
    weights: Optional[List[str]] = Field(default=None, description="List of weight files to search")
    
    @field_validator('architecture_file', 'weights')
    @classmethod
    def validate_at_least_one(cls, v, info):
        # At least one of architecture_file or weights should be provided
        if info.field_name == 'architecture_file':
            # This is called for architecture_file, check if weights is also None
            return v
        return v
    
    @field_validator('architecture_file')
    @classmethod
    def validate_architecture_file(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("architecture_file list cannot be empty if provided")
        return v
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("weights list cannot be empty if provided")
        return v


class DetectionSweepTrainParametersConfig(BaseConfig):
    """Training parameters for detection hyperparameter sweep."""
    lr0: List[float] = Field(description="List of initial learning rates to search")
    lrf: List[float] = Field(description="List of learning rate factors to search")
    batch: List[int] = Field(description="List of batch sizes to search")
    epochs: List[int] = Field(description="List of epoch counts to search")
    imgsz: List[int] = Field(description="List of image sizes to search")
    optimizer: List[str] = Field(description="List of optimizers to search")
    weight_decay: List[float] = Field(description="List of weight decay values to search")
    box: Optional[List[float]] = Field(default=None, description="List of box loss weights to search")
    cls: Optional[List[float]] = Field(default=None, description="List of class loss weights to search")
    dfl: Optional[List[float]] = Field(default=None, description="List of DFL loss weights to search")
    
    @field_validator('lr0')
    @classmethod
    def validate_lr0(cls, v):
        if not v or len(v) == 0:
            raise ValueError("lr0 list cannot be empty")
        for value in v:
            if value <= 0.0:
                raise ValueError(f"lr0 values must be greater than 0.0, got {value}")
        return v
    
    @field_validator('lrf')
    @classmethod
    def validate_lrf(cls, v):
        if not v or len(v) == 0:
            raise ValueError("lrf list cannot be empty")
        for value in v:
            if value <= 0.0:
                raise ValueError(f"lrf values must be greater than 0.0, got {value}")
        return v
    
    @field_validator('batch')
    @classmethod
    def validate_batch(cls, v):
        if not v or len(v) == 0:
            raise ValueError("batch list cannot be empty")
        for value in v:
            if value <= 0:
                raise ValueError(f"batch values must be greater than 0, got {value}")
        return v
    
    @field_validator('epochs')
    @classmethod
    def validate_epochs(cls, v):
        if not v or len(v) == 0:
            raise ValueError("epochs list cannot be empty")
        for value in v:
            if value <= 0:
                raise ValueError(f"epochs values must be greater than 0, got {value}")
        return v
    
    @field_validator('imgsz')
    @classmethod
    def validate_imgsz(cls, v):
        if not v or len(v) == 0:
            raise ValueError("imgsz list cannot be empty")
        for value in v:
            if value <= 0:
                raise ValueError(f"imgsz values must be greater than 0, got {value}")
        return v
    
    @field_validator('optimizer')
    @classmethod
    def validate_optimizer(cls, v):
        if not v or len(v) == 0:
            raise ValueError("optimizer list cannot be empty")
        valid_optimizers = ["SGD", "Adam", "AdamW", "RMSprop"]
        for value in v:
            if value not in valid_optimizers:
                raise ValueError(f"optimizer must be one of {valid_optimizers}, got {value}")
        return v
    
    @field_validator('weight_decay')
    @classmethod
    def validate_weight_decay(cls, v):
        if not v or len(v) == 0:
            raise ValueError("weight_decay list cannot be empty")
        for value in v:
            if value < 0.0:
                raise ValueError(f"weight_decay values must be non-negative, got {value}")
        return v
    
    @field_validator('box', 'cls', 'dfl')
    @classmethod
    def validate_loss_weights(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("loss weight list cannot be empty if provided")
        if v is not None:
            for value in v:
                if value < 0.0:
                    raise ValueError(f"loss weight values must be non-negative, got {value}")
        return v


class DetectionSweepParametersConfig(BaseConfig):
    """Parameters configuration for detection hyperparameter sweep."""
    model: Optional[DetectionSweepModelParametersConfig] = Field(default=None, description="Model parameters to search")
    train: DetectionSweepTrainParametersConfig = Field(description="Training parameters to search")


class DetectionSweepConfig(BaseConfig):
    """Hyperparameter sweep configuration for detection models."""
    base_config: str = Field(description="Path to base training configuration file")
    parameters: DetectionSweepParametersConfig = Field(description="Hyperparameter search space")
    sweep_name: str = Field(description="Name of the sweep experiment")
    n_trials: int = Field(gt=0, le=1000, description="Number of optimization trials")
    objective: SweepObjectiveTypes = Field(default=SweepObjectiveTypes.MAP_50, description="Objective to optimize")
    direction: SweepDirectionTypes = Field(default=SweepDirectionTypes.MAXIMIZE, description="Direction to optimize")
    seed: int = Field(description="Random seed for reproducibility")    
    timeout: Optional[int] = Field(default=None, description="Maximum time for optimization in seconds")
    output: Optional[SweepOutputConfig] = Field(default=None, description="Output configuration (optional, uses defaults if not provided)")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DetectionSweepConfig":
        return super().from_yaml(yaml_path)
    
    @field_validator('base_config')
    @classmethod
    def validate_base_config_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Base config file does not exist: {v}")
        return v
    
    @field_validator('sweep_name')
    @classmethod
    def validate_sweep_name(cls, v):
        if not v or not v.strip():
            raise ValueError("sweep_name cannot be empty")
        return v.strip()
