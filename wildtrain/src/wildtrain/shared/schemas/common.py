"""Common/shared configuration models used across classification and detection."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import Field, field_validator

from .base import BaseConfig


class LoggingConfig(BaseConfig):
    """Logging configuration."""
    mlflow: bool = True
    log_dir: str = Field(default=str("./logs"), description="Logging directory")
    experiment_name: str = Field(default="wildtrain", description="MLflow experiment name")
    run_name: str = Field(default="default", description="MLflow run name")


class DatasetStatsConfig(BaseConfig):
    """Dataset statistics configuration."""
    mean: List[float] = Field(description="Mean values for normalization")
    std: List[float] = Field(description="Standard deviation values for normalization")
    
    @field_validator ('mean', 'std')
    @classmethod
    def validate_stats_length(cls, v):
        if len(v) != 3:
            raise ValueError("Mean and std must have exactly 3 values (RGB)")
        return v


class TransformConfig(BaseConfig):
    """Individual transform configuration."""
    name: str = Field(description="Transform name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Transform parameters")


class TransformsConfig(BaseConfig):
    """Transforms configuration for train and validation."""
    train: List[TransformConfig] = Field(default_factory=list, description="Training transforms")
    val: List[TransformConfig] = Field(default_factory=list, description="Validation transforms")


class CurriculumConfig(BaseConfig):
    """Curriculum learning configuration."""
    enabled: bool = Field(default=False, description="Enable curriculum learning")
    type: Literal["difficulty"] = Field(default="difficulty", description="Curriculum type")
    difficulty_strategy: Literal["linear"] = Field(default="linear", description="Difficulty strategy")
    start_difficulty: float = Field(default=0.0, ge=0.0, le=1.0, description="Starting difficulty")
    end_difficulty: float = Field(default=1.0, ge=0.0, le=1.0, description="Ending difficulty")
    warmup_epochs: int = Field(default=0, ge=0, description="Warmup epochs")
    log_frequency: int = Field(default=1, ge=1, description="Log frequency")
    
    @field_validator('end_difficulty')
    @classmethod
    def validate_end_difficulty(cls, v, info):
        if hasattr(info.data, 'start_difficulty') and v <= info.data.start_difficulty:
            raise ValueError("end_difficulty must be greater than start_difficulty")
        return v


class SingleClassConfig(BaseConfig):
    """Single class configuration for evaluation."""
    enable: bool = Field(description="Whether to enable single class mode")
    background_class_name: str = Field(description="Name of the background class")
    single_class_name: str = Field(description="Name of the single class")
    keep_classes: Optional[List[str]] = Field(default=None, description="Classes to keep (if None, all classes kept)")
    discard_classes: Optional[List[str]] = Field(default=None, description="Classes to discard")
    
    @field_validator('background_class_name', 'single_class_name')
    @classmethod
    def validate_class_names(cls, v):
        if not v or not v.strip():
            raise ValueError("Class name cannot be empty")
        return v.strip()


class MLflowConfig(BaseConfig):
    """MLflow configuration."""
    experiment_name: str = Field(default=None, description="MLflow experiment name")
    run_name: str = Field(default=None, description="MLflow run name")
    alias: str = Field(default=None, description="MLflow alias")
    name: str = Field(default=None, description="MLflow name")
    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    dwnd_location: Optional[str] = Field(default=None, description="DWND location")
    log_model: bool = Field(default=False, description="Log model")


# Update forward references
SingleClassConfig.model_rebuild()
