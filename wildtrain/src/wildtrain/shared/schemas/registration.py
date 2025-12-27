"""Model registration and inference configuration models."""

from typing import Optional
from pydantic import Field, field_validator

from .base import BaseConfig
from .yolo import YoloConfig


class RegistrationBase(BaseConfig):
    name: Optional[str] = Field(default=None, description="Model name for registration")
    batch_size: int = Field(default=8, gt=0, description="Batch size for inference")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    export_format: str = Field(default="pt", description="export format")
    dynamic:bool = Field(default=False, description="handle different sizes")

    @field_validator('export_format')
    @classmethod
    def validate_export_format(cls, v):
        valid_formats = ["torchscript", "openvino", "onnx", "pt"]
        if v not in valid_formats:
            raise ValueError(f"Export format must be one of {valid_formats}, got: {v}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v


class LocalizerRegistrationConfig(BaseConfig):
    """Configuration for registering a detection model to MLflow Model Registry.
    
    This configuration is specifically for registering localizer models.
    """
    yolo: Optional[YoloConfig] = Field(None,description="yolo config")
    processing: RegistrationBase = Field(description="processing information")


class ClassifierRegistrationConfig(BaseConfig):
    """Configuration for registering a classification model to MLflow Model Registry."""
    weights: Optional[str] = Field(default=None,description="Path to the model checkpoint file")
    processing: RegistrationBase = Field(description="processing information")
    

class DetectorRegistrationConfig(BaseConfig):
    """Base configuration for model registration.
    
    This is a union configuration that can handle both detector and classifier registration.
    """
    localizer: LocalizerRegistrationConfig = Field(description="Detector registration configuration")
    classifier: ClassifierRegistrationConfig = Field(description="Classifier registration configuration")
    processing: RegistrationBase = Field(description="processing information")


class InferenceConfig(BaseConfig):
    port: int = Field(default=4141, description="Port to run the server on")
    workers_per_device: int = Field(default=1, description="Number of workers per device")
    mlflow_registry_name: str = Field(default="detector", description="MLflow registry name")
    mlflow_alias: str = Field(default="demo", description="MLflow alias")
    mlflow_local_dir: str = Field(default="models-registry", description="MLflow local directory")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")


# Update forward references
LocalizerRegistrationConfig.model_rebuild()
ClassifierRegistrationConfig.model_rebuild()
DetectorRegistrationConfig.model_rebuild()
InferenceConfig.model_rebuild()
