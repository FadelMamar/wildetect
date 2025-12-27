"""Visualization configuration models for FiftyOne and Label Studio integration."""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator

from .base import BaseConfig
from .common import MLflowConfig
from .yolo import YoloConfig


class LabelStudioConfig(BaseConfig):
    """Label Studio configuration."""
    url: str = Field(default="http://localhost:8080", description="Label Studio URL")
    api_key: str = Field(default=None, description="Label Studio API key")
    project_id: int = Field(default=1, description="Label Studio project ID")
    model_tag: str = Field(default="version-demo", description="Model tag")


class FiftyOneConfig(BaseConfig):
    """FiftyOne configuration."""
    dataset_name: Optional[str] = Field(default=None, description="FiftyOne dataset name")
    prediction_field: Optional[str] = Field(default=None, description="Prediction field name")


class DetectionVisualizationConfig(BaseConfig):
    """Visualization configuration."""
    fiftyone: FiftyOneConfig = Field(description="FiftyOne configuration")
    localizer: YoloConfig = Field(description="Localizer configuration")
    classifier_weights: Optional[str] = Field(default=None, description="Classifier weights path")
    batch_size: int = Field(gt=0, description="Processing batch size")
    debug: bool = Field(default=False, description="Debug mode")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    label_studio: LabelStudioConfig = Field(description="Label Studio configuration")


class ClassificationVisualizationConfig(BaseConfig):
    """Classification visualization configuration.
    """
    dataset_name: str = Field(description="Name of the FiftyOne dataset to use or create")
    weights: str = Field(description="str to the classifier checkpoint (.ckpt) file")
    prediction_field: str = Field(default="classification_predictions", description="Field name to store predictions in FiftyOne samples")
    batch_size: int = Field(default=32, description="Batch size for prediction inference")
    device: str = Field(default="cpu", description="Device to run inference on (e.g., 'cpu' or 'cuda')")
    debug: bool = Field(default=False, description="If set, only process a small number of samples for debugging")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    label_to_class_map: Optional[dict] = Field(default=None, description="Label to class map")
        
    @field_validator('weights')
    @classmethod
    def validate_checkpoint_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Classifier checkpoint does not exist: {v}")
        return v
      
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v
    
    @field_validator('dataset_name')
    @classmethod
    def validate_dataset_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()
    
    @field_validator('prediction_field')
    @classmethod
    def validate_prediction_field(cls, v):
        if not v or not v.strip():
            raise ValueError("Prediction field name cannot be empty")
        return v.strip()


# Update forward references
ClassificationVisualizationConfig.model_rebuild()
