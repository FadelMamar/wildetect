"""Calibration configuration schemas for inference hyperparameter optimization."""

from typing import List, Optional
from pydantic import Field, field_validator

from .base import BaseConfig, SweepObjectiveTypes, SweepDirectionTypes
from .sweep import SweepOutputConfig


class CalibrationParametersConfig(BaseConfig):
    """Parameters to calibrate (inference hyperparameters)."""
    conf_thres: List[float] = Field(
        description="List of confidence threshold values to search"
    )
    iou_thres: List[float] = Field(
        description="List of IoU threshold values to search"
    )
    max_det: Optional[List[int]] = Field(
        default=None,
        description="List of maximum detections values to search (optional)"
    )
    
    @field_validator('conf_thres')
    @classmethod
    def validate_conf_thres(cls, v):
        if not v or len(v) == 0:
            raise ValueError("conf_thres list cannot be empty")
        for value in v:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"conf_thres values must be between 0.0 and 1.0, got {value}")
        return v
    
    @field_validator('iou_thres')
    @classmethod
    def validate_iou_thres(cls, v):
        if not v or len(v) == 0:
            raise ValueError("iou_thres list cannot be empty")
        for value in v:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"iou_thres values must be between 0.0 and 1.0, got {value}")
        return v
    
    @field_validator('max_det')
    @classmethod
    def validate_max_det(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("max_det list cannot be empty if provided")
            for value in v:
                if value <= 0:
                    raise ValueError(f"max_det values must be positive, got {value}")
        return v


class CalibrationConfig(BaseConfig):
    """Main calibration configuration for inference hyperparameter optimization."""
    base_config: str = Field(
        description="Path to base DetectionEvalConfig YAML file"
    )
    parameters: CalibrationParametersConfig = Field(
        description="Hyperparameter search space for calibration"
    )
    calibration_name: str = Field(
        description="Name of the calibration experiment"
    )
    n_trials: int = Field(
        gt=0, 
        le=1000, 
        description="Number of optimization trials"
    )
    objective: SweepObjectiveTypes = Field(
        default=SweepObjectiveTypes.F1_SCORE,
        description="Objective metric to optimize"
    )
    direction: SweepDirectionTypes = Field(
        default=SweepDirectionTypes.MAXIMIZE,
        description="Direction to optimize (maximize or minimize)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Maximum time for optimization in seconds"
    )
    output: Optional[SweepOutputConfig] = Field(
        default=None,
        description="Output configuration (optional, uses defaults if not provided)"
    )
    
    @field_validator('calibration_name')
    @classmethod
    def validate_calibration_name(cls, v):
        if not v or not v.strip():
            raise ValueError("calibration_name cannot be empty")
        return v.strip()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CalibrationConfig":
        return super().from_yaml(yaml_path)
