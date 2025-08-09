"""
Pydantic configuration models for WildDetect.

This module provides Pydantic BaseModel classes for configuration validation,
type checking, and automatic serialization/deserialization.
It extends the existing dataclasses from config.py to avoid code duplication.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator

from .config import FlightSpecs, LoaderConfig, PredictionConfig

logger = logging.getLogger(__name__)


class FlightSpecsModel(BaseModel):
    """Flight specifications configuration model."""

    sensor_height: float = Field(
        default=24.0,
        gt=0,
        description="Sensor height in mm",
    )
    focal_length: float = Field(default=35.0, gt=0, description="Focal length in mm")
    flight_height: float = Field(
        default=180.0, gt=0, description="Flight height in meters"
    )

    @field_validator("sensor_height", "focal_length", "flight_height")
    @classmethod
    def validate_positive_floats(cls, v: float) -> float:
        """Validate that flight specs are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def to_flight_specs(self) -> FlightSpecs:
        """Convert to existing FlightSpecs dataclass."""
        return FlightSpecs(
            sensor_height=self.sensor_height,
            focal_length=self.focal_length,
            flight_height=self.flight_height,
        )


class ModelConfigModel(BaseModel):
    """Model configuration model."""

    path: Optional[str] = Field(default=None, description="Path to model weights")
    type: str = Field(default="yolo", description="Model type")
    confidence_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Confidence threshold"
    )
    device: str = Field(default="auto", description="Device to run inference on")
    batch_size: int = Field(default=32, gt=0, description="Batch size for inference")
    nms_iou: float = Field(default=0.5, ge=0.0, le=1.0, description="NMS IoU threshold")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and set device."""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class ProcessingConfigModel(BaseModel):
    """Processing configuration model."""

    tile_size: int = Field(default=800, description="Tile size for processing")  # type: ignore
    overlap_ratio: float = Field(default=0.2, description="Tile overlap ratio")  # type: ignore
    pipeline_type: Literal["single", "multi", "async"] = Field(
        default="single", description="Pipeline type"
    )  # type: ignore
    queue_size: int = Field(
        default=64, description="Queue size for multi-threaded pipeline"
    )  # type: ignore
    max_concurrent: int = Field(
        default=10, description="Maximum concurrent requests for async pipeline"
    )  # type: ignore


class ROIClassifierConfigModel(BaseModel):
    """ROI Classifier configuration model."""

    weights: Optional[str] = Field(default=None, description="Path to ROI weights")
    feature_extractor_path: str = Field(
        default="facebook/dinov2-with-registers-small",
        description="Feature extractor path",
    )
    cls_label_map: Dict[int, str] = Field(
        default_factory=lambda: {0: "groundtruth", 1: "other"},
        description="Classification label mapping",
    )
    keep_classes: List[str] = Field(
        default_factory=lambda: ["groundtruth"], description="Classes to keep"
    )
    cls_imgsz: int = Field(default=128, gt=0, description="Image size for classifier")


class InferenceServiceConfigModel(BaseModel):
    """Inference service configuration model."""

    url: Optional[str] = Field(default=None, description="Inference service URL")


class ProfilingConfigModel(BaseModel):
    """Profiling configuration model."""

    enable: bool = Field(default=False, description="Enable profiling")
    memory_profile: bool = Field(default=False, description="Enable memory profiling")
    line_profile: bool = Field(
        default=False, description="Enable line-by-line profiling"
    )
    gpu_profile: bool = Field(default=False, description="Enable GPU profiling")


class OutputConfigModel(BaseModel):
    """Output configuration model."""

    directory: str = Field(default="results", description="Output directory")
    dataset_name: Optional[str] = Field(
        default=None, description="FiftyOne dataset name"
    )


class LoggingConfigModel(BaseModel):
    """Logging configuration model."""

    verbose: bool = Field(default=False, description="Verbose logging")
    log_file: Optional[str] = Field(default=None, description="Log file path")


class CampaignConfigModel(BaseModel):
    """Campaign configuration model."""

    id: str = Field(description="Campaign identifier")
    pilot_name: str = Field(default="Unknown", description="Pilot name")
    target_species: Optional[List[str]] = Field(
        default=None, description="Target species"
    )


class ExportConfigModel(BaseModel):
    """Export configuration model."""

    to_fiftyone: bool = Field(default=True, description="Export to FiftyOne")
    create_map: bool = Field(default=True, description="Create geographic map")
    output_directory: Optional[str] = Field(
        default=None, description="Output directory"
    )
    export_to_labelstudio: bool = Field(
        default=True, description="Export to Label Studio"
    )


class GeographicConfigModel(BaseModel):
    """Geographic visualization configuration model."""

    create_map: bool = Field(default=True, description="Create geographic map")
    show_confidence: bool = Field(default=False, description="Show confidence scores")
    output_directory: str = Field(
        default="visualizations", description="Output directory"
    )
    map_type: str = Field(default="folium", description="Map type")
    zoom_level: int = Field(default=12, ge=1, le=20, description="Map zoom level")
    center_on_data: bool = Field(default=True, description="Center map on data")


class VisualizationConfigModel(BaseModel):
    """Visualization configuration model."""

    show_detections: bool = Field(default=True, description="Show detections")
    show_footprints: bool = Field(default=True, description="Show footprints")
    show_statistics: bool = Field(default=True, description="Show statistics")
    color_by_confidence: bool = Field(default=False, description="Color by confidence")
    confidence_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Confidence threshold"
    )


class OutputFormatConfigModel(BaseModel):
    """Output format configuration model."""

    format: str = Field(default="html", description="Output format")
    include_legend: bool = Field(default=True, description="Include legend")
    include_statistics: bool = Field(default=True, description="Include statistics")
    auto_open: bool = Field(default=False, description="Auto-open output")


class DetectConfigModel(BaseModel):
    """Configuration model for detect command."""

    images: List[str] = Field(description="Image directory path")
    model: ModelConfigModel = Field(default_factory=ModelConfigModel)
    processing: ProcessingConfigModel = Field(default_factory=ProcessingConfigModel)
    flight_specs: FlightSpecsModel = Field(default_factory=FlightSpecsModel)
    roi_classifier: ROIClassifierConfigModel = Field(
        default_factory=ROIClassifierConfigModel
    )
    inference_service: InferenceServiceConfigModel = Field(
        default_factory=InferenceServiceConfigModel
    )
    profiling: ProfilingConfigModel = Field(default_factory=ProfilingConfigModel)
    output: OutputConfigModel = Field(default_factory=OutputConfigModel)
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)

    def to_prediction_config(self, verbose: bool = False) -> PredictionConfig:
        """Convert to existing PredictionConfig dataclass."""
        return PredictionConfig(
            model_path=self.model.path,
            model_type=self.model.type,
            confidence_threshold=self.model.confidence_threshold,
            device=self.model.device,
            batch_size=self.model.batch_size,
            tilesize=self.processing.tile_size,
            flight_specs=self.flight_specs.to_flight_specs(),
            roi_weights=self.roi_classifier.weights,
            cls_imgsz=self.roi_classifier.cls_imgsz,
            keep_classes=self.roi_classifier.keep_classes,
            feature_extractor_path=self.roi_classifier.feature_extractor_path,
            cls_label_map=self.roi_classifier.cls_label_map,
            inference_service_url=self.inference_service.url,
            verbose=verbose,
            nms_iou=self.model.nms_iou,
            overlap_ratio=self.processing.overlap_ratio,
            pipeline_type=self.processing.pipeline_type,
            queue_size=self.processing.queue_size,
            max_concurrent=self.processing.max_concurrent,
        )

    def to_loader_config(self) -> LoaderConfig:
        """Convert to existing LoaderConfig dataclass."""
        return LoaderConfig(
            tile_size=self.processing.tile_size,
            batch_size=self.model.batch_size,
            num_workers=0,
            overlap=self.processing.overlap_ratio,
            flight_specs=self.flight_specs.to_flight_specs(),
        )


class CensusConfigModel(BaseModel):
    """Configuration model for census command."""

    campaign: CampaignConfigModel = Field(description="Campaign configuration")
    detection: DetectConfigModel = Field(default_factory=DetectConfigModel)
    export: ExportConfigModel = Field(default_factory=ExportConfigModel)
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)


class VisualizeConfigModel(BaseModel):
    """Configuration model for visualize command."""

    geographic: GeographicConfigModel = Field(default_factory=GeographicConfigModel)
    flight_specs: FlightSpecsModel = Field(default_factory=FlightSpecsModel)
    visualization: VisualizationConfigModel = Field(
        default_factory=VisualizationConfigModel
    )
    output: OutputFormatConfigModel = Field(default_factory=OutputFormatConfigModel)
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)


# Union type for all configuration models
ConfigModel = Union[DetectConfigModel, CensusConfigModel, VisualizeConfigModel]


def create_default_config(command_type: str) -> ConfigModel:
    """Create default configuration for a command type."""
    if command_type == "detect":
        return DetectConfigModel()
    elif command_type == "census":
        return CensusConfigModel(campaign=CampaignConfigModel(id="campaign_001"))
    elif command_type == "visualize":
        return VisualizeConfigModel()
    else:
        raise ValueError(f"Unknown command type: {command_type}")


def validate_config_dict(config_dict: Dict[str, Any], command_type: str) -> ConfigModel:
    """Validate and convert dictionary to appropriate config model."""
    try:
        if command_type == "detect":
            return DetectConfigModel(**config_dict)
        elif command_type == "census":
            return CensusConfigModel(**config_dict)
        elif command_type == "visualize":
            return VisualizeConfigModel(**config_dict)
        else:
            raise ValueError(f"Unknown command type: {command_type}")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def config_model_to_dict(config_model: ConfigModel) -> Dict[str, Any]:
    """Convert config model to dictionary."""
    return config_model.model_dump()
