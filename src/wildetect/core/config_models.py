"""
Pydantic configuration models for WildDetect.

This module provides Pydantic BaseModel classes for configuration validation,
type checking, and automatic serialization/deserialization.
It extends the existing dataclasses from config.py to avoid code duplication.
"""

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

from ..core.data.utils import get_images_paths
from .config import (
    DetectionPipelineTypes,
    DetectionTypes,
    FlightSpecs,
    LoaderConfig,
    PredictionConfig,
)

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
    gsd: Optional[float] = Field(default=None, gt=0, description="GSD in cm/px")

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
            gsd=self.gsd,
        )


class ExifGPSUpdateConfig(BaseModel):
    """Configuration for updating EXIF GPS data from CSV."""

    image_folder: Optional[str] = Field(
        default=None, description="Path to folder containing images"
    )
    csv_path: Optional[str] = Field(
        default=None, description="Path to CSV file with GPS coordinates"
    )
    skip_rows: int = Field(default=0, description="Number of rows to skip in CSV")
    filename_col: str = Field(
        default="filename", description="CSV column name for filenames"
    )
    lat_col: str = Field(default="latitude", description="CSV column name for latitude")
    lon_col: str = Field(
        default="longitude", description="CSV column name for longitude"
    )
    alt_col: str = Field(default="altitude", description="CSV column name for altitude")

    @classmethod
    def from_yaml(cls, path: str) -> "ExifGPSUpdateConfig":
        """Load configuration from YAML file."""
        with open(
            path,
            "r",
        ) as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError("YAML file is empty or invalid")
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


class ModelConfigModel(BaseModel):
    """Model configuration model."""

    mlflow_model_name: Optional[str] = Field(
        default=None, description="MLFlow model name"
    )
    mlflow_model_alias: Optional[str] = Field(
        default=None, description="MLFlow model alias"
    )
    device: str = Field(default="auto", description="Device to run inference on")


class LabelStudioConfigModel(BaseModel):
    """Label Studio configuration model."""

    url: Optional[str] = Field(default=None, description="Label Studio URL")
    api_key: Optional[str] = Field(default=None, description="Label Studio API key")
    download_resources: bool = Field(default=True, description="Download resources")
    project_id: Optional[int] = Field(
        default=None, description="Label Studio project ID"
    )
    json_path: Optional[str] = Field(default=None, description="Label Studio JSON path")
    dotenv_path: Optional[str] = Field(
        default=None, description="Label Studio dotenv path"
    )
    parse_ls_config: bool = Field(default=True, description="Parse Label Studio config")
    ls_xml_config: Optional[str] = Field(
        default=None, description="Label Studio XML config"
    )
    from_name: str = Field(default="label", description="Label Studio from name")
    to_name: str = Field(default="image", description="Label Studio to name")
    label_type: str = Field(
        default="rectanglelabels", description="Label Studio label type"
    )

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, v: str) -> str:
        """Validate that json_path exists."""
        if v is None:
            return v
        if not Path(v).exists():
            raise ValueError(f"JSON path does not exist: {v}")
        return v


class ProcessingConfigModel(BaseModel):
    """Processing configuration model."""

    tile_size: int = Field(default=800, description="Tile size for processing")  # type: ignore
    overlap_ratio: float = Field(default=0.2, description="Tile overlap ratio")  # type: ignore
    pipeline_type: DetectionPipelineTypes = Field(
        default=DetectionPipelineTypes.DEFAULT, description="Pipeline type"
    )  # type: ignore
    queue_size: int = Field(
        default=64, gt=0, description="Queue size for multi-threaded pipeline"
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for inference")
    num_data_workers: int = Field(
        default=2, ge=0, description="Number of workers for data loading"
    )
    num_inference_workers: int = Field(
        default=2, ge=1, description="Number of workers for inference"
    )
    pin_memory: bool = Field(default=False, description="Pin memory for data loading")
    prefetch_factor: Optional[int] = Field(
        default=None, description="Prefetch factor for data loading"
    )
    max_concurrent: int = Field(
        default=4, ge=1, description="Maximum number of concurrent inference tasks"
    )
    nms_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="NMS threshold for detections stitching",
    )
    max_errors: int = Field(
        default=5, ge=1, description="Maximum number of errors before stopping"
    )


class InferenceServiceConfigModel(BaseModel):
    """Inference service configuration model."""

    url: Optional[str] = Field(default=None, description="Inference service URL")
    timeout: int = Field(default=60, ge=0, description="Timeout for inference")


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

    image_paths: Optional[List[str]] = Field(
        default=None, description="Image directory path"
    )
    image_dir: Optional[str] = Field(default=None, description="Image directory path")
    exif_gps_update: Optional[ExifGPSUpdateConfig] = Field(
        default=None, description="EXIF GPS update configuration"
    )
    model: ModelConfigModel = Field(default_factory=ModelConfigModel)
    processing: ProcessingConfigModel = Field(default_factory=ProcessingConfigModel)
    flight_specs: FlightSpecsModel = Field(default_factory=FlightSpecsModel)
    inference_service: InferenceServiceConfigModel = Field(
        default_factory=InferenceServiceConfigModel
    )
    profiling: ProfilingConfigModel = Field(default_factory=ProfilingConfigModel)
    output: OutputConfigModel = Field(default_factory=OutputConfigModel)
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)
    labelstudio: LabelStudioConfigModel = Field(default_factory=LabelStudioConfigModel)

    def to_prediction_config(self, verbose: bool = False) -> PredictionConfig:
        """Convert to existing PredictionConfig dataclass."""
        return PredictionConfig(
            mlflow_model_name=self.model.mlflow_model_name,
            mlflow_model_alias=self.model.mlflow_model_alias,
            inference_service_url=self.inference_service.url,
            timeout=self.inference_service.timeout,
            device=self.model.device,
            batch_size=self.processing.batch_size,
            tilesize=self.processing.tile_size,
            flight_specs=self.flight_specs.to_flight_specs(),
            verbose=verbose,
            overlap_ratio=self.processing.overlap_ratio,
            pipeline_type=self.processing.pipeline_type,
            queue_size=self.processing.queue_size,
            max_concurrent=self.processing.max_concurrent,
            num_workers=self.processing.num_inference_workers,
            nms_threshold=self.processing.nms_threshold,
            max_errors=self.processing.max_errors,
        )

    def to_loader_config(self) -> LoaderConfig:
        """Convert to existing LoaderConfig dataclass."""
        cfg = dict()
        if self.exif_gps_update is not None:
            if (self.exif_gps_update.csv_path is not None) and (
                self.exif_gps_update.image_folder is not None
            ):
                try:
                    df = pd.read_csv(
                        self.exif_gps_update.csv_path,
                        skiprows=self.exif_gps_update.skip_rows,
                        sep=";",
                    )
                except Exception:
                    logger.info(
                        f"Failed to read CSV file {self.exif_gps_update.csv_path} with separator ';', trying with ','"
                    )
                    df = pd.read_csv(
                        self.exif_gps_update.csv_path,
                        skiprows=self.exif_gps_update.skip_rows,
                        sep=",",
                    )
                df["image_path"] = df[self.exif_gps_update.filename_col].apply(
                    lambda x: os.path.join(self.exif_gps_update.image_folder, x)
                )

                cfg = dict(
                    lat_col=self.exif_gps_update.lat_col,
                    lon_col=self.exif_gps_update.lon_col,
                    alt_col=self.exif_gps_update.alt_col,
                    csv_data=df,
                )

        return LoaderConfig(
            tile_size=self.processing.tile_size,
            batch_size=self.processing.batch_size,
            num_workers=self.processing.num_data_workers,
            overlap=self.processing.overlap_ratio,
            flight_specs=self.flight_specs.to_flight_specs(),
            pin_memory=self.processing.pin_memory,
            prefetch_factor=self.processing.prefetch_factor,
            **cfg,
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
    labelstudio: LabelStudioConfigModel = Field(default_factory=LabelStudioConfigModel)
    image_dir: Optional[str] = Field(default=None, description="Image directory")
    csv_output_path: Optional[str] = Field(default=None, description="CSV output path")
    detection_type: DetectionTypes = Field(
        default=DetectionTypes.ANNOTATIONS, description="Detection type"
    )


class TestImagesConfigModel(BaseModel):
    """Test images configuration for benchmarking."""

    path: str = Field(description="Path to test images directory")
    recursive: bool = Field(default=True, description="Search recursively for images")
    max_images: int = Field(
        default=100, gt=0, le=10000, description="Maximum number of images to use"
    )
    supported_formats: List[str] = Field(
        default=["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif", "*.bmp"],
        description="Supported image formats",
    )


class HyperparameterSearchConfigModel(BaseModel):
    """Hyperparameter search space configuration for benchmarking."""

    batch_size: List[int] = Field(
        default=[8, 16, 32, 64, 128, 256, 512], description="Batch sizes to test"
    )
    num_workers: List[int] = Field(
        default=[0, 2, 4, 8, 16], description="Number of workers to test"
    )
    tile_size: List[int] = Field(
        default=[400, 800, 1200, 1600], description="Tile sizes to test"
    )
    overlap_ratio: List[float] = Field(
        default=[0.1, 0.2, 0.3], description="Overlap ratios to test"
    )


class BenchmarkFormatTypes(StrEnum):
    """Types of benchmark format."""

    JSON = "json"
    CSV = "csv"
    BOTH = "both"


class BenchmarkDirectionTypes(StrEnum):
    """Types of benchmark direction."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class BenchmarkSamplerTypes(StrEnum):
    """Types of benchmark sampler."""

    TPE = "TPE"
    RANDOM = "Random"
    GRID = "Grid"


class BenchmarkObjectiveTypes(StrEnum):
    """Types of benchmark objective."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"


class BenchmarkOutputConfigModel(BaseModel):
    """Output configuration for benchmark results."""

    directory: str = Field(
        default="results/benchmarks", description="Output directory for results"
    )
    save_plots: bool = Field(default=True, description="Save performance plots")
    save_results: bool = Field(default=True, description="Save detailed results")
    format: BenchmarkFormatTypes = Field(
        default=BenchmarkFormatTypes.JSON, description="Output format"
    )
    include_optimization_history: bool = Field(
        default=True, description="Include optimization history"
    )
    auto_open: bool = Field(
        default=False, description="Auto-open results after completion"
    )


class BenchmarkExecutionConfigModel(BaseModel):
    """Benchmark execution configuration."""

    n_trials: int = Field(
        default=30, gt=1, le=1000, description="Number of optimization trials"
    )
    timeout: int = Field(
        default=3600, gt=0, description="Maximum time for optimization in seconds"
    )
    direction: BenchmarkDirectionTypes = Field(
        default=BenchmarkDirectionTypes.MAXIMIZE,
        description="Optimization direction (minimize latency, maximize throughput)",
    )
    objective: BenchmarkObjectiveTypes = Field(
        default=BenchmarkObjectiveTypes.THROUGHPUT, description="Objective to optimize"
    )
    sampler: BenchmarkSamplerTypes = Field(
        default=BenchmarkSamplerTypes.TPE, description="Optuna sampler type"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class BenchmarkConfigModel(BaseModel):
    """Configuration model for benchmark command."""

    # Core benchmark settings
    execution: BenchmarkExecutionConfigModel = Field(
        default_factory=BenchmarkExecutionConfigModel
    )

    # Test data configuration
    test_images: TestImagesConfigModel = Field(description="Test images configuration")

    # Hyperparameter search space
    hyperparameters: HyperparameterSearchConfigModel = Field(
        default_factory=HyperparameterSearchConfigModel
    )

    # Output configuration
    output: BenchmarkOutputConfigModel = Field(
        default_factory=BenchmarkOutputConfigModel
    )

    # Detection pipeline configuration (reuses existing models)
    model: ModelConfigModel = Field(default_factory=ModelConfigModel)
    processing: ProcessingConfigModel = Field(default_factory=ProcessingConfigModel)
    flight_specs: FlightSpecsModel = Field(default_factory=FlightSpecsModel)
    inference_service: InferenceServiceConfigModel = Field(
        default_factory=InferenceServiceConfigModel
    )

    # Logging and profiling
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)
    profiling: ProfilingConfigModel = Field(default_factory=ProfilingConfigModel)

    @field_validator("test_images")
    @classmethod
    def validate_test_images_path(
        cls, v: TestImagesConfigModel
    ) -> TestImagesConfigModel:
        """Validate that test images path exists."""
        if not Path(v.path).exists():
            raise ValueError(f"Test images path does not exist: {v.path}")
        return v

    @field_validator("hyperparameters")
    @classmethod
    def validate_hyperparameter_ranges(
        cls, v: HyperparameterSearchConfigModel
    ) -> HyperparameterSearchConfigModel:
        """Validate hyperparameter search ranges are reasonable."""
        if not v.batch_size:
            raise ValueError("At least one batch size must be specified")
        if not v.num_workers:
            raise ValueError("At least one num_workers value must be specified")
        if max(v.batch_size) > 1024:
            raise ValueError("Batch size values should be reasonable (max 1024)")
        return v

    def to_prediction_config(self) -> PredictionConfig:
        """Convert to existing PredictionConfig dataclass."""
        return PredictionConfig(
            mlflow_model_name=self.model.mlflow_model_name,
            mlflow_model_alias=self.model.mlflow_model_alias,
            inference_service_url=self.inference_service.url,
            timeout=self.inference_service.timeout,
            device=self.model.device,
            batch_size=self.processing.batch_size,
            tilesize=self.processing.tile_size,
            flight_specs=self.flight_specs.to_flight_specs(),
            verbose=self.logging.verbose,
            overlap_ratio=self.processing.overlap_ratio,
            pipeline_type=self.processing.pipeline_type,
            queue_size=self.processing.queue_size,
            max_concurrent=self.processing.max_concurrent,
            nms_threshold=self.processing.nms_threshold,
            max_errors=self.processing.max_errors,
        )

    def to_loader_config(self) -> LoaderConfig:
        """Convert to existing LoaderConfig dataclass."""
        return LoaderConfig(
            tile_size=self.processing.tile_size,
            batch_size=self.processing.batch_size,
            num_workers=self.processing.num_data_workers,
            overlap=self.processing.overlap_ratio,
            flight_specs=self.flight_specs.to_flight_specs(),
            prefetch_factor=self.processing.prefetch_factor,
        )

    def get_image_paths(self) -> list:
        """Find test images based on configuration."""

        # Use the existing utility function to find image files
        image_paths = get_images_paths(
            images_dir=self.test_images.path,
            patterns=tuple(self.test_images.supported_formats),
        )
        # Limit to max_images if specified
        if (
            self.test_images.max_images
            and len(image_paths) > self.test_images.max_images
        ):
            image_paths = image_paths[: self.test_images.max_images]

        return image_paths


ConfigModel = Union[
    DetectConfigModel, CensusConfigModel, VisualizeConfigModel, BenchmarkConfigModel
]


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
        elif command_type == "benchmark":
            return BenchmarkConfigModel(**config_dict)
        else:
            raise ValueError(f"Unknown command type: {command_type}")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def config_model_to_dict(config_model: ConfigModel) -> Dict[str, Any]:
    """Convert config model to dictionary."""
    return config_model.model_dump()
