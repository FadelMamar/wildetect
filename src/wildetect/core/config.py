"""
Configuration utilities for WildDetect.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pandas as pd
import torch
import yaml
from torch.utils.data._utils import pin_memory

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[3]


class DetectionPipelineTypes(StrEnum):
    """Types of detection pipelines."""

    MT = "mt"
    MT_SIMPLE = "mt_simple"
    MP = "mp"
    ASYNC = "async"
    SIMPLE = "simple"
    RASTER = "raster"
    MT_RASTER = "mt_raster"
    DEFAULT = "default"


class DetectionTypes(StrEnum):
    """Types of detection."""

    ANNOTATIONS = "annotations"
    PREDICTIONS = "predictions"


@dataclass
class FlightSpecs:
    """Configuration for flight specifications, including sensor and flight parameters."""

    sensor_height: float = 24  # in mm
    focal_length: float = 35  # in mm
    flight_height: float = 180  # in meters
    gsd: Optional[float] = None  # in cm/px


@dataclass
class PredictionConfig:
    """Configuration for prediction/inference, including tiling, thresholds, and device settings."""

    mlflow_model_name: Optional[str] = None
    mlflow_model_alias: Optional[str] = None

    tilesize: int = 800
    overlap_ratio: float = 0.2
    device: str = "auto"  # "cuda" if torch.cuda.is_available() else "cpu"

    flight_specs: FlightSpecs = field(default_factory=FlightSpecs)
    batch_size: int = 8
    verbose: bool = False
    buffer_size: int = 24
    timeout: int = 60

    # inference service
    inference_service_url: Optional[str] = None

    # Pipeline configuration
    pipeline_type: DetectionPipelineTypes = DetectionPipelineTypes.DEFAULT
    queue_size: int = 24  # Queue size for multi-threaded pipeline
    num_workers: int = (
        3  # Number of workers for multi-processing pipeline or dataset mode prediction
    )
    max_concurrent: int = 10  # Maximum concurrent requests for async pipeline

    def __post_init__(self):
        """Validate that required attributes are not None after initialization."""
        for a in [
            self.batch_size,
            self.tilesize,
            self.overlap_ratio,
        ]:
            assert a is not None

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.mlflow_model_name is None:
            self.mlflow_model_name = os.environ.get("MLFLOW_DETECTOR_NAME", None)

        if self.mlflow_model_alias is None:
            self.mlflow_model_alias = os.environ.get("MLFLOW_DETECTOR_ALIAS", None)

        if self.mlflow_model_name is None or self.mlflow_model_alias is None:
            raise ValueError(
                "MLFLOW_DETECTOR_NAME and MLFLOW_DETECTOR_ALIAS must be set or provided in the config"
            )

    @classmethod
    def from_yaml(
        cls, yaml_path: str = "configs/prediction.yaml"
    ) -> "PredictionConfig":
        """
        Create a PredictionConfig instance from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            PredictionConfig instance with values loaded from YAML

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        yaml_path_obj = Path(yaml_path)

        if not yaml_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path_obj, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")

        if config_data is None:
            return cls()

        # Handle nested FlightSpecs if present
        if "flight_specs" in config_data and isinstance(
            config_data["flight_specs"], dict
        ):
            flight_specs_data = config_data.pop("flight_specs")
            config_data["flight_specs"] = FlightSpecs(**flight_specs_data)

        # Convert YAML data to dataclass instance
        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Save the configuration to a YAML file."""
        vars_dict = vars(self)
        vars_dict["flight_specs"] = vars(self.flight_specs)
        return vars_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionConfig":
        flight_specs = data.pop("flight_specs")
        if isinstance(flight_specs, dict):
            flight_specs = FlightSpecs(**flight_specs)
        elif isinstance(flight_specs, FlightSpecs):
            pass
        else:
            raise ValueError(f"Invalid flight specs type: {type(flight_specs)}")

        data["flight_specs"] = flight_specs
        return cls(**data)


@dataclass
class LoaderConfig:
    """Configuration for the data loader."""

    # Directory and file settings
    supported_formats: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".tif",
        ".bmp",
    )
    recursive: bool = True

    # Tile extraction settings
    tile_size: int = 800
    overlap: float = 0.2  # Overlap ratio between tiles

    # Batch processing
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False

    # GPS and metadata
    flight_specs: Optional[FlightSpecs] = field(default_factory=FlightSpecs)

    csv_data: Optional[pd.DataFrame] = None
    lat_col: Optional[str] = None
    lon_col: Optional[str] = None
    alt_col: Optional[str] = None

    def csv_data_to_dict(self) -> Dict[str, Any]:
        """Convert CSV data to dictionary."""
        assert (self.lat_col is not None) and (self.lon_col is not None) and (self.alt_col is not None), "lat_col, lon_col, and alt_col must be provided"
        cfg = {self.lat_col: 'latitude', self.lon_col: 'longitude', self.alt_col: 'altitude'}
        return (self.csv_data
                            .rename(columns=cfg)
                            .set_index('image_path')
                            .to_dict(orient='index')
                        )
