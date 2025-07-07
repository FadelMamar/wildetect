"""
Configuration utilities for WildDetect.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


@dataclass
class FlightSpecs:
    """Configuration for flight specifications, including sensor and flight parameters."""

    sensor_height: float = 24  # in mm
    focal_length: float = 35  # in mm
    flight_height: float = 180  # in meters


@dataclass
class PredictionConfig:
    """Configuration for prediction/inference, including tiling, thresholds, and device settings."""

    model_path: Optional[str] = None
    model_type: str = "yolo"

    tilesize: int = 640
    overlap_ratio: float = 0.2
    confidence_threshold: float = 0.25
    min_area: int = 10 * 10
    max_area: Optional[int] = None
    nms_iou: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    flight_specs: FlightSpecs = field(default_factory=FlightSpecs)

    batch_size: int = 8

    roi_weights: Optional[str] = None

    # Image classifier imgsz
    cls_imgsz: int = 96

    verbose: bool = False

    buffer_size = 24
    timeout = 60

    # inference service
    inference_service_url: Optional[str] = None

    def __post_init__(self):
        """Validate that required attributes are not None after initialization."""
        for a in [
            self.batch_size,
            self.nms_iou,
            self.cls_imgsz,
            self.tilesize,
            self.confidence_threshold,
            self.overlap_ratio,
        ]:
            assert a is not None

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_path is None:
            self.model_path = os.environ.get("WILDETECT_MODEL_PATH")
        else:
            self.model_path = str(Path(self.model_path).resolve())

        if self.roi_weights is None:
            self.roi_weights = os.environ.get("WILDETECT_ROI_WEIGHTS")
        else:
            self.roi_weights = str(Path(self.roi_weights).resolve())

        if self.inference_service_url is None:
            self.inference_service_url = os.environ.get(
                "WILDETECT_INFERENCE_SERVICE_URL"
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
    tile_size: int = 640
    overlap: float = 0.2  # Overlap ratio between tiles

    # Batch processing
    batch_size: int = 1
    num_workers: int = 0

    # GPS and metadata
    flight_specs: Optional[FlightSpecs] = field(default_factory=FlightSpecs)
    extract_gps: bool = True

    # Caching
    cache_images: bool = False
    cache_dir: Optional[str] = None


def get_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        # Return default config if file doesn't exist
        return _get_default_config()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "model": {
            "type": "yolo",
            "weights": "models/yolo_wildlife.pt",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_size": [640, 640],
            "device": "auto",
        },
        "detection": {
            "min_confidence": 0.3,
            "max_detections": 100,
            "enable_tracking": False,
            "species_classes": [
                "elephant",
                "giraffe",
                "zebra",
                "lion",
                "rhino",
                "buffalo",
                "antelope",
                "deer",
                "bear",
                "wolf",
                "fox",
                "rabbit",
                "bird",
                "other",
            ],
        },
        "paths": {
            "data_dir": "data",
            "images_dir": "data/images",
            "annotations_dir": "data/annotations",
            "models_dir": "models",
            "datasets_dir": "data/datasets",
            "logs_dir": "logs",
        },
        "fiftyone": {
            "dataset_name": "wildlife_detection",
            "max_samples": 10000,
            "enable_brain": True,
            "brain_methods": ["similarity", "hardest", "mistakenness"],
        },
        "database": {"url": "sqlite:///data/annotations.db", "echo": False},
        "api": {"host": "0.0.0.0", "port": 8000, "workers": 1, "reload": True},
        "ui": {
            "title": "WildDetect - Wildlife Detection System",
            "theme": "light",
            "page_icon": "🦁",
        },
        "processing": {
            "batch_size": 4,
            "num_workers": 2,
            "enable_augmentation": True,
            "resize_method": "letterbox",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/wildetect.log",
        },
    }


def create_directories():
    """Create necessary directories for the application."""
    config = get_config()

    directories = [
        config["paths"]["data_dir"],
        config["paths"]["images_dir"],
        config["paths"]["annotations_dir"],
        config["paths"]["models_dir"],
        config["paths"]["datasets_dir"],
        config["paths"]["logs_dir"],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
