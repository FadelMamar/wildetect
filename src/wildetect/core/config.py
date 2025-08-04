"""
Configuration utilities for WildDetect.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import albumentations as A
import torch
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[3]


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

    cls_label_map: Optional[Dict[int, str]] = field(
        default_factory=lambda: {0: "groundtruth", 1: "other"}
    )
    feature_extractor_path: Optional[str] = "facebook/dinov2-with-registers-small"
    roi_weights: Optional[str] = None
    transform: Optional[A.Compose] = None

    # Image classifier imgsz
    cls_imgsz: int = 96
    keep_classes: Optional[Sequence[str]] = ("groundtruth",)

    verbose: bool = False

    buffer_size = 24
    timeout = 60

    # inference service
    inference_service_url: Optional[str] = None

    # Pipeline configuration
    pipeline_type: Literal["single", "multi"] = "single"
    queue_size: int = 24  # Queue size for multi-threaded pipeline

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
            model_path = os.environ.get("WILDETECT_MODEL_PATH", None)
            if model_path:
                if Path(model_path).exists():
                    self.model_path = model_path
                elif os.environ.get("MLFLOW_DETECTOR_NAME", None) and os.environ.get(
                    "MLFLOW_DETECTOR_ALIAS", None
                ):
                    pass
                else:
                    logger.warning(
                        "Model path not found in environment variables or config file."
                    )
        else:
            self.model_path = str(Path(self.model_path).resolve())

        if self.roi_weights is None:
            roi_weights = os.environ.get("ROI_MODEL_PATH", None)
            if roi_weights:
                if Path(roi_weights).exists():
                    self.roi_weights = roi_weights
            elif os.environ.get("MLFLOW_ROI_NAME", None) and os.environ.get(
                "MLFLOW_ROI_ALIAS", None
            ):
                pass
            else:
                logger.warning(
                    "ROI model path not found in environment variables or config file."
                )
        else:
            if not Path(self.roi_weights).exists():
                logger.warning(f"ROI weights file not found: {self.roi_weights}")
                self.roi_weights = None
            else:
                self.roi_weights = str(Path(self.roi_weights).resolve())

        # if self.inference_service_url is None:
        #    self.inference_service_url = os.environ.get(
        #        "WILDETECT_INFERENCE_SERVICE_URL", None
        #    )

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
    tile_size: int = 640
    overlap: float = 0.2  # Overlap ratio between tiles

    # Batch processing
    batch_size: int = 1
    num_workers: int = 0

    # GPS and metadata
    flight_specs: Optional[FlightSpecs] = field(default_factory=FlightSpecs)


def get_config(
    config_path: Union[str, Path] = "config/settings.yaml"
) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    if not config_path.exists():
        # Return default config if file doesn't exist
        return _get_default_config()

    with open(config_path, "r", encoding="utf-8") as f:
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
