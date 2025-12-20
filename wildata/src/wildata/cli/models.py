"""
Configuration models for CLI commands.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator

from ..config import (
    ENV_FILE,
    AugmentationConfig,
    ROIConfig,
    TilingConfig,
    TransformationConfig,
)


class ImportDatasetConfig(BaseModel):
    """Configuration for importing datasets."""

    # Required parameters
    source_path: str = Field(..., description="Path to source dataset")
    source_format: str = Field(..., description="Source format (coco/yolo)")
    dataset_name: str = Field(..., description="Name for the dataset")

    # Pipeline configuration
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="train", description="Split name (train/val/test)")
    enable_dvc: bool = Field(default=False, description="Enable DVC integration")

    # Processing options
    processing_mode: str = Field(
        default="batch", description="Processing mode (streaming/batch)"
    )
    track_with_dvc: bool = Field(default=False, description="Track dataset with DVC")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")

    # Label Studio options
    dotenv_path: Optional[str] = Field(
        default=ENV_FILE, description="Path to .env file"
    )
    ls_xml_config: Optional[str] = Field(
        default=None, description="Label Studio XML config path"
    )
    ls_parse_config: bool = Field(
        default=False, description="Parse Label Studio config"
    )

    # ROI configuration
    roi_config: Optional[ROIConfig] = Field(
        default=None, description="ROI configuration"
    )
    disable_roi: bool = Field(default=False, description="Disable ROI extraction")

    # Transformation pipeline configuration
    transformations: Optional[TransformationConfig] = Field(
        default=None, description="Transformation pipeline config"
    )

    # Validation methods
    @field_validator("source_format", mode="before")
    @classmethod
    def validate_source_format(cls, v: Any) -> str:
        if v not in ["coco", "yolo", "ls"]:
            raise ValueError('source_format must be either "coco" or "yolo"')
        return v

    @field_validator("split_name", mode="before")
    @classmethod
    def validate_split_name(cls, v: Any) -> str:
        if v not in ["train", "val", "test"]:
            raise ValueError("split_name must be one of: train, val, test")
        return v

    @field_validator("processing_mode", mode="before")
    @classmethod
    def validate_processing_mode(cls, v: Any) -> str:
        if v not in ["streaming", "batch"]:
            raise ValueError('processing_mode must be either "streaming" or "batch"')
        return v

    @field_validator("dotenv_path", mode="before")
    @classmethod
    def validate_dotenv_path(cls, v: Any) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Dotenv path does not exist: {v}")
        return v

    @field_validator("ls_xml_config", mode="before")
    @classmethod
    def validate_ls_xml_config(cls, v: Any) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Label Studio XML config path does not exist: {v}")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "ImportDatasetConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)

    @field_validator("source_path", mode="before")
    @classmethod
    def validate_source_path(cls, v: Any) -> str:
        if not Path(v).is_file():
            raise ValueError(f"Source path is not a file: {v}")
        return str(v)


class ROIDatasetConfig(BaseModel):
    """Configuration for creating ROI datasets."""

    source_path: str = Field(..., description="Path to source dataset")
    source_format: str = Field(..., description="Source format (coco/yolo)")
    dataset_name: str = Field(..., description="Name for the dataset")
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="val", description="Split name (train/val/test)")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")
    roi_config: ROIConfig = Field(..., description="ROI configuration")
    ls_xml_config: Optional[str] = Field(
        default=None, description="Label Studio XML config path"
    )
    ls_parse_config: bool = Field(
        default=False, description="Parse Label Studio config"
    )
    draw_original_bboxes: bool = Field(
        default=False, description="Draw original bounding boxes on ROI images"
    )

    @field_validator("source_format", mode="before")
    @classmethod
    def validate_source_format(cls, v: Any) -> str:
        if v not in ["coco", "yolo", "ls"]:
            raise ValueError('source_format must be either "coco" or "yolo"')
        return v

    @field_validator("split_name", mode="before")
    @classmethod
    def validate_split_name(cls, v: Any) -> str:
        if v not in ["train", "val", "test"]:
            raise ValueError("split_name must be one of: train, val, test")
        return v

    @field_validator("source_path", mode="before")
    @classmethod
    def validate_source_path(cls, v: Any) -> str:
        if not Path(v).exists():
            raise ValueError(f"Source path does not exist: {v}")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "ROIDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


class BulkImportDatasetConfig(BaseModel):
    """Configuration for bulk importing datasets from a directory."""

    source_paths: List[str] = Field(
        ..., description="List of directories containing source dataset files"
    )
    source_format: str = Field(..., description="Source format (coco/yolo/ls)")
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="train", description="Split name (train/val/test)")
    enable_dvc: bool = Field(default=False, description="Enable DVC integration")
    processing_mode: str = Field(
        default="batch", description="Processing mode (streaming/batch)"
    )
    track_with_dvc: bool = Field(default=False, description="Track dataset with DVC")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")
    ls_xml_config: Optional[str] = Field(
        default=None, description="Label Studio XML config path"
    )
    ls_parse_config: bool = Field(
        default=False, description="Parse Label Studio config"
    )
    roi_config: Optional[ROIConfig] = Field(
        default=None, description="ROI configuration"
    )
    disable_roi: bool = Field(default=False, description="Disable ROI extraction")
    transformations: Optional[TransformationConfig] = Field(
        default=None, description="Transformation pipeline config"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "BulkImportDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @field_validator("source_paths", mode="before")
    @classmethod
    def validate_source_paths(cls, v: list[str]) -> list[str]:
        for path in v:
            if not Path(path).is_dir():
                raise ValueError(f"Source path is not a directory: {path}")
        return v

    # Validation methods
    @field_validator("source_format", mode="before")
    @classmethod
    def validate_source_format(cls, v: str) -> str:
        if v not in ["coco", "yolo", "ls"]:
            raise ValueError('source_format must be either "coco" or "yolo"')
        return v

    @field_validator("split_name", mode="before")
    @classmethod
    def validate_split_name(cls, v: str) -> str:
        if v not in ["train", "val", "test"]:
            raise ValueError("split_name must be one of: train, val, test")
        return v

    @field_validator("processing_mode", mode="before")
    @classmethod
    def validate_processing_mode(cls, v: str) -> str:
        if v not in ["streaming", "batch"]:
            raise ValueError('processing_mode must be either "streaming" or "batch"')
        return v

    @field_validator("ls_xml_config", mode="before")
    @classmethod
    def validate_ls_xml_config(cls, v: str) -> Optional[str]:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Label Studio XML config path does not exist: {v}")
        return v


class BulkCreateROIDatasetConfig(BaseModel):
    """Configuration for bulk creation of ROI datasets from a directory."""

    source_paths: List[str] = Field(
        ..., description="List of directories containing source dataset files"
    )
    source_format: str = Field(..., description="Source format (coco/yolo)")
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="val", description="Split name (train/val/test)")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")
    roi_config: ROIConfig = Field(..., description="ROI configuration")
    ls_xml_config: Optional[str] = Field(
        default=None, description="Label Studio XML config path"
    )
    ls_parse_config: bool = Field(
        default=False, description="Parse Label Studio config"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "BulkCreateROIDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


class ImportConfig(BaseModel):
    """Legacy import configuration (kept for backward compatibility)."""

    source_path: str
    format_type: str
    dataset_name: str
    track_with_dvc: bool = False
    augment: bool = False
    rotation_range: Tuple[float, float] = (-10.0, 10.0)
    probability: float = 0.5
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: Tuple[float, float] = (0.01, 0.1)
    tile: bool = False
    tile_size: int = 512
    stride: int = 256
    min_visibility: float = 0.1
    max_negative_tiles: int = 3
    negative_positive_ratio: float = 1.0


class ExifGPSUpdateConfig(BaseModel):
    """Configuration for updating EXIF GPS data from CSV."""

    image_folder: str = Field(..., description="Path to folder containing images")
    csv_path: str = Field(..., description="Path to CSV file with GPS coordinates")
    output_dir: str = Field(..., description="Output directory for updated images")
    skip_rows: int = Field(default=0, description="Number of rows to skip in CSV")
    filename_col: str = Field(
        default="filename", description="CSV column name for filenames"
    )
    lat_col: str = Field(default="latitude", description="CSV column name for latitude")
    lon_col: str = Field(
        default="longitude", description="CSV column name for longitude"
    )
    alt_col: str = Field(default="altitude", description="CSV column name for altitude")

    @field_validator("image_folder", mode="before")
    @classmethod
    def validate_image_folder(cls, v: Any) -> str:
        if not Path(v).exists():
            raise ValueError(f"Image folder does not exist: {v}")
        return str(v)

    @field_validator("csv_path", mode="before")
    @classmethod
    def validate_csv_path(cls, v: Any) -> str:
        if not Path(v).exists():
            raise ValueError(f"CSV file does not exist: {v}")
        return str(v)

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v: Any) -> str:
        return str(v)

    @field_validator("skip_rows", mode="before")
    @classmethod
    def validate_skip_rows(cls, v: Any) -> int:
        v = int(v)
        if v < 0:
            raise ValueError("skip_rows must be non-negative")
        return v

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
