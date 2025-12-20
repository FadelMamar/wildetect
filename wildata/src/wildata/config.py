from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from altair.utils.data import sample
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).parents[2]
ENV_FILE = str(ROOT / ".env")


class ROIConfig(BaseModel):
    """ROI configuration for CLI."""

    random_roi_count: int = Field(default=1, description="Number of random ROIs")
    roi_box_size: int = Field(default=128, description="ROI box size")
    min_roi_size: int = Field(default=32, description="Minimum ROI size")
    dark_threshold: float = Field(default=0.5, description="Dark threshold")
    background_class: str = Field(
        default="background", description="Background class name"
    )
    save_format: str = Field(default="jpg", description="Save format")
    quality: int = Field(default=95, description="Image quality")
    sample_background: bool = Field(
        default=True, description="Sample background from dataset"
    )

    @field_validator("random_roi_count", mode="before")
    @classmethod
    def validate_random_roi_count(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("roi_box_size", mode="before")
    @classmethod
    def validate_roi_box_size(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("min_roi_size", mode="before")
    @classmethod
    def validate_min_roi_size(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("quality", mode="before")
    @classmethod
    def validate_quality(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("dark_threshold", mode="before")
    @classmethod
    def validate_dark_threshold(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("dark_threshold must be between 0 and 1")
        return v

    @field_validator("save_format", mode="before")
    @classmethod
    def validate_save_format(cls, v: Any) -> str:
        if v not in ["jpg", "jpeg", "png"]:
            raise ValueError("save_format must be one of: jpg, jpeg, png")
        return v


class TilingConfig(BaseModel):
    """Tiling configuration for CLI."""

    tile_size: int = Field(default=512, description="Tile size")
    stride: int = Field(default=416, description="Stride between tiles")
    min_visibility: float = Field(default=0.1, description="Minimum visibility ratio")
    max_negative_tiles_in_negative_image: int = Field(
        default=3, description="Max negative tiles in negative image"
    )
    negative_positive_ratio: float = Field(
        default=1.0, description="Negative to positive ratio"
    )
    dark_threshold: float = Field(default=0.5, description="Dark threshold")

    @field_validator("tile_size", mode="before")
    @classmethod
    def validate_tile_size(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("stride", mode="before")
    @classmethod
    def validate_stride(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("max_negative_tiles_in_negative_image", mode="before")
    @classmethod
    def validate_max_negative_tiles_in_negative_image(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("min_visibility", mode="before")
    @classmethod
    def validate_min_visibility(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v

    @field_validator("dark_threshold", mode="before")
    @classmethod
    def validate_dark_threshold(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v

    @field_validator("negative_positive_ratio", mode="before")
    @classmethod
    def validate_negative_positive_ratio(cls, v: Any) -> float:
        v = float(v)
        if v < 0:
            raise ValueError("Ratio must be non-negative")
        return v


class AugmentationConfig(BaseModel):
    """Augmentation configuration for CLI."""

    rotation_range: Tuple[float, float] = Field(
        default=(-45, 45), description="Rotation range"
    )
    probability: float = Field(
        default=1.0, description="Probability of applying augmentation"
    )
    brightness_range: Tuple[float, float] = Field(
        default=(-0.2, 0.4), description="Brightness range"
    )
    scale: Tuple[float, float] = Field(default=(1.0, 2.0), description="Scale range")
    translate: Tuple[float, float] = Field(
        default=(-0.1, 0.2), description="Translation range"
    )
    shear: Tuple[float, float] = Field(default=(-5, 5), description="Shear range")
    contrast_range: Tuple[float, float] = Field(
        default=(-0.2, 0.4), description="Contrast range"
    )
    noise_std: Tuple[float, float] = Field(
        default=(0.01, 0.1), description="Noise standard deviation range"
    )
    seed: int = Field(default=41, description="Random seed")
    num_transforms: int = Field(default=2, description="Number of transformations")

    @field_validator("probability", mode="before")
    @classmethod
    def validate_probability(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v

    @field_validator("num_transforms", mode="before")
    @classmethod
    def validate_num_transforms(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class BboxClippingConfig(BaseModel):
    """Bounding box clipping configuration for CLI."""

    tolerance: int = Field(default=5, description="Tolerance for clipping")
    skip_invalid: bool = Field(default=False, description="Skip invalid annotations")

    @field_validator("tolerance", mode="before")
    @classmethod
    def validate_tolerance(cls, v: Any) -> int:
        v = int(v)
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v


class TransformationConfig(BaseModel):
    """Transformation pipeline configuration for CLI."""

    enable_bbox_clipping: bool = Field(default=True, description="Enable bbox clipping")
    bbox_clipping: Optional[BboxClippingConfig] = Field(
        default=None, description="Bbox clipping config"
    )

    enable_augmentation: bool = Field(default=False, description="Enable augmentation")
    augmentation: Optional[AugmentationConfig] = Field(
        default=None, description="Augmentation config"
    )

    enable_tiling: bool = Field(default=False, description="Enable tiling")
    tiling: Optional[TilingConfig] = Field(default=None, description="Tiling config")
