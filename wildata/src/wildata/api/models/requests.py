"""
Request models for API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...cli.models import (
    BulkCreateROIDatasetConfig,
    BulkImportDatasetConfig,
    ExifGPSUpdateConfig,
    ImportDatasetConfig,
    ROIDatasetConfig,
)


class ImportDatasetRequest(ImportDatasetConfig):
    """Request model for importing a dataset - inherits from CLI config."""

    class Config:
        # Allow extra fields for API-specific extensions
        extra = "allow"


class BulkImportRequest(BulkImportDatasetConfig):
    """Request model for bulk importing datasets - inherits from CLI config."""

    class Config:
        # Allow extra fields for API-specific extensions
        extra = "allow"


class CreateROIRequest(ROIDatasetConfig):
    """Request model for creating ROI datasets - inherits from CLI config."""

    class Config:
        # Allow extra fields for API-specific extensions
        extra = "allow"


class BulkCreateROIRequest(BulkCreateROIDatasetConfig):
    """Request model for bulk creating ROI datasets - inherits from CLI config."""

    class Config:
        # Allow extra fields for API-specific extensions
        extra = "allow"


class UpdateGPSRequest(ExifGPSUpdateConfig):
    """Request model for updating GPS data - inherits from CLI config."""

    class Config:
        # Allow extra fields for API-specific extensions
        extra = "allow"


class VisualizeRequest(BaseModel):
    """Request model for visualization endpoints."""

    dataset_name: str = Field(..., description="Name for the FiftyOne dataset")
    root_data_directory: Optional[str] = Field(None, description="Root data directory")
    split: str = Field(default="train", description="Dataset split (train/val/test)")
    load_as_single_class: bool = Field(
        default=False, description="Load as single class"
    )
    background_class_name: str = Field(
        default="background", description="Background class name"
    )
    single_class_name: str = Field(default="wildlife", description="Single class name")
    keep_classes: Optional[List[str]] = Field(None, description="Classes to keep")
    discard_classes: Optional[List[str]] = Field(None, description="Classes to discard")
