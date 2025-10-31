"""
Response models for API endpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .jobs import JobResult, JobStatus


class ErrorResponse(BaseModel):
    """Standard error response model."""

    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(), description="Error timestamp"
    )


class JobStatusResponse(BaseModel):
    """Job status response model."""

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(..., description="Job progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    result: Optional[JobResult] = Field(None, description="Job result")
    job_type: str = Field(..., description="Type of job")


class ImportDatasetResponse(BaseModel):
    """Response model for dataset import."""

    success: bool = Field(..., description="Whether the import was successful")
    dataset_name: str = Field(..., description="Name of the imported dataset")
    job_id: Optional[str] = Field(None, description="Background job ID if async")
    message: Optional[str] = Field(None, description="Result message")
    dataset_info_path: Optional[str] = Field(None, description="Path to dataset info")
    framework_paths: Optional[Dict[str, str]] = Field(
        None, description="Framework-specific paths"
    )
    processing_mode: Optional[str] = Field(None, description="Processing mode used")
    dvc_tracked: Optional[bool] = Field(
        None, description="Whether dataset is tracked with DVC"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class BulkImportResponse(BaseModel):
    """Response model for bulk dataset import."""

    success: bool = Field(..., description="Whether the bulk import was successful")
    job_id: str = Field(..., description="Background job ID")
    total_datasets: int = Field(..., description="Total number of datasets to import")
    message: Optional[str] = Field(None, description="Result message")
    error: Optional[str] = Field(None, description="Error message if failed")


class CreateROIResponse(BaseModel):
    """Response model for ROI dataset creation."""

    success: bool = Field(..., description="Whether the ROI creation was successful")
    dataset_name: str = Field(..., description="Name of the created ROI dataset")
    job_id: Optional[str] = Field(None, description="Background job ID if async")
    message: Optional[str] = Field(None, description="Result message")
    output_path: Optional[str] = Field(None, description="Output path for ROI dataset")
    error: Optional[str] = Field(None, description="Error message if failed")


class BulkCreateROIResponse(BaseModel):
    """Response model for bulk ROI dataset creation."""

    success: bool = Field(
        ..., description="Whether the bulk ROI creation was successful"
    )
    job_id: str = Field(..., description="Background job ID")
    total_datasets: int = Field(
        ..., description="Total number of ROI datasets to create"
    )
    message: Optional[str] = Field(None, description="Result message")
    error: Optional[str] = Field(None, description="Error message if failed")


class UpdateGPSResponse(BaseModel):
    """Response model for GPS update."""

    success: bool = Field(..., description="Whether the GPS update was successful")
    job_id: Optional[str] = Field(None, description="Background job ID if async")
    message: Optional[str] = Field(None, description="Result message")
    updated_images_count: Optional[int] = Field(
        None, description="Number of images updated"
    )
    output_dir: Optional[str] = Field(None, description="Output directory")
    error: Optional[str] = Field(None, description="Error message if failed")


class DatasetInfo(BaseModel):
    """Dataset information model."""

    dataset_name: str = Field(..., description="Dataset name")
    total_images: Optional[int] = Field(None, description="Total number of images")
    total_annotations: Optional[int] = Field(
        None, description="Total number of annotations"
    )
    splits: Optional[List[str]] = Field(None, description="Available splits")
    created_at: Optional[datetime] = Field(
        None, description="Dataset creation timestamp"
    )
    last_modified: Optional[datetime] = Field(
        None, description="Last modification timestamp"
    )


class DatasetListResponse(BaseModel):
    """Response model for dataset listing."""

    datasets: List[DatasetInfo] = Field(..., description="List of datasets")
    total_count: int = Field(..., description="Total number of datasets")
    root_directory: str = Field(..., description="Root directory for datasets")


class VisualizationResponse(BaseModel):
    """Response model for dataset visualization."""

    success: bool = Field(..., description="Whether the visualization was successful")
    job_id: Optional[str] = Field(None, description="Background job ID if async")
    message: Optional[str] = Field(None, description="Result message")
    dataset_name: str = Field(..., description="Name of the dataset")
    split: str = Field(..., description="Dataset split")
    visualization_url: Optional[str] = Field(None, description="URL to visualization")
    error: Optional[str] = Field(None, description="Error message if failed")
