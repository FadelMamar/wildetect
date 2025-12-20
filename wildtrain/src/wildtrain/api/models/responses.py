"""Response models for the WildTrain API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .common import BaseResponse, JobResponse, JobStatus


class TrainingResponse(JobResponse):
    """Response model for training operations."""
    model_path: Optional[str] = Field(default=None, description="Path to trained model")
    logs_path: Optional[str] = Field(default=None, description="Path to training logs")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Training metrics")
    job_id: Optional[str] = Field(default=None, description="Job ID")


class EvaluationResponse(BaseResponse):
    """Response model for evaluation operations."""
    metrics: Dict[str, Any] = Field(description="Evaluation metrics")
    results_path: Optional[str] = Field(default=None, description="Path to evaluation results")
    confusion_matrix: Optional[Dict[str, Any]] = Field(default=None, description="Confusion matrix")
    class_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Per-class metrics")
    job_id: Optional[str] = Field(default=None, description="Job ID")


class PipelineResponse(JobResponse):
    """Response model for pipeline operations."""
    results_dir: Optional[str] = Field(default=None, description="Path to pipeline results")
    training_results: Optional[Dict[str, Any]] = Field(default=None, description="Training results")
    evaluation_results: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation results")


class VisualizationResponse(JobResponse):
    """Response model for visualization operations."""
    visualization_url: Optional[str] = Field(default=None, description="URL to visualization")
    fiftyone_dataset: Optional[str] = Field(default=None, description="FiftyOne dataset name")
    sample_count: Optional[int] = Field(default=None, description="Number of samples processed")


class DatasetStatsResponse(BaseResponse):
    """Response model for dataset statistics."""
    stats: Dict[str, Any] = Field(description="Dataset statistics")
    class_distribution: Dict[str, int] = Field(description="Class distribution")
    mean: List[float] = Field(description="Mean values")
    std: List[float] = Field(description="Standard deviation values")
    total_samples: int = Field(description="Total number of samples")
    split_info: Dict[str, Any] = Field(description="Split information")


class ConfigValidationResponse(BaseResponse):
    """Response model for configuration validation."""
    is_valid: bool = Field(description="Configuration validity")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    config_type: str = Field(description="Configuration type")


class TemplateResponse(BaseResponse):
    """Response model for configuration templates."""
    template: str = Field(description="Configuration template")
    config_type: str = Field(description="Configuration type")


class FileUploadResponse(BaseResponse):
    """Response model for file upload."""
    file_path: str = Field(description="Path to uploaded file")
    file_size: int = Field(description="File size in bytes")
    file_type: str = Field(description="Type of uploaded file")


class JobListResponse(BaseResponse):
    """Response model for job listing."""
    jobs: List[Dict[str, Any]] = Field(description="List of jobs")
    total_count: int = Field(description="Total number of jobs")
    filtered_count: int = Field(description="Number of jobs after filtering")


class JobDetailResponse(JobResponse):
    """Response model for detailed job information."""
    logs: List[Dict[str, Any]] = Field(description="Job logs")
    metadata: Dict[str, Any] = Field(description="Job metadata")
    result_files: List[Dict[str, Any]] = Field(description="Result files")
    created_at: datetime = Field(description="Job creation time")
    started_at: Optional[datetime] = Field(default=None, description="Job start time")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion time")
    duration: Optional[float] = Field(default=None, description="Job duration in seconds")


class HealthResponse(BaseResponse):
    """Response model for health check."""
    status: str = Field(description="Health status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Uptime in seconds")
    active_jobs: int = Field(description="Number of active jobs")
    disk_usage: Dict[str, Any] = Field(description="Disk usage information")
