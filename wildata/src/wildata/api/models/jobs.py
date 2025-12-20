"""
Background job models.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResult(BaseModel):
    """Job result model."""

    success: bool = Field(..., description="Whether the job completed successfully")
    message: Optional[str] = Field(None, description="Result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")


class BackgroundJob(BaseModel):
    """Background job model."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(
        default=JobStatus.PENDING, description="Current job status"
    )
    job_type: str = Field(..., description="Type of job (e.g., 'import_dataset')")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Job creation timestamp"
    )
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Job progress percentage"
    )
    result: Optional[JobResult] = Field(None, description="Job result")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Job parameters"
    )
    user_id: Optional[str] = Field(None, description="User who created the job")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
