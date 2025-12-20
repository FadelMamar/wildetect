"""Common models and utilities for the API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaseRequest(BaseModel):
    """Base request model with common fields."""
    debug: bool = Field(default=False, description="Enable debug mode")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = Field(description="Operation success status")
    message: str = Field(default="",description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class JobResponse(BaseResponse):
    """Response model for job-based operations."""
    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Job status")
    progress: Optional[float] = Field(default=None, description="Job progress (0-1)")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")


class ErrorResponse(BaseResponse):
    """Error response model."""
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
