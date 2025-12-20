"""Error handling utilities for the WildTrain API."""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WildTrainAPIException(Exception):
    """Base exception for WildTrain API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(WildTrainAPIException):
    """Exception for validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class FileNotFoundAPIError(WildTrainAPIException):
    """Exception for file not found errors."""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"File not found: {file_path}",
            status_code=404,
            error_code="FILE_NOT_FOUND"
        )


class JobNotFoundError(WildTrainAPIException):
    """Exception for job not found errors."""
    
    def __init__(self, job_id: str):
        super().__init__(
            message=f"Job not found: {job_id}",
            status_code=404,
            error_code="JOB_NOT_FOUND"
        )


class JobExecutionError(WildTrainAPIException):
    """Exception for job execution errors."""
    
    def __init__(self, job_id: str, error: str):
        super().__init__(
            message=f"Job execution failed: {error}",
            status_code=500,
            error_code="JOB_EXECUTION_ERROR",
            details={"job_id": job_id, "error": error}
        )


class ConfigurationError(WildTrainAPIException):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


async def wildtrain_exception_handler(request: Request, exc: WildTrainAPIException) -> JSONResponse:
    """Exception handler for WildTrain API exceptions."""
    
    logger.error(
        f"WildTrain API Exception: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    response_content = {
        "success": False,
        "message": exc.message,
        "error_code": exc.error_code,
        "details": exc.details
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content
    )


def handle_validation_error(error: Exception) -> ValidationError:
    """Convert validation errors to WildTrain API exceptions."""
    if hasattr(error, 'errors'):
        # Pydantic validation error
        details = {"validation_errors": getattr(error, 'errors')()}
        message = "Validation error"
    else:
        details = None
        message = str(error)
    
    return ValidationError(message=message, details=details)


def handle_file_error(error: Exception, file_path: str) -> "FileNotFoundAPIError":
    """Convert file errors to WildTrain API exceptions."""
    return FileNotFoundAPIError(file_path=file_path)
