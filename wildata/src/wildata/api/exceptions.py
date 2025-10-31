"""
Custom API exceptions.
"""

from typing import Any, Dict, Optional


class WildDataAPIException(Exception):
    """Base exception for WildData API."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(WildDataAPIException):
    """Validation error exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message, status_code=400, error_code="VALIDATION_ERROR", details=details
        )


class NotFoundError(WildDataAPIException):
    """Resource not found exception."""

    def __init__(self, message: str, resource_type: Optional[str] = None):
        details = {"resource_type": resource_type} if resource_type else {}
        super().__init__(
            message, status_code=404, error_code="NOT_FOUND", details=details
        )


class ConflictError(WildDataAPIException):
    """Resource conflict exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message, status_code=409, error_code="CONFLICT", details=details
        )


class JobError(WildDataAPIException):
    """Job-related exception."""

    def __init__(self, message: str, job_id: Optional[str] = None):
        details = {"job_id": job_id} if job_id else {}
        super().__init__(
            message, status_code=500, error_code="JOB_ERROR", details=details
        )


class FileUploadError(WildDataAPIException):
    """File upload exception."""

    def __init__(self, message: str, file_name: Optional[str] = None):
        details = {"file_name": file_name} if file_name else {}
        super().__init__(
            message, status_code=400, error_code="FILE_UPLOAD_ERROR", details=details
        )


class DatasetError(WildDataAPIException):
    """Dataset operation exception."""

    def __init__(self, message: str, dataset_name: Optional[str] = None):
        details = {"dataset_name": dataset_name} if dataset_name else {}
        super().__init__(
            message, status_code=500, error_code="DATASET_ERROR", details=details
        )
