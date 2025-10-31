"""
Health check endpoints.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends

from ..dependencies import get_api_config
from ..models.responses import ErrorResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", operation_id="health_check")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "wildata-api",
    }


@router.get("/detailed", operation_id="detailed_health_check")
async def detailed_health_check(config=Depends(get_api_config)):
    """Detailed health check with system information."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "wildata-api",
        "version": "0.1.0",
        "config": {
            "host": config.host,
            "port": config.port,
            "debug": config.debug,
            "job_queue_size": config.job_queue_size,
        },
    }
