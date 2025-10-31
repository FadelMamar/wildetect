"""
Shared dependencies and utilities for the API.
"""

import asyncio
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

from .config import api_config
from .exceptions import WildDataAPIException

# Security scheme for future authentication
security = HTTPBearer(auto_error=False)


async def get_api_config():
    """Dependency to get API configuration."""
    return api_config


async def verify_token(token: Optional[str] = Depends(security)):
    """Dependency to verify authentication token (placeholder for future auth)."""
    # TODO: Implement actual JWT verification when auth is added
    if api_config.debug:
        return {"user_id": "debug_user", "username": "debug"}

    # For now, allow all requests (no auth required)
    return {"user_id": "anonymous", "username": "anonymous"}


def handle_api_exception(exc: WildDataAPIException):
    """Convert WildDataAPIException to HTTPException."""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
        },
    )


async def get_background_task_semaphore():
    """Dependency to get semaphore for background tasks."""
    # Limit concurrent background tasks
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent background tasks
    return semaphore
