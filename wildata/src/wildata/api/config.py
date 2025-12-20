"""
API configuration settings.
"""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """API configuration settings."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8441, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],  # TODO: change to specific origins in production
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(
        default=True, description="Allow CORS credentials"
    )

    # Rate limiting
    # rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")

    # Background job settings
    job_queue_size: int = Field(default=100, description="Background job queue size")
    job_timeout: int = Field(
        default=3600, description="Job timeout in seconds"
    )  # 1 hour

    # Database settings (for job persistence)
    database_url: Optional[str] = Field(
        default=None, description="Database URL for job persistence"
    )

    # Security settings
    secret_key: str = Field(
        default="your-secret-key-change-in-production", description="Secret key for JWT"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiry in minutes"
    )

    class Config:
        env_prefix = "WILDATA_API_"
        case_sensitive = False


# Global config instance
api_config = APIConfig()
