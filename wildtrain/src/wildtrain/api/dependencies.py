"""Shared dependencies and utilities for the WildTrain API."""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
from pydantic import Field

logger = logging.getLogger(__name__)


class APISettings(BaseSettings):
    """API settings and configuration."""
    
    # API Configuration
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8442, description="API port")
    
    # File Storage
    upload_dir: Path = Field(default=Path("./uploads"), description="Upload directory")
    results_dir: Path = Field(default=Path("./results"), description="Results directory")
    logs_dir: Path = Field(default=Path("./logs"), description="Logs directory")
    
    # Background Tasks
    max_workers: int = Field(default=4, description="Maximum background workers")
    job_timeout: int = Field(default=3600, description="Job timeout in seconds")
    
    # Security
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_prefix = "WILDTRAIN_API_"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> APISettings:
    """Get API settings singleton."""
    return APISettings()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"wildtrain.api.{name}")


def validate_file_path(file_path: Path, must_exist: bool = True) -> Path:
    """Validate a file path."""
    if must_exist and not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
    # Security check: ensure path is within allowed directories
    settings = get_settings()
    allowed_dirs = [settings.upload_dir, settings.results_dir, settings.logs_dir]
    
    try:
        file_path.resolve().relative_to(Path.cwd())
    except ValueError:
        raise ValueError(f"File path is outside allowed directories: {file_path}")
    
    return file_path


def get_job_logger(job_id: str) -> logging.Logger:
    """Get a logger for a specific job."""
    logger = get_logger(f"job.{job_id}")
    return logger
