"""Shared validation utilities for WildTrain API and CLI modules."""

from pathlib import Path
from typing import Dict, Any, Type, TypeVar
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError
from .config_types import ConfigType

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigFileNotFoundError(Exception):
    """Raised when configuration file is not found."""
    pass


class ConfigParseError(Exception):
    """Raised when configuration file cannot be parsed."""
    pass


T = TypeVar('T', bound=BaseModel)


def validate_config_file(config_path: Path, config_type: ConfigType) -> bool:
    """
    Validate a configuration file using the shared ConfigLoader.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ConfigFileNotFoundError: If config file doesn't exist
        ConfigParseError: If config file cannot be parsed
        ConfigValidationError: If config fails validation
    """
    # Import here to avoid circular import
    from ..cli.config_loader import ConfigLoader
    return ConfigLoader.validate_config_file(config_path, config_type)
