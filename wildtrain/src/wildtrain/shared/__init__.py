"""Shared utilities and types for WildTrain API and CLI modules."""

from .config_types import ConfigType
from .validation import validate_config_file

__all__ = ["ConfigType", "validate_config_file"]
