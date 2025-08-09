"""
Configuration loader for WildDetect CLI commands using OmegaConf and Pydantic.

This module provides utilities for loading YAML configuration files,
merging with command-line overrides, and providing sensible defaults
with Pydantic validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import ROOT
from .config_models import (
    ConfigModel,
    config_model_to_dict,
    create_default_config,
    validate_config_dict,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration files for WildDetect CLI commands using OmegaConf."""

    def __init__(self):
        self.config_cache = {}

    def load_config(
        self,
        command_type: str,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Union[DictConfig, ListConfig]:
        """
        Load configuration for a specific command type.

        Args:
            command_type: The command type (detect, census, visualize)
            config_path: Optional path to custom config file
            overrides: Optional CLI overrides to merge

        Returns:
            OmegaConf DictConfig with merged configuration
        """
        # Load base configuration
        base_config = self._load_base_config(command_type, config_path)

        # Merge with overrides if provided
        if overrides:
            base_config = self._merge_overrides(base_config, overrides)

        # Apply environment variable substitutions
        base_config = self._apply_env_substitutions(base_config)

        return base_config

    def _load_base_config(
        self, command_type: str, config_path: Optional[str] = None
    ) -> Union[DictConfig, ListConfig]:
        """Load base configuration with fallback chain."""

        # 1. Try user-specified config file
        if config_path and Path(config_path).exists():
            logger.info(f"Loading user-specified config: {config_path}")
            return OmegaConf.load(config_path)

        # 2. Try command-specific default config
        default_config_path = ROOT / "config" / f"{command_type}.yaml"
        if default_config_path.exists():
            logger.info(f"Loading default config: {default_config_path}")
            return OmegaConf.load(str(default_config_path))

        # 3. Try global settings
        global_config_path = ROOT / "config" / "settings.yaml"
        if global_config_path.exists():
            logger.info(f"Loading global config: {global_config_path}")
            return OmegaConf.load(str(global_config_path))

        # 4. Use built-in defaults
        logger.info("Using built-in defaults")
        return OmegaConf.create(self._get_builtin_defaults(command_type))

    def _merge_overrides(
        self, base_config: Union[DictConfig, ListConfig], overrides: Dict[str, Any]
    ) -> Union[DictConfig, ListConfig]:
        """Merge CLI overrides with base configuration using OmegaConf."""
        if not overrides:
            return base_config

        # Convert overrides to OmegaConf format
        override_config = OmegaConf.create(overrides)

        # Merge configurations
        merged = OmegaConf.merge(base_config, override_config)
        return merged

    def _apply_env_substitutions(
        self, config: Union[DictConfig, ListConfig]
    ) -> Union[DictConfig, ListConfig]:
        """Apply environment variable substitutions to configuration."""
        # OmegaConf handles environment variable interpolation automatically
        # when using ${ENV_VAR} syntax in YAML files
        return config

    def _get_builtin_defaults(self, command_type: str) -> Dict[str, Any]:
        """Get built-in default configuration for a command type."""
        defaults = {
            "detect": {
                "model": {
                    "path": None,
                    "type": "yolo",
                    "confidence_threshold": 0.2,
                    "device": "auto",
                    "batch_size": 32,
                    "nms_iou": 0.5,
                },
                "processing": {
                    "tile_size": 800,
                    "overlap_ratio": 0.2,
                    "pipeline_type": "single",
                    "queue_size": 3,
                },
                "flight_specs": {
                    "sensor_height": 24.0,
                    "focal_length": 35.0,
                    "flight_height": 180.0,
                },
                "roi_classifier": {
                    "weights": None,
                    "feature_extractor_path": "facebook/dinov2-with-registers-small",
                    "cls_label_map": {0: "groundtruth", 1: "other"},
                    "keep_classes": ["groundtruth"],
                    "cls_imgsz": 128,
                },
                "inference_service": {"url": None},
                "profiling": {
                    "enable": False,
                    "memory_profile": False,
                    "line_profile": False,
                    "gpu_profile": False,
                },
                "output": {
                    "directory": "results",
                    "save_results": True,
                    "export_to_fiftyone": True,
                    "dataset_name": None,
                },
                "logging": {"verbose": False, "log_file": None},
            },
            "census": {
                "campaign": {
                    "id": "campaign_001",
                    "pilot_name": "Unknown",
                    "target_species": None,
                },
                "detection": {
                    "model": {
                        "path": None,
                        "type": "yolo",
                        "confidence_threshold": 0.2,
                        "device": "auto",
                        "batch_size": 32,
                        "nms_iou": 0.5,
                    },
                    "processing": {
                        "tile_size": 800,
                        "overlap_ratio": 0.2,
                        "pipeline_type": "single",
                        "queue_size": 3,
                    },
                    "flight_specs": {
                        "sensor_height": 24.0,
                        "focal_length": 35.0,
                        "flight_height": 180.0,
                    },
                    "roi_classifier": {
                        "weights": None,
                        "feature_extractor_path": "facebook/dinov2-with-registers-small",
                        "cls_label_map": {0: "groundtruth", 1: "other"},
                        "keep_classes": ["groundtruth"],
                        "cls_imgsz": 96,
                    },
                    "inference_service": {"url": None},
                    "profiling": {
                        "enable": False,
                        "memory_profile": False,
                        "line_profile": False,
                        "gpu_profile": False,
                    },
                },
                "export": {
                    "to_fiftyone": True,
                    "create_map": True,
                    "output_directory": None,
                    "export_to_labelstudio": True,
                },
                "logging": {"verbose": False, "log_file": None},
            },
            "visualize": {
                "geographic": {
                    "create_map": True,
                    "show_confidence": False,
                    "output_directory": "visualizations",
                    "map_type": "folium",
                    "zoom_level": 12,
                    "center_on_data": True,
                },
                "flight_specs": {
                    "sensor_height": 24.0,
                    "focal_length": 35.0,
                    "flight_height": 180.0,
                },
                "visualization": {
                    "show_detections": True,
                    "show_footprints": True,
                    "show_statistics": True,
                    "color_by_confidence": False,
                    "confidence_threshold": 0.2,
                },
                "output": {
                    "format": "html",
                    "include_legend": True,
                    "include_statistics": True,
                    "auto_open": False,
                },
                "logging": {"verbose": False, "log_file": None},
            },
        }

        return defaults.get(command_type, {})

    def validate_config(
        self, config: Union[DictConfig, ListConfig], command_type: str
    ) -> bool:
        """Validate configuration structure and values."""
        try:
            required_sections = {
                "detect": ["model", "processing", "flight_specs"],
                "census": ["campaign", "detection"],
                "visualize": ["geographic", "flight_specs"],
            }

            required = required_sections.get(command_type, [])
            for section in required:
                if not OmegaConf.select(config, section):
                    logger.error(f"Missing required section: {section}")
                    return False

            # Validate specific values
            if command_type in ["detect", "census"]:
                model_config = OmegaConf.select(config, "model")
                if model_config:
                    conf = OmegaConf.select(model_config, "confidence_threshold")
                    if conf is not None and not (0.0 <= conf <= 1.0):
                        logger.error(f"Invalid confidence_threshold: {conf}")
                        return False

                    batch_size = OmegaConf.select(model_config, "batch_size")
                    if batch_size is not None and batch_size <= 0:
                        logger.error(f"Invalid batch_size: {batch_size}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def find_config_file(
        self, command_type: str, specified_path: Optional[str] = None
    ) -> Optional[str]:
        """Find configuration file in order of precedence."""
        search_paths = [
            specified_path,  # User-specified path
            str(ROOT / "config" / f"{command_type}.yaml"),  # Default config
            str(ROOT / "config" / f"{command_type}.yml"),
            str(ROOT / f".{command_type}.yaml"),  # Hidden config in current dir
            str(ROOT / f".{command_type}.yml"),
        ]

        for path in search_paths:
            if path and Path(path).exists():
                return path

        return None

    def create_default_config(
        self, command_type: str, output_path: Optional[str] = None
    ) -> str:
        """Create a default configuration file for a command type."""
        if output_path is None:
            output_path = str(ROOT / "config" / f"{command_type}_default.yaml")

        config = OmegaConf.create(self._get_builtin_defaults(command_type))

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write configuration file
        OmegaConf.save(config, output_path)

        logger.info(f"Created default configuration file: {output_path}")
        return output_path

    def load_config_with_pydantic(
        self,
        command_type: str,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ConfigModel:
        """
        Load configuration with Pydantic validation.

        Args:
            command_type: The command type (detect, census, visualize)
            config_path: Optional path to custom config file
            overrides: Optional CLI overrides to merge

        Returns:
            Validated Pydantic config model
        """
        # Load base configuration using existing method
        base_config = self.load_config(command_type, config_path, overrides)

        # Convert OmegaConf to dictionary
        config_dict = OmegaConf.to_container(base_config, resolve=True)

        # Validate with Pydantic
        validated_config = validate_config_dict(config_dict, command_type)

        return validated_config

    def save_config_model(self, config_model: ConfigModel, output_path: str) -> str:
        """Save a Pydantic config model to YAML file."""
        # Convert to dictionary
        config_dict = config_model_to_dict(config_model)

        # Create OmegaConf object
        config = OmegaConf.create(config_dict)

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        OmegaConf.save(config, output_path)

        logger.info(f"Saved configuration to: {output_path}")
        return output_path

    def create_pydantic_default_config(
        self, command_type: str, output_path: Optional[str] = None
    ) -> str:
        """Create a default configuration file using Pydantic models."""
        if output_path is None:
            output_path = str(ROOT / "config" / f"{command_type}_pydantic_default.yaml")

        # Create default config using Pydantic
        default_config = create_default_config(command_type)

        # Save to file
        return self.save_config_model(default_config, output_path)


# Global config loader instance
config_loader = ConfigLoader()


def load_config_from_yaml(
    config_path: Optional[str],
    command_type: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Union[DictConfig, ListConfig]:
    """Load configuration from YAML file with fallbacks using OmegaConf."""
    return config_loader.load_config(command_type, config_path, overrides)


def validate_config_file(config_path: str, command_type: str) -> bool:
    """Validate a configuration file."""
    try:
        config = OmegaConf.load(config_path)
        return config_loader.validate_config(config, command_type)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def find_config_file(
    command_type: str, specified_path: Optional[str] = None
) -> Optional[str]:
    """Find configuration file in order of precedence."""
    return config_loader.find_config_file(command_type, specified_path)


def load_config_with_pydantic(
    command_type: str,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigModel:
    """Load configuration with Pydantic validation."""
    return config_loader.load_config_with_pydantic(command_type, config_path, overrides)


def save_config_model(config_model: ConfigModel, output_path: str) -> str:
    """Save a Pydantic config model to YAML file."""
    return config_loader.save_config_model(config_model, output_path)


def create_pydantic_default_config(
    command_type: str, output_path: Optional[str] = None
) -> str:
    """Create a default configuration file using Pydantic models."""
    return config_loader.create_pydantic_default_config(command_type, output_path)
