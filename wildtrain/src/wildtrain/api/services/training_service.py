"""Training service for integrating CLI functionality with the API."""

import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from omegaconf import OmegaConf
from ...cli.config_loader import ConfigLoader
from ...cli.models import ClassificationConfig, DetectionConfig
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for handling training operations."""

    @staticmethod
    def train_classifier(config: Dict[str, Any]) -> None:
        """Train a classification model using the CLI."""
        try:
            validated_config = ClassificationConfig(**config)
            
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = validated_config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.train import classifier
            classifier(config=Path(temp_config_path), template=False)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Classification training completed successfully")
            return

        except Exception as e:
            logger.error(f"Classification training failed: {e}")
            raise

    @staticmethod
    def train_detector(config: Dict[str, Any]) -> None:
        """Train a detection model using the CLI."""
        try:
            
            validated_config = DetectionConfig(**config)
            
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = validated_config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.train import detector
            detector(config=Path(temp_config_path), template=False)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Detection training completed successfully")
            return

        except Exception as e:
            logger.error(f"Detection training failed: {e}")
            raise

    @staticmethod
    def generate_classification_train_template() -> str:
        """Generate a classification training template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION)
            return template
        except Exception as e:
            logger.error(f"Failed to generate classification training template: {e}")
            raise

    @staticmethod
    def generate_detection_train_template() -> str:
        """Generate a detection training template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.DETECTION)
            return template
        except Exception as e:
            logger.error(f"Failed to generate detection training template: {e}")
            raise
