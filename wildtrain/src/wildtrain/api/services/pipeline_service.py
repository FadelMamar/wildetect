"""Pipeline service for integrating CLI functionality with the API."""

import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from omegaconf import OmegaConf
from ...cli.config_loader import ConfigLoader
from ...cli.models import ClassificationPipelineConfig, DetectionPipelineConfig
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for handling pipeline operations."""

    @staticmethod
    def run_classification_pipeline(config: ClassificationPipelineConfig) -> Dict[str, Any]:
        """Run classification pipeline using the CLI."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.pipeline import classification
            classification(config=Path(temp_config_path))

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Classification pipeline completed successfully")
            return 

        except Exception as e:
            logger.error(f"Classification pipeline failed: {e}")
            raise

    @staticmethod
    def run_detection_pipeline(config: DetectionPipelineConfig) -> Dict[str, Any]:
        """Run detection pipeline using the CLI."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.pipeline import detection
            detection(config=Path(temp_config_path))

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Detection pipeline completed successfully")
            return 

        except Exception as e:
            logger.error(f"Detection pipeline failed: {e}")
            raise

    @staticmethod
    def generate_classification_pipeline_template() -> str:
        """Generate a classification pipeline template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_PIPELINE)
            return template
        except Exception as e:
            logger.error(f"Failed to generate classification pipeline template: {e}")
            raise

    @staticmethod
    def generate_detection_pipeline_template() -> str:
        """Generate a detection pipeline template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.DETECTION_PIPELINE)
            return template
        except Exception as e:
            logger.error(f"Failed to generate detection pipeline template: {e}")
            raise
