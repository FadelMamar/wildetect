"""Evaluation service for integrating CLI functionality with the API."""

import tempfile
from pathlib import Path
from typing import Dict, Any
import logging

from omegaconf import OmegaConf
from ...cli.config_loader import ConfigLoader
from ...cli.models import ClassificationEvalConfig, DetectionEvalConfig
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for handling evaluation operations."""

    @staticmethod
    def evaluate_classifier(config: ClassificationEvalConfig, debug: bool) -> Dict[str, Any]:
        """Evaluate a classification model using the CLI evaluator."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = Path(f.name)

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.evaluate import classifier
            results = classifier(config=temp_config_path, template=False, debug=debug)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Classification evaluation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Classification evaluation failed: {e}")
            raise

    @staticmethod
    def evaluate_detector(config: DetectionEvalConfig, model_type: str, debug: bool) -> Dict[str, Any]:
        """Evaluate a detection model using the CLI evaluator."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = Path(f.name)

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.evaluate import detector
            results = detector(config=temp_config_path, template=False, debug=debug, model_type=model_type)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Detection evaluation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Detection evaluation failed: {e}")
            raise

    @staticmethod
    def generate_classification_eval_template() -> str:
        """Generate a classification evaluation template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_EVAL)
            return template
        except Exception as e:
            logger.error(f"Failed to generate classification evaluation template: {e}")
            raise

    @staticmethod
    def generate_detection_eval_template() -> str:
        """Generate a detection evaluation template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.DETECTION_EVAL)
            return template
        except Exception as e:
            logger.error(f"Failed to generate detection evaluation template: {e}")
            raise
