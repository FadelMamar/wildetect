"""Training service for integrating the harness with the API."""

import logging
from typing import Any, Dict, Optional

from ...harness.config import experiment_from_mapping
from ...harness.runner import run_experiment
from ...shared.config_loader import ConfigLoader
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for handling training operations."""

    @staticmethod
    def train_experiment(
        config: Dict[str, Any],
        task: Optional[str] = None,
        debug: bool = False,
    ):
        """Train whichever task the document names, or ``task`` when it does not."""
        try:
            experiment = experiment_from_mapping(config, task=task)
            logger.info("Starting %s training", experiment.task)
            return run_experiment(experiment, debug=debug)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            raise

    @staticmethod
    def train_classifier(config: Dict[str, Any], debug: bool = False):
        """Train a classification model. Flat YAML is still accepted."""
        return TrainingService.train_experiment(
            config, task="classification", debug=debug
        )

    @staticmethod
    def train_detector(config: Dict[str, Any], debug: bool = False):
        """Train a detection model. Flat YAML is still accepted."""
        return TrainingService.train_experiment(config, task="detection", debug=debug)

    @staticmethod
    def generate_classification_train_template() -> str:
        """Generate a classification training template."""
        try:
            return ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION)
        except Exception as exc:
            logger.error("Failed to generate classification training template: %s", exc)
            raise

    @staticmethod
    def generate_detection_train_template() -> str:
        """Generate a detection training template."""
        try:
            return ConfigLoader.generate_default_config(ConfigType.DETECTION)
        except Exception as exc:
            logger.error("Failed to generate detection training template: %s", exc)
            raise
