"""Training service for integrating the harness with the API."""

import logging
from typing import Any, Dict

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
        debug: bool = False,
    ):
        """Train the task named in the experiment document."""
        try:
            experiment = experiment_from_mapping(config)
            logger.info("Starting %s training", experiment.task)
            return run_experiment(experiment, debug=debug)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            raise

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
