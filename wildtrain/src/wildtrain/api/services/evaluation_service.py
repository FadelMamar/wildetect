"""Evaluation template helpers. Evaluation jobs go through the harness."""

import logging

from ...cli.config_loader import ConfigLoader
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class EvaluationService:
    """Template helpers. ``POST /evaluation/`` dispatches the harness task."""

    @staticmethod
    def generate_classification_eval_template() -> str:
        """Generate a classification evaluation template."""
        try:
            template = ConfigLoader.generate_default_config(
                ConfigType.CLASSIFICATION_EVAL
            )
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
