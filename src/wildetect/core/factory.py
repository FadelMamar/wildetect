"""
Factory for creating detector instances.
"""

import logging
from typing import Any, Dict, Optional

from .config import PredictionConfig
from .metrics import ModelMetrics
from .registry import Detector, ModelRegistry

logger = logging.getLogger(__name__)


class DetectorFactory:
    """Factory for creating detector instances."""

    @staticmethod
    def create_detector(
        config: PredictionConfig,
    ) -> Detector:
        """Create detector instance with configuration.

        Args:
            model_type: Type of detector to create
            config: Prediction configuration
            model_path: Path to model weights
            device: Device to run inference on
            **kwargs: Additional configuration parameters

        Returns:
            Configured detector instance

        Raises:
            ValueError: If model type is not supported
        """

        # Create detector instance
        detector = ModelRegistry.create_detector(config)

        # Load the model
        try:
            detector.load_model()
            logger.info(f"Successfully created and loaded {config.model_type} detector")
        except Exception as e:
            logger.error(f"Failed to load {config.model_type} detector: {e}")
            raise

        return detector

    @staticmethod
    def create_detector_with_metrics(
        config: PredictionConfig,
    ) -> tuple[Detector, ModelMetrics]:
        """Create detector with metrics tracking.

        Args:
            model_type: Type of detector to create
            config: Prediction configuration
            **kwargs: Additional configuration parameters

        Returns:
            Tuple of (detector, metrics)
        """
        detector = DetectorFactory.create_detector(config)

        # Import here to avoid circular imports

        metrics = ModelMetrics()

        # Add metrics to detector for tracking
        detector.metrics = metrics

        return detector, metrics

    @staticmethod
    def list_available_models() -> list[str]:
        """Get list of available model types.

        Returns:
            List of registered model names
        """
        return ModelRegistry.list_available_models()

    @staticmethod
    def get_model_info(model_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model type.

        Args:
            model_type: Name of the model type

        Returns:
            Model information dictionary or None if not found
        """
        detector_class = ModelRegistry.get_detector(model_type)
        if detector_class is None:
            return None

        return {
            "name": model_type,
            "class": detector_class.__name__,
            "module": detector_class.__module__,
            "doc": detector_class.__doc__,
        }
