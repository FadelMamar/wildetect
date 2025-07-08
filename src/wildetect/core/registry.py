"""
Model registry for managing different detector types.
"""

import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from ..utils.utils import load_registered_model
from .config import PredictionConfig
from .data.detection import Detection

logger = logging.getLogger(__name__)


class Detector(ABC):
    """Abstract base class for all detection models."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.class_names = []
        self._is_loaded = False
        self.metrics = None  # Will be set by factory if needed
        self.metadata = dict()

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def _predict_batch(self, batch: torch.Tensor) -> List[List[Detection]]:
        pass

    @abstractmethod
    def predict(self, batch: torch.Tensor) -> List[List[Detection]]:
        """Run prediction on an image."""
        pass

    def warmup(self, image_size=(640, 640)) -> None:
        """Warm up the model with a dummy prediction."""
        try:
            dummy_batch = torch.randn(1, 3, *image_size)
            self.predict(dummy_batch)
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names

    def load_from_mlflow(self) -> Tuple[Any, Dict[str, str]]:
        """Load the model from MLflow."""
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", None)
        mlflow_model_alias = os.environ.get("MLFLOW_MODEL_ALIAS", None)
        try:
            model, metadata = load_registered_model(
                name=mlflow_model_name,
                alias=mlflow_model_alias,
                load_unwrapped=True,
            )
            logger.info(
                f"Loaded YOLO model from MLflow: {mlflow_model_name}/{mlflow_model_alias}"
            )
        except Exception:
            logger.error(f"Error loading model from MLflow: {traceback.format_exc()}")
            return None, dict()

        return model, metadata

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


class ModelRegistry:
    """Registry for available model types."""

    _detectors: Dict[str, Type[Detector]] = {}

    @classmethod
    def register_model(cls, name: str, detector_class: Type[Detector]) -> None:
        """Register a new model type.

        Args:
            name: Name of the model type
            detector_class: Detector class to register
        """
        if not issubclass(detector_class, Detector):
            raise ValueError(f"Detector class must inherit from Detector base class")

        cls._detectors[name] = detector_class
        logger.info(f"Registered detector: {name}")

    @classmethod
    def get_detector(cls, name: str) -> Optional[Type[Detector]]:
        """Get detector class by name.

        Args:
            name: Name of the detector type

        Returns:
            Detector class or None if not found
        """
        return cls._detectors.get(name)

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available model types.

        Returns:
            List of registered model names
        """
        return list(cls._detectors.keys())

    @classmethod
    def create_detector(cls, config: PredictionConfig) -> Detector:
        """Create detector instance by name.

        Args:
            name: Name of the detector type
            config: Configuration for the detector

        Returns:
            Detector instance

        Raises:
            ValueError: If detector type is not registered
        """
        detector_class = cls.get_detector(config.model_type)
        if detector_class is None:
            available = cls.list_available_models()
            raise ValueError(
                f"Unknown detector type '{config.model_type}'. Available: {available}"
            )

        return detector_class(config)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered detectors (mainly for testing)."""
        cls._detectors.clear()
