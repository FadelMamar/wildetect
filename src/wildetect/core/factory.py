"""
Factory for creating detector instances.
"""

from typing import Optional, Dict, Any
import logging

from .config import PredictionConfig
from .registry import ModelRegistry, Detector

logger = logging.getLogger(__name__)


class DetectorFactory:
    """Factory for creating detector instances."""
    
    @staticmethod
    def create_detector(
        model_type: str,
        config: Optional[PredictionConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
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
        # Create default config if not provided
        if config is None:
            config = PredictionConfig()
        
        # Override config with provided parameters
        if device is not None:
            config.device = device
        
        # Apply additional kwargs to config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create detector instance
        detector = ModelRegistry.create_detector(model_type, config)
        
        # Load the model
        try:
            detector.load_model()
            logger.info(f"Successfully created and loaded {model_type} detector")
        except Exception as e:
            logger.error(f"Failed to load {model_type} detector: {e}")
            raise
        
        return detector
    
    @staticmethod
    def create_detector_with_metrics(
        model_type: str,
        config: Optional[PredictionConfig] = None,
        **kwargs
    ) -> tuple[Detector, 'ModelMetrics']:
        """Create detector with metrics tracking.
        
        Args:
            model_type: Type of detector to create
            config: Prediction configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Tuple of (detector, metrics)
        """
        detector = DetectorFactory.create_detector(model_type, config, **kwargs)
        
        # Import here to avoid circular imports
        from .metrics import ModelMetrics
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
            'name': model_type,
            'class': detector_class.__name__,
            'module': detector_class.__module__,
            'doc': detector_class.__doc__
        } 