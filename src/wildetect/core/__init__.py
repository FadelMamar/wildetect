"""
Core detection functionality for WildDetect.
"""

from .registry import Detector, ModelRegistry
from .factory import DetectorFactory
from .metrics import ModelMetrics, MetricsTracker
from .data import Detection

# Register available detectors
from .detectors.yolo_detector import YOLODetector

# Register the YOLO detector
ModelRegistry.register_model("yolo", YOLODetector)

__all__ = [
    "Detector",
    "ModelRegistry", 
    "DetectorFactory",
    "ModelMetrics",
    "MetricsTracker",
    "Detection",
    "YOLODetector"
] 