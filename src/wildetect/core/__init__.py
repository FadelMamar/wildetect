"""
Core detection functionality for WildDetect.
"""

from .data import Detection

# Register available detectors
from .detectors.yolo_detector import YOLODetector
from .factory import DetectorFactory
from .metrics import MetricsTracker, ModelMetrics
from .registry import Detector, ModelRegistry

# Register the YOLO detector
ModelRegistry.register_model("yolo", YOLODetector)

__all__ = [
    "Detector",
    "ModelRegistry",
    "DetectorFactory",
    "ModelMetrics",
    "MetricsTracker",
    "Detection",
    "YOLODetector",
]
