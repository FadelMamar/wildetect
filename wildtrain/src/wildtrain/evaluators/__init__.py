"""Evaluators package for model evaluation."""

from .ultralytics import UltralyticsEvaluator
from .calibrator import DetectionCalibrator

__all__ = [
    "UltralyticsEvaluator",
    "DetectionCalibrator",
]
