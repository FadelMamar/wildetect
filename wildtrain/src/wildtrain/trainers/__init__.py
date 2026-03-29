"""
Trainers module for WildTrain.

This module provides trainer classes for different types of computer vision tasks.
"""

from .base import ModelTrainer
from .classification_trainer import ClassifierTrainer
from .detection_trainer import UltralyticsDetectionTrainer

__all__ = [
    "ModelTrainer",
    "ClassifierTrainer",
    "UltralyticsDetectionTrainer",
]
