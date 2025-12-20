"""Shared configuration types for WildTrain API and CLI modules."""

from enum import Enum


class ConfigType(Enum):
    """Configuration types."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    CLASSIFICATION_EVAL = "classification_eval"
    DETECTION_EVAL = "detection_eval"
    CLASSIFICATION_VISUALIZATION = "classification_visualization"
    DETECTION_VISUALIZATION = "detection_visualization"
    DETECTION_PIPELINE = "detection_pipeline"
    CLASSIFICATION_PIPELINE = "classification_pipeline"
    DETECTOR_REGISTRATION = "detector_registration"
    CLASSIFIER_REGISTRATION = "classifier_registration"
    MODEL_REGISTRATION = "model_registration"

