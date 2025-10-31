"""CLI module for WildTrain using Typer and Rich."""

from .cli import app
from ..shared.validation import validate_config_file

# Import individual commands for backward compatibility
from .commands.train import classifier as train_classifier, detector as train_detector
from .commands.evaluate import classifier as evaluate_classifier, detector as evaluate_detector
from .commands.pipeline import classification as run_classification_pipeline, detection as run_detection_pipeline
from .commands.visualize import classifier_predictions as visualize_classifier_predictions, detector_predictions as visualize_detector_predictions
from .commands.dataset import stats as get_dataset_stats
from .commands.config import validate as validate_config, template as show_config_template

__all__ = [
    "app",
    # Backward compatibility exports
    "train_classifier",
    "train_detector", 
    "get_dataset_stats",
    "run_detection_pipeline",
    "run_classification_pipeline", 
    "visualize_classifier_predictions",
    "visualize_detector_predictions",
    "evaluate_detector",
    "evaluate_classifier", 
    "show_config_template",
    "validate_config",
    "validate_config_file"
]