"""CLI module for WildTrain using Typer and Rich."""

from ..shared.validation import validate_config_file
from .cli import app
from .commands.config import template as show_config_template
from .commands.config import validate as validate_config
from .commands.dataset import stats as get_dataset_stats
from .commands.evaluate import evaluate as evaluate_document
from .commands.visualize import visualize as visualize_document

__all__ = [
    "app",
    "get_dataset_stats",
    "evaluate_document",
    "visualize_document",
    "show_config_template",
    "validate_config",
    "validate_config_file",
]
