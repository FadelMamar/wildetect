"""Evaluation-related CLI commands."""

import traceback
import typer
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf
import traceback

from ...shared.config_loader import ConfigLoader
from ...evaluators.ultralytics import UltralyticsEvaluator
from ...evaluators.classification import ClassificationEvaluator
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

evaluate_app = typer.Typer(name="evaluate", help="Evaluation commands")


@evaluate_app.command()
def classifier(
    config: Path = typer.Option("","--config", "-c", help="Path to classification evaluation YAML config file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
) -> Dict[str, Any]:
    """Evaluate a classifier using a YAML config file."""
    
    console.print(f"[bold green]Running classifier evaluation with config:[/bold green] {config}")
    log_file = log_file_path("evaluate_classifier")
    setup_logging(log_file=log_file)
    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_eval_config(config)
        console.print(f"[bold green]✓[/bold green] Classification evaluation configuration validated successfully")
        
        console.print("cfg:",validated_config)
        
        evaluator = ClassificationEvaluator(config=validated_config)
        results = evaluator.evaluate(debug=debug)
        console.print("\n[bold blue]Classifier Evaluation Results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Evaluation failed: {traceback.format_exc()}")
        raise typer.Exit(1)
    
    return results


@evaluate_app.command()
def detector(
    config: Path = typer.Option("","--config", "-c", help="Path to YOLO evaluation YAML config file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
) -> Dict[str, Any]:
    """Evaluate a YOLO model using a YAML config file."""

    log_file = log_file_path("evaluate_detector")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        cfg = ConfigLoader.load_detection_eval_config(config)
        console.print(f"[bold green]✓[/bold green] Detection evaluation configuration validated successfully")        
        console.print("cfg:",cfg)
        
        evaluator = UltralyticsEvaluator(config=cfg)
        results = evaluator.evaluate(debug=debug)
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Evaluation failed: {traceback.format_exc()}")
        raise typer.Exit(1)
    
    return results
