"""Evaluation-related CLI commands."""

import traceback
import typer
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf
import traceback

from ..config_loader import ConfigLoader
from ...evaluators.ultralytics import UltralyticsEvaluator
from ...evaluators.classification import ClassificationEvaluator
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

evaluate_app = typer.Typer(name="evaluate", help="Evaluation commands")


@evaluate_app.command()
def classifier(
    config: Path = typer.Option("","--config", "-c", help="Path to classification evaluation YAML config file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of evaluation"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
) -> Dict[str, Any]:
    """Evaluate a classifier using a YAML config file."""
    if template:
        console.print(f"[bold green]Showing default classification evaluation configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_EVAL)
            console.print(f"\n[bold blue]Default classification evaluation configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return {}
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {traceback.format_exc()}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running classifier evaluation with config:[/bold green] {config}")

    log_file = log_file_path("evaluate_classifier")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_eval_config(config)
        console.print(f"[bold green]✓[/bold green] Classification evaluation configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        # Use model_dump with exclude_none=False to preserve all fields including defaults
        cfg = OmegaConf.create(validated_config.model_dump(exclude_none=False))
        console.print("cfg:",cfg)
        
        evaluator = ClassificationEvaluator(config=cfg)
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
    model_type: str = typer.Option("yolo", "--type", "-t", help="Type of detector to evaluate (yolo, yolo_v8, yolo_v11)"),
    template: bool = typer.Option(False, "--template", help="Show default configuration template instead of evaluation"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
) -> Dict[str, Any]:
    """Evaluate a YOLO model using a YAML config file."""
    if template:
        console.print(f"[bold green]Showing default detection evaluation configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.DETECTION_EVAL)
            console.print(f"\n[bold blue]Default detection evaluation configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return {}
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {traceback.format_exc()}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running {model_type} evaluation with config:[/bold green] {config}")

    log_file = log_file_path("evaluate_detector")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_detection_eval_config(config)
        console.print(f"[bold green]✓[/bold green] Detection evaluation configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        # Use model_dump with exclude_none=False to preserve all fields including defaults
        cfg = OmegaConf.create(validated_config.model_dump(exclude_none=False))
        console.print("cfg:",cfg)
        
        if model_type == "yolo":
            evaluator = UltralyticsEvaluator(config=cfg)
        else:
            raise ValueError(f"Invalid detector type: {model_type}")
        results = evaluator.evaluate(debug=debug)
        console.print(f"\n[bold blue]{model_type} Evaluation Results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Evaluation failed: {traceback.format_exc()}")
        raise typer.Exit(1)
    
    return results
