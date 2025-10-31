"""Pipeline-related CLI commands."""

import traceback
import typer
from pathlib import Path
from omegaconf import OmegaConf

from ..config_loader import ConfigLoader
from ...pipeline.detection_pipeline import DetectionPipeline
from ...pipeline.classification_pipeline import ClassificationPipeline
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

pipeline_app = typer.Typer(name="pipeline", help="Pipeline commands")


@pipeline_app.command()
def detection(
    config: Path = typer.Option(None,"--config", "-c", help="Path to unified detection pipeline YAML config"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of running pipeline")
) -> None:
    """Run the full detection pipeline (train + eval) for object detection."""
    if template:
        console.print(f"[bold green]Showing default detection pipeline configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.DETECTION_PIPELINE)
            console.print(f"\n[bold blue]Default detection pipeline configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running detection pipeline with config:[/bold green] {config}")

    log_file = log_file_path("run_detection_pipeline")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_pipeline_config(config, pipeline_type="detection")
        console.print(f"[bold green]✓[/bold green] Detection pipeline configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        pipeline = DetectionPipeline(str(config))
        results = pipeline.run()
        console.print("\n[bold blue]Detection pipeline completed. Evaluation results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Pipeline failed: {traceback.format_exc()}")
        raise typer.Exit(1)


@pipeline_app.command()
def classification(
    config: Path = typer.Option(None,"--config", "-c", help="Path to unified classification pipeline YAML config"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of running pipeline")
) -> None:
    """Run the full classification pipeline (train + eval) for image classification."""
    if template:
        console.print(f"[bold green]Showing default classification pipeline configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_PIPELINE)
            console.print(f"\n[bold blue]Default classification pipeline configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Running classification pipeline with config:[/bold green] {config}")

    log_file = log_file_path("run_classification_pipeline")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_pipeline_config(config, pipeline_type="classification")
        console.print(f"[bold green]✓[/bold green] Classification pipeline configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        pipeline = ClassificationPipeline(str(config))
        results = pipeline.run()
        console.print("\n[bold blue]Classification pipeline completed. Evaluation results:[/bold blue]")
        console.print(results)
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Pipeline failed: {traceback.format_exc()}")
        raise typer.Exit(1)
