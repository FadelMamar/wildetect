"""Training-related CLI commands."""

import typer
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from ..config_loader import ConfigLoader
from ...trainers.classification_trainer import ClassifierTrainer
from ...trainers.detection_trainer import UltralyticsDetectionTrainer
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console, setup_logging, log_file_path

train_app = typer.Typer(name="train", help="Training commands")


@train_app.command()
def classifier(
    config: Path = typer.Option("","--config", "-c", help="Path to training configuration file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of training")
) -> None:
    """Train a classification model."""
    if template:
        console.print(f"[bold green]Showing default classification configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION)
            console.print(f"\n[bold blue]Default classification configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Training classifier with config:[/bold green] {config}")
    log_file = log_file_path("train_classifier")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_classification_config(config)
        console.print(f"[bold green]✓[/bold green] Configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        ClassifierTrainer(DictConfig(cfg)).run()
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Training failed: {str(e)}")
        raise typer.Exit(1)


@train_app.command()
def detector(
    config: Path = typer.Option("","--config", "-c", help="Path to training configuration file"),
    template: bool = typer.Option(False, "--template", "-t", help="Show default configuration template instead of training")
) -> None:
    """Train a detection model."""
    if template:
        console.print(f"[bold green]Showing default detection configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.DETECTION)
            console.print(f"\n[bold blue]Default detection configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Training detector with config:[/bold green] {config}")
    log_file = log_file_path("train_detector")
    setup_logging(log_file=log_file)

    try:
        # Load and validate configuration using Pydantic
        validated_config = ConfigLoader.load_detection_config(config)
        console.print(f"[bold green]✓[/bold green] Configuration validated successfully")
        
        # Convert validated config back to DictConfig for backward compatibility
        cfg = OmegaConf.create(validated_config.model_dump())
        console.print("cfg:",cfg)
        
        UltralyticsDetectionTrainer(DictConfig(cfg)).run()
        
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] Configuration error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Training failed: {str(e)}")
        raise typer.Exit(1)
