"""Registration-related CLI commands."""

import typer
from pathlib import Path
import traceback

from ..config_loader import ConfigLoader
from ...models.register import ModelRegistrar
from ...shared.config_types import ConfigType
from .utils import console, setup_logging, log_file_path

register_app = typer.Typer(name="register", help="Model registration commands")


@register_app.command()
def classifier(
    config: Path = typer.Option(None, "--config", "-c", help="Path to classifier registration configuration file"),
    weights_path: Path = typer.Option(None, "--weights-path", help="Path to the model checkpoint file"),
    name: str = typer.Option("classifier", "--name", "-n", help="Model name for registration"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size for inference"),
    mlflow_tracking_uri: str = typer.Option("http://localhost:5000", "--mlflow-uri", help="MLflow tracking server URI"),
    template: bool = typer.Option(False, "--template", help="Show default configuration template instead of registration")
) -> None:
    """Register a classification model to MLflow Model Registry."""
    if template:
        console.print(f"[bold green]Showing default classifier registration configuration template...[/bold green]")
        try:
            template_yaml = ConfigLoader.generate_default_config(ConfigType.CLASSIFIER_REGISTRATION)
            console.print(f"\n[bold blue]Default classifier registration configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template_yaml}```")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
            raise typer.Exit(1)
    
    if config is not None:
        # Load configuration from file
        console.print(f"[bold green]Loading classifier registration config from:[/bold green] {config}")
        try:
            validated_config = ConfigLoader.load_classifier_registration_config(config)
            console.print(f"[bold green]✓[/bold green] Classifier registration configuration validated successfully")
            
            # Use config values
            weights_path = Path(validated_config.weights)
            name = validated_config.processing.name
            batch_size = validated_config.processing.batch_size
            mlflow_tracking_uri = validated_config.processing.mlflow_tracking_uri
            
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Configuration error: {traceback.format_exc()}")
            raise typer.Exit(1)
    else:
        # Use command line arguments
        if weights_path is None:
            console.print(f"[bold red]✗[/bold red] Must provide either --config or --weights-path")
            raise typer.Exit(1)
    
    console.print(f"[bold green]Registering classifier model:[/bold green] {weights_path}")
    
    log_file = log_file_path("register_classifier")
    setup_logging(log_file=log_file)
    
    try:
        # Create model registrar
        registrar = ModelRegistrar(mlflow_tracking_uri=mlflow_tracking_uri)
        
        # Register the classifier
        registrar.register_classifier(
            weights=weights_path,
            name=name,
            batch_size=batch_size
        )
        
        console.print(f"[bold green]✓[/bold green] Successfully registered classifier model: {name}")
        console.print(f"[bold blue]Model registered to MLflow tracking URI:[/bold blue] {mlflow_tracking_uri}")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]✗[/bold red] Model weights not found: {traceback.format_exc()}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Registration failed: {traceback.format_exc()}")
        if log_file.exists():
            console.print(f"[dim]Check logs at: {log_file}[/dim]")
        raise typer.Exit(1)


@register_app.command()
def detector(config: Path = typer.Argument(..., help="Path to classifier registration configuration file")):

    log_file = log_file_path("register_detector")
    setup_logging(log_file=log_file)
    
    try:
        # Create model registrar
        registrar = ModelRegistrar()
        # Register the classifier
        registrar.register_detector(
            config_path=config
        )
        cfg = ConfigLoader.load_detector_registration_config(config)
        console.print("Configuration:",cfg)
        console.print(f"[bold green]✓[/bold green] Successfully registered detector model.")
        console.print(f"[bold blue]Model registered to MLflow tracking URI:[/bold blue]")
        

    except Exception:
        console.print(f"[bold red]✗[/bold red] Registration failed: {traceback.format_exc()}")
        if log_file.exists():
            console.print(f"[dim]Check logs at: {log_file}[/dim]")
        raise typer.Exit(1)
