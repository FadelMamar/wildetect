"""Configuration-related CLI commands."""

import typer
from pathlib import Path
from typing import Optional

from ..config_loader import ConfigLoader
from ...shared.config_types import ConfigType
from ...shared.validation import ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
from .utils import console

config_app = typer.Typer(name="config", help="Configuration management commands")


@config_app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to configuration file"),
    config_type: str = typer.Option(
        "classification", 
        "--type", 
        "-t", 
        help="Configuration type (classification, detection, classification_eval, detection_eval, classification_visualization, detection_visualization, pipeline, detector_registration, classifier_registration, model_registration)"
    ),
) -> None:
    """Validate a configuration file using Pydantic models."""
    console.print(f"[bold green]Validating {config_type} configuration:[/bold green] {config}")
    
    try:
        is_valid = ConfigLoader.validate_config_file(config, ConfigType(config_type))
        if is_valid:
            console.print(f"[bold green]✓[/bold green] Configuration is valid!")
        else:
            console.print(f"[bold red]✗[/bold red] Configuration validation failed!")
            raise typer.Exit(1)
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as e:
        console.print(f"[bold red]✗[/bold red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Unexpected error: {str(e)}")
        raise typer.Exit(1)


@config_app.command()
def template(
    config_type: str = typer.Argument(..., help="Configuration type to show template for"),
    save_to: Optional[Path] = typer.Option(None, "--save", "-s", help="Save template to file")
) -> None:
    """Show a default YAML configuration template for the specified config type."""
    console.print(f"[bold green]Generating {config_type} configuration template...[/bold green]")
    
    try:
        template = ConfigLoader.generate_default_config(ConfigType(config_type))
        
        if save_to:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, 'w', encoding='utf-8') as f:
                f.write(template)
            console.print(f"[bold green]✓[/bold green] Template saved to: {save_to}")
        else:
            console.print(f"\n[bold blue]Default {config_type} configuration template:[/bold blue]")
            console.print(f"\n```yaml\n{template}```")
            
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
        raise typer.Exit(1)
