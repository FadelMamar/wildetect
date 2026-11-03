"""Configuration-related CLI commands."""

from pathlib import Path
from typing import Optional

import typer

from ...shared.config_loader import ConfigLoader
from ...shared.config_types import ConfigType
from ...shared.validation import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)
from .utils import console

config_app = typer.Typer(name="config", help="Configuration management commands")


@config_app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to configuration file"),
    config_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help=(
            "Optional kind: experiment, eval, visualization, "
            "registration, or pipeline"
        ),
    ),
) -> None:
    """Validate an experiment, eval, visualization, or registration file."""
    from ...harness.ops import validate_harness_file

    label = config_type or "harness document"
    console.print(f"[bold green]Validating {label}:[/bold green] {config}")
    try:
        if config_type is None:
            kind = validate_harness_file(config)
        else:
            kind = _validate_kind(config, config_type)
        console.print(f"[bold green]✓[/bold green] Valid {kind} file")
    except (ConfigFileNotFoundError, ConfigParseError, ConfigValidationError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)


def _validate_kind(config: Path, config_type: str) -> str:
    from ...harness.ops import validate_kind

    return validate_kind(config, config_type)


@config_app.command()
def template(
    config_type: str = typer.Argument(
        ..., help="Configuration type to show template for"
    ),
    save_to: Optional[Path] = typer.Option(
        None, "--save", "-s", help="Save template to file"
    ),
) -> None:
    """Show a default YAML configuration template for the specified config type."""
    console.print(
        f"[bold green]Generating {config_type} configuration template...[/bold green]"
    )

    try:
        template = ConfigLoader.generate_default_config(ConfigType(config_type))

        if save_to:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(template)
            console.print(f"[bold green]✓[/bold green] Template saved to: {save_to}")
        else:
            console.print(
                "\n[bold blue]Default "
                f"{config_type} configuration template:[/bold blue]"
            )
            console.print(f"\n```yaml\n{template}```")

    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to generate template: {str(e)}")
        raise typer.Exit(1)
