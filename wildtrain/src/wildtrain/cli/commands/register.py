"""Register a model through the harness registry."""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ...harness.ops import load_registration_document
from ...harness.runner import default_registry
from .utils import console, log_file_path, setup_logging

register_app = typer.Typer(
    name="register",
    help="Register the task named in the file",
    invoke_without_command=True,
)


@register_app.callback(invoke_without_command=True)
def register(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Registration YAML with task, name, and weights",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Register the model described in the file."""
    console.print(f"[bold green]Registering:[/bold green] {config}")
    setup_logging(log_file=log_file_path("register"))
    try:
        document = load_registration_document(config)
        default_registry().get(document.task).register(document)
        console.print(f"[bold green]✓[/bold green] Registered {document.name}")
    except (FileNotFoundError, ValidationError, KeyError, ValueError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception:
        console.print(
            f"[bold red]✗[/bold red] Registration failed:\n{traceback.format_exc()}"
        )
        raise typer.Exit(1)
