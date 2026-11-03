"""Visualize predictions through the harness registry."""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ...harness.ops import load_visualization_document
from ...harness.runner import default_registry
from .utils import console, log_file_path, setup_logging

visualize_app = typer.Typer(
    name="visualize",
    help="Upload predictions for the task in the file",
    invoke_without_command=True,
)


@visualize_app.callback(invoke_without_command=True)
def visualize(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Visualization YAML with task and dataset_name",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Visualize the task named in the file."""
    console.print(f"[bold green]Visualizing:[/bold green] {config}")
    setup_logging(log_file=log_file_path("visualize"))
    try:
        document = load_visualization_document(config)
        default_registry().get(document.task).visualize(document)
        console.print(
            f"[bold blue]Predictions uploaded to {document.dataset_name}[/bold blue]"
        )
    except (FileNotFoundError, ValidationError, KeyError, ValueError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception:
        console.print(
            f"[bold red]✗[/bold red] Visualization failed:\n{traceback.format_exc()}"
        )
        raise typer.Exit(1)
