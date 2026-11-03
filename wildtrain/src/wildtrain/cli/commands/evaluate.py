"""Evaluate through the harness registry."""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ...harness.config import experiment_from_mapping
from ...harness.ops import load_eval_document, load_yaml
from ...harness.runner import default_registry
from .utils import console, log_file_path, setup_logging

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate an experiment or a standalone eval file",
    invoke_without_command=True,
)


@evaluate_app.callback(invoke_without_command=True)
def evaluate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Experiment YAML or a standalone eval document",
        exists=True,
        dir_okay=False,
    ),
    debug: bool = typer.Option(False, "--debug", help="Score a few batches only"),
) -> None:
    """Evaluate the task named in the file."""
    console.print(f"[bold green]Evaluating:[/bold green] {config}")
    setup_logging(log_file=log_file_path("evaluate"))
    try:
        target = _load_target(config)
        task = default_registry().get(target.task)
        report = task.evaluate(target, debug=debug)
        console.print(report)
    except (FileNotFoundError, ValidationError, KeyError, ValueError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception:
        console.print(
            f"[bold red]✗[/bold red] Evaluation failed:\n{traceback.format_exc()}"
        )
        raise typer.Exit(1)


def _load_target(path: Path):
    payload = load_yaml(path)
    if "train" in payload and "data" in payload:
        return experiment_from_mapping(payload)
    return load_eval_document(path)
