"""Run an experiment YAML through the harness registry."""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ...harness.config import load_experiment
from ...harness.runner import run_experiment
from .utils import console, log_file_path, setup_logging

run_app = typer.Typer(name="run", help="Run an experiment file")


@run_app.callback(invoke_without_command=True)
def run_experiment_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help=(
            "Experiment YAML. Canonical form has tracking/run blocks plus "
            "classification or detection. A flat training YAML with task: "
            "also loads."
        ),
        exists=True,
        dir_okay=False,
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Run a short debug training pass"
    ),
) -> None:
    """Train the task named in the experiment file."""
    console.print(f"[bold green]Running experiment:[/bold green] {config}")
    log_file = log_file_path("run_experiment")
    setup_logging(log_file=log_file)

    try:
        experiment = load_experiment(config)
        console.print(
            f"[bold green]✓[/bold green] Loaded task [bold]{experiment.task}[/bold]"
        )
        result = run_experiment(experiment, debug=debug)
        console.print(
            "Best checkpoint: "
            f"{result.best_path} ({result.score_name}={result.best_score})"
        )
    except (FileNotFoundError, ValidationError, KeyError, ValueError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception:
        console.print(f"[bold red]✗[/bold red] Run failed:\n{traceback.format_exc()}")
        raise typer.Exit(1)
