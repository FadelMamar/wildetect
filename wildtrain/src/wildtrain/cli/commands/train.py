"""Training commands.

``train classifier`` and ``train detector`` are adapters. They load today's
YAML (no ``task`` key required) into an ``ExperimentConfig`` and call
``run_experiment``, which dispatches through the harness registry.
"""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ...harness.config import load_experiment
from ...harness.runner import run_experiment
from .utils import console, log_file_path, setup_logging

train_app = typer.Typer(name="train", help="Training commands")


def _train(config: Path, task: str, log_name: str) -> None:
    console.print(f"[bold green]Training {task} with config:[/bold green] {config}")
    setup_logging(log_file=log_file_path(log_name))
    try:
        experiment = load_experiment(config, task=task)
        console.print(
            f"[bold green]✓[/bold green] Loaded task [bold]{experiment.task}[/bold]"
        )
        result = run_experiment(experiment)
        console.print(
            "Best checkpoint: "
            f"{result.best_path} ({result.score_name}={result.best_score})"
        )
    except (FileNotFoundError, ValidationError, KeyError, ValueError) as exc:
        console.print(f"[bold red]✗[/bold red] Configuration error: {exc}")
        raise typer.Exit(1)
    except Exception:
        console.print(
            f"[bold red]✗[/bold red] Training failed:\n{traceback.format_exc()}"
        )
        raise typer.Exit(1)


@train_app.command()
def classifier(
    config: Path = typer.Option(
        "", "--config", "-c", help="Path to training configuration file"
    ),
) -> None:
    """Train a classification model from a classification YAML or experiment file."""
    _train(config, "classification", "train_classifier")


@train_app.command()
def detector(
    config: Path = typer.Option(
        "", "--config", "-c", help="Path to training configuration file"
    ),
) -> None:
    """Train a detection model from a detection YAML or experiment file."""
    _train(config, "detection", "train_detector")
