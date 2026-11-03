"""Sweep command. One runner for either task."""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ...harness.sweep import ExperimentSweeper, load_sweep
from .utils import console, log_file_path, setup_logging

sweep_app = typer.Typer(name="sweep", help="Hyperparameter sweep")


@sweep_app.callback(invoke_without_command=True)
def sweep_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help=(
            "Sweep YAML. Either an experiment file with a sweep block, or a "
            "legacy base_config plus parameters file."
        ),
        exists=True,
        dir_okay=False,
    ),
    debug: bool = typer.Option(False, "--debug", help="Short debug training trials"),
) -> None:
    """Search the categorical grid in the sweep file."""
    console.print(f"[bold green]Running sweep:[/bold green] {config}")
    setup_logging(log_file=log_file_path("sweep"))
    try:
        experiment = load_sweep(config)
        console.print(
            f"[bold green]✓[/bold green] Loaded task [bold]{experiment.task}[/bold]"
        )
        study = ExperimentSweeper(experiment, debug=debug).run()
        best = study.best_trial
        console.print(f"Best value: {best.value}")
        console.print(f"Best params: {best.params}")
    except (FileNotFoundError, ValidationError, KeyError, ValueError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception:
        console.print(f"[bold red]✗[/bold red] Sweep failed:\n{traceback.format_exc()}")
        raise typer.Exit(1)
