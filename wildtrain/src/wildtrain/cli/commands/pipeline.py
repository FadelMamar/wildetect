"""The old pipeline commands have been removed."""

import typer

from .utils import console

pipeline_app = typer.Typer(
    name="pipeline",
    help="Removed. Set eval.enabled and use wildtrain run -c.",
    invoke_without_command=True,
)


@pipeline_app.callback()
def pipeline_removed() -> None:
    """Train-then-eval is an experiment with eval.enabled."""
    console.print("[bold red]wildtrain pipeline has been removed.[/bold red]")
    console.print(
        "Train then evaluate with one experiment file: set eval.enabled: true "
        "and run [bold]wildtrain run -c experiment.yaml[/bold]."
    )
    raise typer.Exit(1)
