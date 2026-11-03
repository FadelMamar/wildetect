"""The old train commands have been removed."""

import typer

from .utils import console

train_app = typer.Typer(
    name="train",
    help="Removed. Use wildtrain run -c experiment.yaml.",
    invoke_without_command=True,
)


@train_app.callback()
def train_removed() -> None:
    """Tell callers to use wildtrain run."""
    console.print(
        "[bold red]wildtrain train classifier and wildtrain train detector "
        "have been removed.[/bold red]"
    )
    console.print("Train with [bold]wildtrain run -c experiment.yaml[/bold].")
    console.print("Sweep with [bold]wildtrain sweep -c experiment.yaml[/bold].")
    raise typer.Exit(1)
