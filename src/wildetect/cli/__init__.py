"""
Command Line Interface for WildDetect using Typer.

This module provides a modular CLI structure with commands organized by functionality.
"""

import importlib.metadata
import logging
import sys
from typing import Optional

import typer
from rich.console import Console

# Import command groups
from .commands import (
    core_commands,
    service_commands,
    utility_commands,
    visualization_commands,
)
from .utils import setup_logging

# Create Typer app
app = typer.Typer(
    name="wildetect",
    help="WildDetect - Wildlife Detection System",
    add_completion=True,
)

# Create console for rich output
console = Console()

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("wildetect")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.1"


def raise_exit():
    """Raise typer exit."""
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=lambda value: (
            print(f"wildetect version: {__version__}") or raise_exit()
        )
        if value
        else None,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    """WildDetect - Wildlife Detection System"""
    pass


# Register command groups
app.add_typer(
    core_commands.app, name="detection", help="Detection and analysis commands"
)
app.add_typer(
    visualization_commands.app, name="visualization", help="Visualization commands"
)
app.add_typer(service_commands.app, name="services", help="Service management commands")
app.add_typer(utility_commands.app, name="utils", help="Utility commands")

if __name__ == "__main__":
    app()
