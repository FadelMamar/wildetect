"""Shared utilities for CLI commands."""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from ...utils.logging import ROOT

# Create Rich console
console = Console()


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Setup rich logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [RichHandler(console=console, rich_tracebacks=True)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def display_welcome_message() -> None:
    """Display the welcome message for WildTrain CLI."""
    welcome_text = Text("🚀 WildTrain", style="bold blue")
    subtitle = Text("Modular Computer Vision Framework", style="italic")

    console.print(
        Panel(welcome_text + "\n" + subtitle, border_style="blue", padding=(1, 2))
    )


def log_file_path(command_name: str) -> Path:
    """Generate log file path for a command."""
    return ROOT / "logs" / command_name / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
