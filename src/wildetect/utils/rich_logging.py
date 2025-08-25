"""
Rich logging utilities for enhanced terminal output and logging.
"""
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich import print as rprint
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.traceback import install

# Install Rich traceback handler for better error display
install(show_locals=True)

# Create Rich console instance
console = Console()

# Configure Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# Create logger instance
logger = logging.getLogger(__name__)


class RichLogger:
    """Enhanced logger with Rich formatting and utilities."""

    def __init__(self, name: str = __name__, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.console = console

        # Ensure RichHandler is added
        if not any(isinstance(h, RichHandler) for h in self.logger.handlers):
            rich_handler = RichHandler(console=console, rich_tracebacks=True)
            self.logger.addHandler(rich_handler)

    def info(self, message: str, **kwargs):
        """Log info message with Rich formatting."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with Rich formatting."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with Rich formatting."""
        self.logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with Rich formatting."""
        self.logger.debug(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with Rich formatting."""
        self.logger.critical(message, **kwargs)

    def success(self, message: str):
        """Log success message with green formatting."""
        self.console.print(f"✅ [green]{message}[/green]")

    def failure(self, message: str):
        """Log failure message with red formatting."""
        self.console.print(f"❌ [red]{message}[/red]")

    def info_blue(self, message: str):
        """Log info message with blue formatting."""
        self.console.print(f"ℹ️ [blue]{message}[/blue]")

    def info_yellow(self, message: str):
        """Log info message with yellow formatting."""
        self.console.print(f"⚠️ [yellow]{message}[/yellow]")


class ProgressTracker:
    """Rich progress tracker for long-running operations."""

    def __init__(self, description: str = "Processing", total: Optional[int] = None):
        self.description = description
        self.total = total
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task_id: Optional[TaskID] = None

    def __enter__(self):
        """Start progress tracking."""
        self.progress.start()
        if self.total:
            self.task_id = self.progress.add_task(self.description, total=self.total)
        else:
            self.task_id = self.progress.add_task(self.description, total=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking."""
        self.progress.stop()

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress."""
        if self.task_id is not None:
            if description:
                self.progress.update(self.task_id, description=description)
            if advance:
                self.progress.advance(self.task_id, advance)

    def set_description(self, description: str):
        """Set task description."""
        if self.task_id is not None:
            self.progress.update(self.task_id, description=description)


class StatusDisplay:
    """Rich status display for showing current operation status."""

    def __init__(self, title: str = "Status"):
        self.title = title
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", size=10),
            Layout(name="footer", size=3),
        )

        self.layout["header"].update(Panel(self.title, style="bold blue"))
        self.layout["body"].update("Ready to start...")
        self.layout["footer"].update("")

        self.live = Live(self.layout, refresh_per_second=4)

    def __enter__(self):
        """Start live display."""
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop live display."""
        self.live.stop()

    def update_status(self, status: str, style: str = "default"):
        """Update status message."""
        self.layout["body"].update(Panel(status, style=style))

    def update_footer(self, footer: str, style: str = "default"):
        """Update footer message."""
        self.layout["footer"].update(Panel(footer, style=style))


def create_table(title: str, columns: list, data: list) -> Table:
    """Create a Rich table with given columns and data."""
    table = Table(title=title)

    # Add columns
    for col in columns:
        if isinstance(col, dict):
            table.add_column(**col)
        else:
            table.add_column(col)

    # Add data rows
    for row in data:
        table.add_row(*row)

    return table


def print_section_header(title: str, style: str = "bold blue"):
    """Print a section header with Rich formatting."""
    console.print(f"\n[bold]{'='*50}[/bold]")
    console.print(f"[{style}]{title}[/{style}]")
    console.print(f"[bold]{'='*50}[/bold]")


def print_success(message: str):
    """Print a success message with Rich formatting."""
    console.print(f"✅ [bold green]{message}[/bold green]")


def print_error(message: str):
    """Print an error message with Rich formatting."""
    console.print(f"❌ [bold red]{message}[/bold red]")


def print_warning(message: str):
    """Print a warning message with Rich formatting."""
    console.print(f"⚠️ [bold yellow]{message}[/bold yellow]")


def print_info(message: str):
    """Print an info message with Rich formatting."""
    console.print(f"ℹ️ [blue]{message}[/blue]")


def print_step(step_number: int, total_steps: int, description: str):
    """Print a step indicator with Rich formatting."""
    console.print(f"🔹 [cyan]Step {step_number}/{total_steps}:[/cyan] {description}")


def print_metric(name: str, value: Any, unit: str = ""):
    """Print a metric with Rich formatting."""
    console.print(f"📊 [cyan]{name}:[/cyan] [bold]{value}[/bold] {unit}")


def create_summary_panel(title: str, content: Dict[str, Any]) -> Panel:
    """Create a summary panel with Rich formatting."""
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in content.items():
        table.add_row(key, str(value))

    return Panel(table, title=title, border_style="blue")


def log_with_context(context: str, message: str, level: str = "info"):
    """Log a message with context information."""
    formatted_message = f"[{context}] {message}"

    if level == "info":
        logger.info(formatted_message)
    elif level == "warning":
        logger.warning(formatted_message)
    elif level == "error":
        logger.error(formatted_message)
    elif level == "debug":
        logger.debug(formatted_message)
    elif level == "critical":
        logger.critical(formatted_message)


@contextmanager
def log_operation(operation_name: str, logger_instance: Optional[RichLogger] = None):
    """Context manager for logging operation start/end."""
    if logger_instance is None:
        logger_instance = RichLogger()

    logger_instance.info(f"🚀 Starting operation: {operation_name}")
    start_time = time.time()

    try:
        yield logger_instance
        execution_time = time.time() - start_time
        logger_instance.success(
            f"✨ Operation '{operation_name}' completed successfully in {execution_time:.2f}s"
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger_instance.failure(
            f"💥 Operation '{operation_name}' failed after {execution_time:.2f}s: {str(e)}"
        )
        raise


# Convenience functions for quick logging
def log_info(message: str):
    """Quick info logging."""
    logger.info(message)


def log_warning(message: str):
    """Quick warning logging."""
    logger.warning(message)


def log_error(message: str):
    """Quick error logging."""
    logger.error(message)


def log_debug(message: str):
    """Quick debug logging."""
    logger.debug(message)


def log_critical(message: str):
    """Quick critical logging."""
    logger.critical(message)


# Export commonly used items
__all__ = [
    "RichLogger",
    "ProgressTracker",
    "StatusDisplay",
    "create_table",
    "print_section_header",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_step",
    "print_metric",
    "create_summary_panel",
    "log_with_context",
    "log_operation",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    "log_critical",
    "console",
    "logger",
]
