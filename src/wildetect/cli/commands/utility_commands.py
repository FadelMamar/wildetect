"""
Utility commands.
"""

import logging
import os
import sys
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from ...core.config_loader import create_pydantic_default_config
from ..utils import setup_logging

app = typer.Typer(name="utils", help="Utility commands")
console = Console()


@app.command()
def info():
    """Display system information and dependencies."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        console.print("[bold green]WildDetect System Information[/bold green]")

        # System info table
        table = Table(title="System Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Python version
        table.add_row("Python Version", sys.version.split()[0])

        # PyTorch version
        table.add_row("PyTorch Version", torch.__version__)

        # CUDA availability
        cuda_available = torch.cuda.is_available()
        table.add_row("CUDA Available", str(cuda_available))

        if cuda_available:
            table.add_row("CUDA Version", torch.version.cuda)
            table.add_row("GPU Count", str(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                table.add_row(f"GPU {i}", gpu_name)

        console.print(table)

        # Dependencies info
        console.print("\n[bold green]Dependencies[/bold green]")
        deps_table = Table()
        deps_table.add_column("Package", style="cyan")
        deps_table.add_column("Status", style="green")

        # Check key dependencies
        dependencies = [
            "torch",
            "torchvision",
            "ultralytics",
            "fiftyone",
            "streamlit",
            "fastapi",
            "uvicorn",
            "rich",
            "typer",
        ]

        for dep in dependencies:
            try:
                __import__(dep)
                deps_table.add_row(dep, "✓ Installed")
            except ImportError:
                deps_table.add_row(dep, "✗ Missing")

        console.print(deps_table)

        # Project info
        console.print("\n[bold green]Project Information[/bold green]")
        project_table = Table()
        project_table.add_column("Property", style="cyan")
        project_table.add_column("Value", style="green")

        # Get project root
        from ...core.config import ROOT

        project_table.add_row("Project Root", str(ROOT))

        # Check if models directory exists
        models_dir = ROOT / "models"
        project_table.add_row(
            "Models Directory", "✓ Exists" if models_dir.exists() else "✗ Missing"
        )

        # Check if weights directory exists
        weights_dir = ROOT / "weights"
        project_table.add_row(
            "Weights Directory", "✓ Exists" if weights_dir.exists() else "✗ Missing"
        )

        console.print(project_table)

        # Environment info
        console.print("\n[bold green]Environment Variables[/bold green]")
        env_table = Table()
        env_table.add_column("Variable", style="cyan")
        env_table.add_column("Value", style="green")

        env_vars = ["CUDA_VISIBLE_DEVICES", "ROI_MODEL_PATH", "MLFLOW_TRACKING_URI"]
        for var in env_vars:
            value = os.getenv(var, "Not set")
            env_table.add_row(var, value)

        console.print(env_table)

    except Exception as e:
        console.print(f"[red]Error getting system info: {e}[/red]")
        logger.error(f"System info failed: {e}")
        raise typer.Exit(1)


@app.command()
def clear_results(
    results_dir: str = typer.Option(
        "results", help="Directory containing results to clear."
    ),
):
    """Delete all detection results in the specified directory."""
    if not typer.confirm(
        f"Are you sure you want to delete all results in '{results_dir}'?"
    ):
        typer.echo("Operation cancelled.")
        raise typer.Exit(0)
    import shutil
    from pathlib import Path

    results_path = Path(results_dir)
    if results_path.exists() and results_path.is_dir():
        shutil.rmtree(results_path)
        typer.secho(
            f"All results in '{results_dir}' have been deleted.", fg=typer.colors.GREEN
        )
        raise typer.Exit(0)
    else:
        typer.secho(
            f"Directory '{results_dir}' does not exist.", fg=typer.colors.YELLOW
        )
        raise typer.Exit(0)


@app.command()
def install_cuda(
    cuda_version: Optional[str] = typer.Option(
        None, "--cuda-version", "-c", help="Specific CUDA version (118, 121)"
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Force CPU-only installation"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Install PyTorch with CUDA support for optimal performance."""
    from ...utils.cuda_installer import install_cpu_torch, install_cuda_torch

    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        if cpu_only:
            console.print("[yellow]Installing CPU-only PyTorch...[/yellow]")
            install_cpu_torch()
        else:
            console.print("[yellow]Installing PyTorch with CUDA support...[/yellow]")
            install_cuda_torch(cuda_version)

        console.print("[green]✅ Installation completed![/green]")

    except Exception as e:
        logger.error(f"Installation failed: {e}")
        console.print(f"[red]❌ Installation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_config(
    command_type: str = typer.Argument(
        ..., help="Command type (detect, census, visualize)"
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for configuration file"
    ),
    use_pydantic: bool = typer.Option(
        True, "--pydantic", help="Use Pydantic models for validation"
    ),
):
    """Create a default configuration file for a command type."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        if use_pydantic:
            output_file = create_pydantic_default_config(command_type, output_path)
            console.print(
                f"[green]Created Pydantic-validated config: {output_file}[/green]"
            )
        else:
            # Use legacy config creation
            from ...core.config_loader import config_loader

            output_file = config_loader.create_default_config(command_type, output_path)
            console.print(f"[green]Created legacy config: {output_file}[/green]")

        console.print(f"[green]Configuration file created successfully![/green]")
        console.print(
            f"[green]You can now use this config with: wildetect {command_type} --config {output_file}[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error creating config: {e}[/red]")
        logger.error(f"Config creation failed: {e}")
        raise typer.Exit(1)
