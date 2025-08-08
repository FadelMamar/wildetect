"""
Service management commands.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from ...core.config import ROOT
from ...core.detectors.detection_server import run_inference_server
from ...core.visualization.fiftyone_manager import FiftyOneManager
from ..utils import setup_logging

app = typer.Typer(name="services", help="Service management commands")
console = Console()


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run the UI on"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to run the UI on"),
    open_browser: bool = typer.Option(
        True, "--no-browser", help="Don't open browser automatically"
    ),
):
    """Launch the WildDetect web interface with CLI integration."""
    try:
        # Get the path to the UI module
        ui_path = ROOT / "src" / "wildetect" / "ui" / "main.py"

        if not ui_path.exists():
            console.print(f"[red]UI module not found at: {ui_path}[/red]")
            raise typer.Exit(1)

        # Launch Streamlit
        cmd = [
            "uv",
            "run",
            "streamlit",
            "run",
            str(ui_path),
            "--server.port",
            str(port),
            "--server.address",
            host,
            "--server.headless",
            "true" if not open_browser else "false",
        ]

        subprocess.Popen(
            cmd,
            env=os.environ.copy(),  # creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        console.print(f"[green]Launching WildDetect UI on http://{host}:{port}[/green]")
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

    except ImportError:
        console.print("[red]Streamlit not installed. Please install it with:[/red]")
        console.print("[yellow]pip install streamlit[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error launching UI: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def api(
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h", help="Host to run the API server on"
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port to run the API server on"
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
):
    """Launch the WildDetect FastAPI server."""
    try:
        import uvicorn

        from ...api.main import app

        console.print(
            f"[green]Starting WildDetect API server on http://{host}:{port}[/green]"
        )
        console.print(
            f"[yellow]API documentation available at: http://{host}:{port}/docs[/yellow]"
        )
        console.print(
            f"[yellow]ReDoc documentation available at: http://{host}:{port}/redoc[/yellow]"
        )
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

        uvicorn.run(
            "wildetect.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )

    except ImportError:
        console.print(
            "[red]FastAPI dependencies not installed. Please install them with:[/red]"
        )
        console.print("[yellow]pip install fastapi uvicorn python-multipart[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error launching API server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def inference_server(
    port: int = typer.Option(
        4141, "--port", "-p", help="Port to run the inference server on"
    ),
    workers_per_device: int = typer.Option(
        1, "--workers", "-w", help="Number of workers per device"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Launch the inference server for wildlife detection."""
    log_file = (
        ROOT
        / "logs"
        / "inference_service"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(verbose=verbose, log_file=log_file)
    logger = logging.getLogger(__name__)

    load_dotenv(ROOT / ".env", override=True)

    try:
        console.print(f"[green]Starting inference server on port {port}...[/green]")
        console.print(f"[green]Workers per device: {workers_per_device}[/green]")

        # Launch the inference server
        run_inference_server(port=port, workers_per_device=workers_per_device)

    except Exception as e:
        console.print(f"[red]Error starting inference server: {e}[/red]")
        logger.error(f"Inference server failed: {e}")
        raise typer.Exit(1)


@app.command()
def fiftyone(
    dataset_name: str = typer.Option(
        "wildlife_detection", "--dataset", "-d", help="Dataset name"
    ),
    action: str = typer.Option(
        "launch", "--action", "-a", help="Action to perform: launch, info, export"
    ),
    export_format: str = typer.Option(
        "coco", "--format", "-f", help="Export format (coco, yolo, pascal)"
    ),
    export_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Export output path"
    ),
):
    """Manage FiftyOne datasets for wildlife detection."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        if action == "launch":
            console.print("[green]Launching FiftyOne app...[/green]")
            FiftyOneManager.launch_app()
            console.print("[green]FiftyOne app launched successfully![/green]")

        elif action == "info":
            console.print(f"[green]Getting dataset info for: {dataset_name}[/green]")
            fo_manager = FiftyOneManager(dataset_name)
            dataset_info = fo_manager.get_dataset_info()

            table = Table(title="Dataset Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Dataset Name", dataset_info["name"])
            table.add_row("Total Samples", str(dataset_info["num_samples"]))
            table.add_row("Fields", str(len(dataset_info["fields"])))

            console.print(table)

            # Get annotation statistics
            annotation_stats = fo_manager.get_annotation_stats()
            console.print(f"\n[bold green]Annotation Statistics:[/bold green]")
            console.print(
                f"  Annotated Samples: {annotation_stats['annotated_samples']}"
            )
            console.print(f"  Total Detections: {annotation_stats['total_detections']}")
            console.print(
                f"  Annotation Rate: {annotation_stats['annotation_rate']:.1f}%"
            )

        elif action == "export":
            if not export_path:
                export_path = f"exports/{dataset_name}_{export_format}"

            console.print(f"[green]Exporting dataset to: {export_path}[/green]")
            fo_manager = FiftyOneManager(dataset_name)

            # Create export directory
            Path(export_path).mkdir(parents=True, exist_ok=True)

            # Export dataset using FiftyOne's export method
            if fo_manager.dataset:
                fo_manager.dataset.export(
                    export_dir=export_path, dataset_type=export_format, overwrite=True
                )
                console.print(
                    f"[green]Dataset exported successfully to: {export_path}[/green]"
                )
            else:
                console.print("[red]Error: Dataset not initialized[/red]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: launch, info, export")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"FiftyOne operation failed: {e}")
        raise typer.Exit(1)
