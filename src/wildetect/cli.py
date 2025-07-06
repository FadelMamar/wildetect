"""
Command Line Interface for WildDetect using Typer.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core.config import LoaderConfig, PredictionConfig
from .core.detection_pipeline import DetectionPipeline

# Create Typer app
app = typer.Typer(
    name="wildetect",
    help="WildDetect - Wildlife Detection System",
    add_completion=False,
)

# Create console for rich output
console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@app.command()
def detect(
    images: List[str] = typer.Argument(..., help="Image paths or directory"),
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Path to model weights"
    ),
    model_type: str = typer.Option("yolo", "--type", "-t", help="Model type"),
    confidence: float = typer.Option(
        0.25, "--confidence", "-c", help="Confidence threshold"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to run inference on"
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    tile_size: int = typer.Option(640, "--tile-size", help="Tile size for processing"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    max_images: Optional[int] = typer.Option(
        None, "--max-images", help="Maximum number of images to process"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run wildlife detection on images."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # Determine if input is directory or file paths
        if len(images) == 1 and Path(images[0]).is_dir():
            image_dir = images[0]
            image_paths = None
            console.print(f"[green]Processing directory: {image_dir}[/green]")
        else:
            image_dir = None
            image_paths = images
            console.print(f"[green]Processing {len(images)} images[/green]")

        # Create configurations
        pred_config = PredictionConfig(
            model_path=model_path,
            model_type=model_type,
            confidence_threshold=confidence,
            device=device,
            batch_size=batch_size,
            tilesize=tile_size,
        )

        loader_config = LoaderConfig(
            tile_size=tile_size,
            batch_size=batch_size,
        )

        # Create detection pipeline
        pipeline = DetectionPipeline(
            config=pred_config,
            loader_config=loader_config,
            model_path=model_path,
            model_type=model_type,
            device=device,
        )

        # Run detection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running detection...", total=None)

            drone_images = pipeline.run_detection(
                image_paths=image_paths,
                image_dir=image_dir,
                max_images=max_images,
                save_results=output + "/results.json" if output else None,
            )

            progress.update(task, completed=True)

        # Display results
        display_results(drone_images, output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Detection failed: {e}")
        raise typer.Exit(1)


@app.command()
def visualize(
    results_path: str = typer.Argument(..., help="Path to detection results"),
    output_dir: str = typer.Option(
        "visualizations", "--output", "-o", help="Output directory for visualizations"
    ),
    show_confidence: bool = typer.Option(
        True, "--show-confidence", help="Show confidence scores"
    ),
):
    """Visualize detection results."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        results_path = Path(results_path)
        if not results_path.exists():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(1)

        # Load results
        import json

        with open(results_path, "r") as f:
            results = json.load(f)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[green]Visualizing {len(results)} results[/green]")

        # TODO: Implement visualization logic
        console.print("[yellow]Visualization not yet implemented[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Visualization failed: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system information."""
    console.print("[bold blue]WildDetect System Information[/bold blue]")

    # Create a table with system info
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Check PyTorch
    try:
        import torch

        table.add_row("PyTorch", "✓", f"Version {torch.__version__}")
    except ImportError:
        table.add_row("PyTorch", "✗", "Not installed")

    # Check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            table.add_row("CUDA", "✓", f"Available - {torch.cuda.get_device_name()}")
        else:
            table.add_row("CUDA", "✗", "Not available")
    except ImportError:
        table.add_row("CUDA", "✗", "PyTorch not installed")

    # Check other dependencies
    dependencies = [
        ("PIL", "PIL"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("ultralytics", "ultralytics"),
    ]

    for name, module in dependencies:
        try:
            __import__(module)
            table.add_row(name, "✓", "Installed")
        except ImportError:
            table.add_row(name, "✗", "Not installed")

    console.print(table)


def display_results(drone_images: List, output_dir: Optional[str] = None):
    """Display detection results."""
    if not drone_images:
        console.print("[yellow]No images processed[/yellow]")
        return

    # Create results table
    table = Table(title="Detection Results")
    table.add_column("Image", style="cyan")
    table.add_column("Detections", style="green")
    table.add_column("Classes", style="yellow")
    table.add_column("Status", style="blue")

    total_detections = 0
    class_counts = {}

    for drone_image in drone_images:
        stats = drone_image.get_statistics()

        # Count detections by class
        for class_name, count in stats.get("class_counts", {}).items():
            class_counts[class_name] = class_counts.get(class_name, 0) + count

        total_detections += stats.get("total_detections", 0)

        # Add row to table
        table.add_row(
            Path(stats["image_path"]).name,
            str(stats.get("total_detections", 0)),
            ", ".join(stats.get("class_counts", {}).keys()) or "None",
            "✓" if stats.get("total_detections", 0) > 0 else "✗",
        )

    console.print(table)

    # Summary
    console.print(f"\n[bold green]Summary:[/bold green]")
    console.print(f"  Images processed: {len(drone_images)}")
    console.print(f"  Total detections: {total_detections}")

    if class_counts:
        console.print(f"  Classes found:")
        for class_name, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            console.print(f"    {class_name}: {count}")

    # Save visualizations if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[green]Saving visualizations to: {output_path}[/green]")

        # TODO: Implement visualization saving
        console.print("[yellow]Visualization saving not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
