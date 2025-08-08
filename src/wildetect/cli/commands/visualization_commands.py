"""
Visualization commands.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from ...core.config import FlightSpecs
from ...core.config_loader import ROOT
from ...core.data.drone_image import DroneImage
from ...core.data.utils import get_images_paths
from ...core.visualization.geographic import GeographicVisualizer
from ..utils import create_geographic_visualization, setup_logging

app = typer.Typer(name="visualization", help="Visualization commands")
console = Console()


@app.command()
def visualize(
    results_path: str = typer.Argument(..., help="Path to detection results"),
    output_dir: str = typer.Option(
        "results", "--output", "-o", help="Output directory for visualizations"
    ),
    create_map: bool = typer.Option(
        True, "--map", help="Create geographic visualization map"
    ),
):
    """Visualize detection results with geographic maps and statistics."""
    setup_logging(
        log_file=str(
            ROOT
            / "logs"
            / "visualize"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    )
    logger = logging.getLogger(__name__)

    try:
        results_path_obj = Path(results_path)
        if not results_path_obj.exists():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(1)

        # Load results
        with open(results_path_obj, "r") as f:
            results = json.load(f)

        # Create output directory
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        console.print(f"[green]Visualizing results from: {results_path}[/green]")

        # Display basic statistics
        if isinstance(results, list):
            total_images = len(results)
            total_detections = sum(
                result.get("total_detections", 0) for result in results
            )
            console.print(
                f"[green]Found {total_images} images with {total_detections} total detections[/green]"
            )

            # Species breakdown
            species_counts = {}
            for result in results:
                for species, count in result.get("class_counts", {}).items():
                    species_counts[species] = species_counts.get(species, 0) + count

            if species_counts:
                console.print(f"\n[bold green]Species Detected:[/bold green]")
                for species, count in sorted(
                    species_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    console.print(f"  {species}: {count}")

        # Create geographic visualization if requested and data available
        if create_map:
            console.print("[green]Creating geographic visualization...[/green]")

            # Try to create visualization from results
            try:
                # If results contain drone_images, use them directly
                if isinstance(results, dict) and "drone_images" in results:
                    drone_images = results["drone_images"]
                    create_geographic_visualization(drone_images, output_dir)
                else:
                    console.print(
                        "[yellow]No geographic data found in results for map creation[/yellow]"
                    )
            except Exception as e:
                console.print(
                    f"[red]Failed to create geographic visualization: {e}[/red]"
                )

        # Export visualization report
        visualization_report = {
            "source_file": str(results_path),
            "total_images": len(results) if isinstance(results, list) else 0,
            "total_detections": sum(
                result.get("total_detections", 0) for result in results
            )
            if isinstance(results, list)
            else 0,
            "species_breakdown": species_counts if "species_counts" in locals() else {},
            "visualization_created": create_map,
            "timestamp": datetime.now().isoformat(),
        }

        report_file = output_dir_obj / "visualization_report.json"
        with open(report_file, "w") as f:
            json.dump(visualization_report, f, indent=2)

        console.print(f"[green]Visualization report saved to: {report_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Visualization failed: {e}")
        raise typer.Exit(1)


@app.command()
def visualize_geographic_bounds(
    image_dir: str = typer.Argument(..., help="Image directory"),
    output_dir: str = typer.Option(
        "visualizations", "--output", "-o", help="Output directory for visualizations"
    ),
    sensor_height: float = typer.Option(
        24.0, "--sensor-height", help="Sensor height in mm"
    ),
    focal_length: float = typer.Option(
        35.0, "--focal-length", help="Focal length in mm"
    ),
    flight_height: float = typer.Option(
        180.0, "--flight-height", help="Flight height in meters"
    ),
) -> None:
    """Convenience function to visualize geographic bounds."""
    setup_logging(
        log_file=str(
            ROOT
            / "logs"
            / "visualize_geographic_bounds"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    )
    logger = logging.getLogger(__name__)
    try:
        assert Path(image_dir).is_dir(), f"Image directory not found: {image_dir}"

        flight_specs = FlightSpecs(
            sensor_height=sensor_height,
            focal_length=focal_length,
            flight_height=flight_height,
        )
        drone_images = [
            DroneImage.from_image_path(image, flight_specs=flight_specs)
            for image in get_images_paths(image_dir)
        ]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / "geographic_visualization.html")
        GeographicVisualizer().create_map(drone_images, output_path)
        console.print(
            f"[green]Geographic visualization saved to: {output_path}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Visualization failed: {e}")
        raise typer.Exit(1)
