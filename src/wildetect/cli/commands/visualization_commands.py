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
from ...core.config_loader import ROOT, load_config_with_pydantic
from ...core.config_models import VisualizeConfigModel
from ...core.data.drone_image import DroneImage
from ...core.data.utils import get_images_paths
from ...core.visualization.geographic import GeographicVisualizer, VisualizationConfig
from ..utils import create_geographic_visualization, setup_logging

app = typer.Typer(name="visualization", help="Visualization commands")
console = Console()


@app.command()
def visualize(
    results_path: str = typer.Argument(..., help="Path to detection results"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    output_dir: str = typer.Option(
        "results", "--output", "-o", help="Override output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    # Essential overrides
    create_map: Optional[bool] = typer.Option(
        None, "--map", help="Override create map setting"
    ),
    show_confidence: Optional[bool] = typer.Option(
        None, "--show-confidence", help="Override show confidence setting"
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None, "--confidence", help="Override confidence threshold"
    ),
    # Output format overrides
    format: Optional[str] = typer.Option(
        None, "--format", help="Override output format"
    ),
    auto_open: Optional[bool] = typer.Option(
        None, "--auto-open", help="Override auto-open setting"
    ),
):
    """Visualize detection results with geographic maps and statistics."""
    setup_logging(
        verbose,
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

        # Load configuration from YAML
        if config:
            loaded_config = load_config_with_pydantic("visualize", config)

            # Ensure we have the correct config type for visualization
            if not isinstance(loaded_config, VisualizeConfigModel):
                console.print(
                    f"[red]Error: Configuration file is not a valid visualization config[/red]"
                )
                raise typer.Exit(1)

            # Apply command-line overrides
            if create_map is not None:
                loaded_config.geographic.create_map = create_map
            if show_confidence is not None:
                loaded_config.geographic.show_confidence = show_confidence
            if confidence_threshold is not None:
                loaded_config.visualization.confidence_threshold = confidence_threshold
            if output_dir:
                loaded_config.geographic.output_directory = output_dir
            if format:
                loaded_config.output.format = format
            if auto_open is not None:
                loaded_config.output.auto_open = auto_open

            # Set output directory
            output_dir_obj = Path(loaded_config.geographic.output_directory)
            create_map = loaded_config.geographic.create_map
            show_confidence = loaded_config.geographic.show_confidence
            confidence_threshold = loaded_config.visualization.confidence_threshold
            output_format = loaded_config.output.format
            auto_open_browser = loaded_config.output.auto_open
        else:
            # Use command-line parameters directly (legacy mode)
            output_dir_obj = Path(output_dir)
            create_map = create_map if create_map is not None else True
            show_confidence = show_confidence if show_confidence is not None else False
            confidence_threshold = (
                confidence_threshold if confidence_threshold is not None else 0.2
            )
            output_format = format if format else "html"
            auto_open_browser = auto_open if auto_open is not None else False

        # Create output directory
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(results_path_obj, "r") as f:
            results = json.load(f)

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
                    create_geographic_visualization(drone_images, str(output_dir_obj))
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
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Override output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    # Essential overrides
    sensor_height: Optional[float] = typer.Option(
        None, "--sensor-height", help="Override sensor height in mm"
    ),
    focal_length: Optional[float] = typer.Option(
        None, "--focal-length", help="Override focal length in mm"
    ),
    flight_height: Optional[float] = typer.Option(
        None, "--flight-height", help="Override flight height in meters"
    ),
    # Visualization overrides
    show_confidence: Optional[bool] = typer.Option(
        None, "--show-confidence", help="Override show confidence setting"
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None, "--confidence", help="Override confidence threshold"
    ),
    # Output format overrides
    format: Optional[str] = typer.Option(
        None, "--format", help="Override output format"
    ),
    auto_open: Optional[bool] = typer.Option(
        None, "--auto-open", help="Override auto-open setting"
    ),
) -> None:
    """Convenience function to visualize geographic bounds."""
    setup_logging(
        verbose,
        log_file=str(
            ROOT
            / "logs"
            / "visualize_geographic_bounds"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    )
    logger = logging.getLogger(__name__)

    try:
        assert Path(image_dir).is_dir(), f"Image directory not found: {image_dir}"

        # Load configuration from YAML
        if config:
            loaded_config = load_config_with_pydantic("visualize", config)

            # Ensure we have the correct config type for visualization
            if not isinstance(loaded_config, VisualizeConfigModel):
                console.print(
                    f"[red]Error: Configuration file is not a valid visualization config[/red]"
                )
                raise typer.Exit(1)

            # Apply command-line overrides
            if sensor_height is not None:
                loaded_config.flight_specs.sensor_height = sensor_height
            if focal_length is not None:
                loaded_config.flight_specs.focal_length = focal_length
            if flight_height is not None:
                loaded_config.flight_specs.flight_height = flight_height
            if show_confidence is not None:
                loaded_config.geographic.show_confidence = show_confidence
            if confidence_threshold is not None:
                loaded_config.visualization.confidence_threshold = confidence_threshold
            if output_dir:
                loaded_config.geographic.output_directory = output_dir
            if format:
                loaded_config.output.format = format
            if auto_open is not None:
                loaded_config.output.auto_open = auto_open

            # Convert to existing dataclasses
            flight_specs = loaded_config.flight_specs.to_flight_specs()
            visualization_config = VisualizationConfig(
                show_detections=loaded_config.visualization.show_detections,
                show_statistics=loaded_config.visualization.show_statistics,
                show_image_bounds=loaded_config.visualization.show_footprints,
                map_center=None,  # Will be auto-calculated
                zoom_start=loaded_config.geographic.zoom_level,
                tiles=loaded_config.geographic.map_type,
            )

            # Set output directory
            output_dir_obj = Path(loaded_config.geographic.output_directory)
            output_format = loaded_config.output.format
            auto_open_browser = loaded_config.output.auto_open
        else:
            # Use command-line parameters directly (legacy mode)
            flight_specs = FlightSpecs(
                sensor_height=sensor_height or 24.0,
                focal_length=focal_length or 35.0,
                flight_height=flight_height or 180.0,
            )
            visualization_config = VisualizationConfig(
                show_detections=True,
                show_statistics=True,
                show_image_bounds=True,
                map_center=None,  # Will be auto-calculated
                zoom_start=12,
                tiles="OpenStreetMap",
            )

            output_dir_obj = Path(output_dir or "visualizations")
            output_format = format if format else "html"
            auto_open_browser = auto_open if auto_open is not None else False

        drone_images = [
            DroneImage.from_image_path(image, flight_specs=flight_specs)
            for image in get_images_paths(image_dir)
        ]

        output_dir_obj.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir_obj / f"geographic_visualization.{output_format}")

        GeographicVisualizer(config=visualization_config).create_map(
            drone_images, output_path
        )

        console.print(
            f"[green]Geographic visualization saved to: {output_path}[/green]"
        )

        if auto_open_browser:
            import webbrowser

            webbrowser.open(output_path)
            console.print(f"[green]Opened visualization in browser[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Visualization failed: {e}")
        raise typer.Exit(1)
