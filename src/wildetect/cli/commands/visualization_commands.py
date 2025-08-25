"""
Visualization commands.
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console

from ...core.config_loader import ROOT, load_config_with_pydantic
from ...core.config_models import VisualizeConfigModel
from ...core.data.drone_image import DroneImage
from ...core.data.utils import get_images_paths
from ...core.visualization.geographic import GeographicVisualizer, VisualizationConfig
from ...core.visualization.labelstudio_manager import LabelStudioManager
from ..utils import export_detection_report, setup_logging

app = typer.Typer(name="visualization", help="Visualization commands")
console = Console()


@app.command()
def visualize(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
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
        # Load configuration from YAML
        loaded_config = load_config_with_pydantic("visualize", config)

        image_dir = loaded_config.image_dir
        flight_specs = loaded_config.flight_specs.to_flight_specs()

        if image_dir is None:
            assert (
                loaded_config.labelstudio.url is not None
            ), "Label Studio URL is required"
            assert (
                loaded_config.labelstudio.api_key is not None
            ), "Label Studio API key is required"
            assert isinstance(
                loaded_config.labelstudio.project_id, int
            ), "Label Studio project ID must be an integer"
            ls_client = LabelStudioManager(
                url=loaded_config.labelstudio.url,
                api_key=loaded_config.labelstudio.api_key,
                download_resources=loaded_config.labelstudio.download_resources,
            )
            drone_images = ls_client.get_all_project_detections(
                loaded_config.labelstudio.project_id,
                load_as_drone_image=True,
                flight_specs=flight_specs,
            )
        else:
            drone_images = [
                DroneImage.from_image_path(image, flight_specs=flight_specs)
                for image in get_images_paths(image_dir)
            ]

        # Save detection gps coordinates to csv
        if loaded_config.csv_output_path is not None:
            export_detection_report(
                drone_images,
                loaded_config.csv_output_path,
                detection_type=loaded_config.detection_type,
            )
            console.print(
                f"[green]{loaded_config.detection_type.capitalize()} report exported to: {loaded_config.csv_output_path}[/green]"
            )

        # Ensure we have the correct config type for visualization
        if not isinstance(loaded_config, VisualizeConfigModel):
            console.print(
                f"[red]Error: Configuration file is not a valid visualization config[/red]"
            )
            raise typer.Exit(1)

        if not loaded_config.geographic.create_map:
            return None

        # Visualize predictions on Folium map
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
        logger.error(f"Visualization failed: {traceback.format_exc()}")
        raise typer.Exit(1)
