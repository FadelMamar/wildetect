"""
CLI utilities and helper functions.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rich.console import Console
from rich.table import Table

from ..core.data.census import CensusDataManager
from ..core.data.drone_image import DroneImage
from ..core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)

console = Console()


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    if isinstance(log_file, str):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_cls_label_map(cls_label_map: List[str]) -> Dict[int, str]:
    """Parse a list of key:value strings into a dictionary with int keys and str values."""
    result = {}
    for item in cls_label_map:
        key, value = item.split(":", 1)
        result[int(key)] = value
    return result


def display_results(drone_images: List[DroneImage], output_dir: Optional[str] = None):
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
        # Get all predictions and filter out empty ones
        non_empty_predictions = drone_image.get_non_empty_predictions()

        # Count detections by class (excluding empty ones)
        for detection in non_empty_predictions:
            class_name = detection.class_name or "unknown"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        total_detections += len(non_empty_predictions)

        # Add row to table
        table.add_row(
            Path(drone_image.image_path).name,
            str(len(non_empty_predictions)),
            ", ".join(set(det.class_name for det in non_empty_predictions)) or "None",
            "✓" if len(non_empty_predictions) > 0 else "✗",
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

        # Filter out drone images with empty detections
        images_with_detections = [
            drone_image
            for drone_image in drone_images
            if len(drone_image.get_non_empty_predictions()) > 0
        ]

        console.print(f"Images with detections: {len(images_with_detections)}")
        console.print(
            f"Images with GPS: {len([img for img in drone_images if img.latitude and img.longitude])}"
        )

        if images_with_detections:
            try:
                # Create geographic visualization
                config = VisualizationConfig()
                map_file = str(output_path / "geographic_visualization.html")

                visualizer = GeographicVisualizer(config)
                visualizer.create_map(images_with_detections, map_file)

                console.print(
                    f"[green]✓ Geographic visualization saved to: {map_file}[/green]"
                )

                # Open the map in the default browser
                subprocess.Popen(f"start {map_file}", shell=True)

                # Get coverage statistics
                coverage_stats = visualizer.get_coverage_statistics(drone_images)
                console.print(f"[green]✓ Coverage statistics calculated[/green]")
                console.print(f"  Images with GPS: {coverage_stats['images_with_gps']}")
                console.print(
                    f"  Images with footprints: {coverage_stats['images_with_footprints']}"
                )
                # if coverage_stats["total_overlap_area"] > 0:
                console.print(
                    f"  Total overlap area: {coverage_stats['total_overlap_area']:.2f} m²"
                )

            except Exception as e:
                console.print(
                    f"[red]✗ Failed to create geographic visualization: {e}[/red]"
                )
                console.print(f"[yellow]Continuing without visualization...[/yellow]")
        else:
            console.print(
                "[yellow]No images with detections found for visualization[/yellow]"
            )


def display_census_results(
    census_manager: CensusDataManager,
    stats: dict,
):
    """Display comprehensive census campaign results."""
    console.print(
        f"\n[bold blue]Census Campaign Results: {census_manager.campaign_id}[/bold blue]"
    )

    # Campaign overview
    table = Table(title="Campaign Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Campaign ID", census_manager.campaign_id)
    table.add_row("Total Images", str(len(census_manager.image_paths)))
    table.add_row("Drone Images Created", str(len(census_manager.drone_images)))

    if "flight_analysis" in stats:
        flight_stats = stats["flight_analysis"]
        table.add_row(
            "Images with GPS", str(flight_stats.get("num_images_with_gps", 0))
        )
        table.add_row("Total Waypoints", str(flight_stats.get("total_waypoints", 0)))
        table.add_row(
            "Total Distance (km)", f"{flight_stats.get('total_distance_km', 0):.2f}"
        )

    console.print(table)

    display_results(census_manager.drone_images, output_dir=None)

    # Geographic coverage
    if census_manager.drone_images:
        geo_stats = get_geographic_coverage(census_manager.drone_images)
        console.print(f"\n[bold green]Geographic Coverage:[/bold green]")
        console.print(f"  Images with GPS: {geo_stats['images_with_gps']}")
        console.print(
            f"  Images with footprints: {geo_stats['images_with_footprints']}"
        )

        if geo_stats["geographic_bounds"] and isinstance(
            geo_stats["geographic_bounds"], dict
        ):
            bounds = geo_stats["geographic_bounds"]
            console.print(f"  Coverage bounds:")
            console.print(
                f"    Latitude: {bounds.get('min_lat', 0):.6f} to {bounds.get('max_lat', 0):.6f}"
            )
            console.print(
                f"    Longitude: {bounds.get('min_lon', 0):.6f} to {bounds.get('max_lon', 0):.6f}"
            )


def get_detection_statistics(drone_images: List[DroneImage]) -> dict:
    """Calculate detection statistics from drone images."""
    total_detections = 0
    species_counts = {}

    for drone_image in drone_images:
        detections = drone_image.get_non_empty_predictions()
        total_detections += len(detections)

        for detection in detections:
            species = detection.class_name
            species_counts[species] = species_counts.get(species, 0) + 1

    return {
        "total_detections": total_detections,
        "species_counts": species_counts,
    }


def get_geographic_coverage(drone_images: List) -> dict:
    """Calculate geographic coverage statistics."""
    coverage = {
        "images_with_gps": 0,
        "images_with_footprints": 0,
        "geographic_bounds": None,
    }

    gps_images = [img for img in drone_images if img.latitude and img.longitude]
    coverage["images_with_gps"] = len(gps_images)

    footprint_images = [img for img in drone_images if img.geographic_footprint]
    coverage["images_with_footprints"] = len(footprint_images)

    if gps_images:
        lats = [img.latitude for img in gps_images if img.latitude is not None]
        lons = [img.longitude for img in gps_images if img.longitude is not None]

        if lats and lons:
            coverage["geographic_bounds"] = {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons),
            }

    return coverage


def analyze_detection_results(results: Union[dict, list]) -> dict:
    """Analyze detection results and generate insights."""
    analysis = {
        "total_images": 0,
        "total_detections": 0,
        "species_breakdown": {},
        "confidence_distribution": {},
        "geographic_coverage": {},
        "processing_efficiency": {},
    }

    # Extract statistics from results
    if isinstance(results, list):
        # Handle list of image results
        analysis["total_images"] = len(results)
        for result in results:
            if "total_detections" in result:
                analysis["total_detections"] += result["total_detections"]
            if "class_counts" in result:
                for species, count in result["class_counts"].items():
                    analysis["species_breakdown"][species] = (
                        analysis["species_breakdown"].get(species, 0) + count
                    )

    return analysis


def display_analysis_results(analysis_results: dict):
    """Display analysis results."""
    console.print(f"\n[bold blue]Analysis Results[/bold blue]")

    table = Table(title="Detection Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Images", str(analysis_results.get("total_images", 0)))
    table.add_row("Total Detections", str(analysis_results.get("total_detections", 0)))
    table.add_row(
        "Species Detected", str(len(analysis_results.get("species_breakdown", {})))
    )

    console.print(table)

    # Species breakdown
    if analysis_results.get("species_breakdown"):
        console.print(f"\n[bold green]Species Breakdown:[/bold green]")
        for species, count in sorted(
            analysis_results["species_breakdown"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            console.print(f"  {species}: {count}")


def export_analysis_report(analysis_results: dict, output_dir: str):
    """Export analysis report."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)

        console.print(f"[green]Analysis report exported to: {report_file}[/green]")

    except Exception as e:
        console.print(f"[red]Failed to export analysis report: {e}[/red]")


def export_detection_report(
    drone_images: List[DroneImage],
    output_path: str,
    detection_type: str = "annotations",
):
    all_detections = []
    for drone_image in drone_images:
        assert isinstance(
            drone_image, DroneImage
        ), f"Drone image must be a DroneImage object, got {type(drone_image)}"
        if detection_type == "annotations":
            all_detections.extend(drone_image.get_non_empty_annotations())
        elif detection_type == "predictions":
            all_detections.extend(drone_image.get_non_empty_predictions())
        else:
            raise ValueError(f"Invalid detection type: {detection_type}")
    gps_coords = [detection.gps_as_decimals for detection in all_detections]
    df = pd.DataFrame(gps_coords, columns=["latitude", "longitude", "altitude"])
    df.to_csv(Path(output_path), index=False)
