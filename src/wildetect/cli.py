"""
Command Line Interface for WildDetect using Typer.
"""

import importlib.metadata
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core.campaign_manager import CampaignConfig, CampaignManager
from .core.config import FlightSpecs, LoaderConfig, PredictionConfig
from .core.data.census import CampaignMetadata, CensusDataManager
from .core.detection_pipeline import DetectionPipeline
from .core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
    visualize_geographic_bounds,
)

ROOT_DIR = Path(__file__).parent.parent.parent

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
    __version__ = "0.0.0"


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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


def raise_exit():
    raise typer.Exit()


@app.command()
def detect(
    images: List[str] = typer.Argument(..., help="Image paths or directory"),
    model_path: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to model weights. Otherwise uses WILDETECT_MODEL_PATH environment variable",
    ),
    model_type: str = typer.Option("yolo", "--type", "-t", help="Model type"),
    confidence: float = typer.Option(
        0.25, "--confidence", "-c", help="Confidence threshold"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to run inference on. auto, cpu or cuda"
    ),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    tile_size: int = typer.Option(800, "--tile-size", help="Tile size for processing"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    roi_weights: Optional[str] = typer.Option(
        None,
        "--roi-weights",
        help="Path to ROI weights file. Otherwise uses ROI_MODEL_PATH environment variable",
    ),
    cls_imgsz: int = typer.Option(96, "--cls-imgsz", help="Image size for classifier"),
    inference_service_url: Optional[str] = typer.Option(
        None, "--inference-service-url", help="URL for inference service"
    ),
    nms_iou: float = typer.Option(0.5, "--nms-iou", help="NMS IoU threshold"),
    overlap_ratio: float = typer.Option(
        0.2, "--overlap-ratio", help="Tile overlap ratio"
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run wildlife detection on images."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # Determine if input is directory or file paths
        logger.info(f"Processing images: {images}")
        if len(images) == 1 and Path(images[0]).is_dir():
            image_dir = images[0]
            image_paths = None
            console.print(f"[green]Processing directory: {image_dir}[/green]")
        else:
            for image in images:
                if not Path(image).exists():
                    raise FileNotFoundError(f"Image not found: {image}")
                if not Path(image).is_file():
                    raise FileNotFoundError(f"Image is not a file: {image}")

            image_dir = None
            image_paths = images
            console.print(f"[green]Processing {len(images)} images[/green]")

        flight_specs = FlightSpecs(
            sensor_height=sensor_height,
            focal_length=focal_length,
            flight_height=flight_height,
        )

        # Create configurations
        pred_config = PredictionConfig(
            model_path=model_path,
            model_type=model_type,
            confidence_threshold=confidence,
            device=device,
            batch_size=batch_size,
            tilesize=tile_size,
            flight_specs=flight_specs,
            roi_weights=roi_weights,
            cls_imgsz=cls_imgsz,
            inference_service_url=inference_service_url,
            verbose=verbose,
            nms_iou=nms_iou,
            overlap_ratio=overlap_ratio,
        )

        loader_config = LoaderConfig(
            tile_size=tile_size,
            batch_size=batch_size,
            num_workers=0,
            flight_specs=flight_specs,
        )

        # Create detection pipeline
        pipeline = DetectionPipeline(
            config=pred_config,
            loader_config=loader_config,
        )
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

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
                save_path=output + "/results.json" if output else None,
            )

            progress.update(task, completed=True)

        # Display results
        display_results(drone_images, output)

    except Exception as e:
        typer.secho(f"[red]Error: {e}[/red]", fg=typer.colors.RED)
        logger.error(f"Detection failed: {e}")
        traceback.print_exc()
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
    create_map: bool = typer.Option(
        True, "--map", help="Create geographic visualization map"
    ),
):
    """Visualize detection results with geographic maps and statistics."""
    setup_logging()
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
        ("folium", "folium"),
        ("shapely", "shapely"),
    ]

    for name, module in dependencies:
        try:
            __import__(module)
            table.add_row(name, "✓", "Installed")
        except ImportError:
            table.add_row(name, "✗", "Not installed")

    console.print(table)


@app.command()
def census(
    campaign_id: str = typer.Argument(..., help="Campaign identifier"),
    images: List[str] = typer.Argument(..., help="Image paths or directory"),
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Path to model weights"
    ),
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
    pilot_name: Optional[str] = typer.Option(
        None, "--pilot", help="Pilot name for campaign metadata"
    ),
    target_species: Optional[List[str]] = typer.Option(
        None, "--species", help="Target species for detection"
    ),
    create_map: bool = typer.Option(
        True, "--map", help="Create geographic visualization map"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run wildlife census campaign with enhanced analysis."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        console.print(
            f"[bold green]Starting Wildlife Census Campaign: {campaign_id}[/bold green]"
        )

        # Determine if input is directory or file paths
        if len(images) == 1 and Path(images[0]).is_dir():
            image_dir = images[0]
            image_paths = None
            console.print(f"[green]Processing directory: {image_dir}[/green]")
        else:
            image_dir = None
            image_paths = images
            console.print(f"[green]Processing {len(images)} images[/green]")

        # Create campaign metadata
        campaign_metadata = {
            "pilot_info": {"name": pilot_name or "Unknown", "experience": "Unknown"},
            "weather_conditions": {
                "temperature": 25,
                "wind_speed": 5,
                "visibility": "good",
            },
            "mission_objectives": ["wildlife_survey", "habitat_mapping"],
            "target_species": target_species
            or ["elephant", "giraffe", "zebra", "lion"],
            "flight_parameters": {"altitude": 100, "speed": 15, "overlap": 0.7},
            "equipment_info": {"drone": "DJI Phantom 4", "camera": "20MP RGB"},
        }

        # Create campaign configuration
        pred_config = PredictionConfig(
            model_path=model_path,
            model_type="yolo",
            confidence_threshold=confidence,
            device=device,
            batch_size=batch_size,
            tilesize=tile_size,
        )

        loader_config = LoaderConfig(
            tile_size=tile_size,
            batch_size=batch_size,
            flight_specs=FlightSpecs(
                sensor_height=24.0, focal_length=35.0, flight_height=180.0
            ),
        )

        # Create campaign configuration
        campaign_config = CampaignConfig(
            campaign_id=campaign_id,
            loader_config=loader_config,
            prediction_config=pred_config,
            metadata=campaign_metadata,
            fiftyone_dataset_name=f"campaign_{campaign_id}",
        )

        # Initialize campaign manager
        campaign_manager = CampaignManager(campaign_config)

        # Run complete campaign
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running census campaign...", total=None)

            results = campaign_manager.run_complete_campaign(
                image_paths=image_paths or [],
                output_dir=output,
                tile_size=tile_size,
                overlap=0.2,
                run_flight_analysis=True,
                run_geographic_merging=True,
                create_visualization=create_map,
                export_to_fiftyone=False,
            )

            progress.update(task, completed=True)

        # Display results
        display_census_results(
            campaign_manager.census_manager, results["statistics"], output
        )

        console.print(
            f"[bold green]Census campaign completed successfully![/bold green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Census campaign failed: {e}")
        raise typer.Exit(1)


@app.command()
def analyze(
    results_path: str = typer.Argument(..., help="Path to detection results"),
    output_dir: str = typer.Option(
        "analysis", "--output", "-o", help="Output directory for analysis"
    ),
    create_map: bool = typer.Option(
        True, "--map", help="Create geographic visualization map"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Analyze detection results with geographic and statistical analysis."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        results_path_obj = Path(results_path)
        if not results_path_obj.exists():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Analyzing results from: {results_path}[/green]")

        # Load results
        with open(results_path_obj, "r") as f:
            results = json.load(f)

        # Create output directory
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        # Analyze results
        analysis_results = analyze_detection_results(results)
        display_analysis_results(analysis_results)

        # Create geographic visualization if requested
        if create_map and "drone_images" in results:
            console.print("[green]Creating geographic visualization...[/green]")
            create_geographic_visualization(results["drone_images"], output_dir)

        # Export analysis report
        export_analysis_report(analysis_results, output_dir)

        console.print(
            f"[bold green]Analysis completed! Results saved to: {output_dir}[/bold green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Analysis failed: {e}")
        raise typer.Exit(1)


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
        # Get all predictions and filter out empty ones
        all_predictions = drone_image.get_all_predictions()
        non_empty_predictions = [det for det in all_predictions if not det.is_empty]

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
            if any(not det.is_empty for det in drone_image.get_all_predictions())
        ]

        console.print(f"Images with detections: {len(images_with_detections)}")
        console.print(
            f"Images with GPS: {len([img for img in images_with_detections if img.latitude and img.longitude])}"
        )

        if images_with_detections:
            try:
                # Create geographic visualization
                config = VisualizationConfig(
                    map_center=None,
                    zoom_start=12,
                    tiles="OpenStreetMap",  # Use OpenStreetMap instead of Stamen Terrain
                    image_bounds_color="purple",
                    image_center_color="orange",
                    overlap_color="red",
                    show_image_path=False,
                    show_image_bounds=True,
                    show_detection_count=True,
                    show_gps_info=True,
                    show_image_centers=True,
                    show_statistics=True,
                )
                map_file = str(output_path / "geographic_visualization.html")

                map_obj = visualize_geographic_bounds(
                    drone_images=images_with_detections,
                    output_path=map_file,
                    config=config,
                )
                console.print(
                    f"[green]✓ Geographic visualization saved to: {map_file}[/green]"
                )

                # Get coverage statistics
                # coverage_stats = visualizer.get_coverage_statistics(images_with_detections)
                # console.print(f"[green]✓ Coverage statistics calculated[/green]")
                # console.print(f"  Images with GPS: {coverage_stats['images_with_gps']}")
                # console.print(f"  Images with footprints: {coverage_stats['images_with_footprints']}")
                # if coverage_stats['total_overlap_area'] > 0:
                #    console.print(f"  Total overlap area: {coverage_stats['total_overlap_area']:.2f} m²")

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
    census_manager: CensusDataManager, stats: dict, output_dir: Optional[str] = None
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

    # Detection results if available
    if census_manager.drone_images:
        detection_stats = get_detection_statistics(census_manager.drone_images)
        if detection_stats["total_detections"] > 0:
            console.print(f"\n[bold green]Detection Results:[/bold green]")
            console.print(f"  Total detections: {detection_stats['total_detections']}")
            console.print(
                f"  Species detected: {len(detection_stats['species_counts'])}"
            )

            if detection_stats["species_counts"]:
                console.print(f"  Species breakdown:")
                for species, count in sorted(
                    detection_stats["species_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    console.print(f"    {species}: {count}")

    # Geographic coverage
    if census_manager.drone_images:
        geo_stats = get_geographic_coverage(census_manager.drone_images)
        console.print(f"\n[bold green]Geographic Coverage:[/bold green]")
        console.print(f"  Images with GPS: {geo_stats['images_with_gps']}")
        console.print(
            f"  Images with footprints: {geo_stats['images_with_footprints']}"
        )

        if geo_stats["geographic_bounds"]:
            bounds = geo_stats["geographic_bounds"]
            console.print(f"  Coverage bounds:")
            console.print(
                f"    Latitude: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}"
            )
            console.print(
                f"    Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}"
            )


def get_detection_statistics(drone_images: List) -> dict:
    """Calculate detection statistics from drone images."""
    total_detections = 0
    species_counts = {}

    for drone_image in drone_images:
        detections = drone_image.get_all_predictions()
        total_detections += len(detections)

        for detection in detections:
            if not detection.is_empty:
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


def create_geographic_visualization(
    drone_images: List, output_dir: Optional[str] = None
):
    """Create geographic visualization of drone images."""
    if not drone_images:
        console.print("[yellow]No drone images available for visualization[/yellow]")
        return

    try:
        # Create visualizer
        config = VisualizationConfig(
            show_image_bounds=True,
            show_image_centers=True,
            show_statistics=True,
        )
        visualizer = GeographicVisualizer(config)

        # Create map
        map_obj = visualizer.create_map(drone_images)

        # Save map if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            map_file = output_path / "geographic_visualization.html"
            map_obj.save(str(map_file))
            console.print(
                f"[green]Geographic visualization saved to: {map_file}[/green]"
            )

        # Display coverage statistics
        coverage_stats = visualizer.get_coverage_statistics(drone_images)
        console.print(f"\n[bold green]Coverage Statistics:[/bold green]")
        console.print(f"  Total images: {coverage_stats['total_images']}")
        console.print(f"  Images with GPS: {coverage_stats['images_with_gps']}")
        console.print(
            f"  Images with footprints: {coverage_stats['images_with_footprints']}"
        )
        console.print(f"  Total detections: {coverage_stats['total_detections']}")
        console.print(f"  Detections with GPS: {coverage_stats['detections_with_gps']}")

    except Exception as e:
        console.print(f"[red]Failed to create geographic visualization: {e}[/red]")


def export_campaign_report(census_manager: CensusDataManager, output_dir: str):
    """Export comprehensive campaign report."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate report data
        report = {
            "campaign_id": census_manager.campaign_id,
            "metadata": census_manager.metadata,
            "statistics": census_manager.get_enhanced_campaign_statistics(),
            "timestamp": datetime.now().isoformat(),
        }

        # Add detection statistics if available
        if census_manager.drone_images:
            report["detection_statistics"] = get_detection_statistics(
                census_manager.drone_images
            )
            report["geographic_coverage"] = get_geographic_coverage(
                census_manager.drone_images
            )

        # Save report
        report_file = output_path / "campaign_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"[green]Campaign report exported to: {report_file}[/green]")

    except Exception as e:
        console.print(f"[red]Failed to export campaign report: {e}[/red]")


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
        import subprocess
        import sys

        # Get the path to the UI module
        ui_path = Path(__file__).parent / "ui" / "main.py"

        if not ui_path.exists():
            console.print(f"[red]UI module not found at: {ui_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Launching WildDetect UI on http://{host}:{port}[/green]")
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

        # Launch Streamlit
        cmd = [
            sys.executable,
            "-m",
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

        subprocess.run(cmd)

    except ImportError:
        console.print("[red]Streamlit not installed. Please install it with:[/red]")
        console.print("[yellow]pip install streamlit[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error launching UI: {e}[/red]")
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


if __name__ == "__main__":
    app()
