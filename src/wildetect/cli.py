"""
Command Line Interface for WildDetect using Typer.
"""

import cProfile
import importlib.metadata
import json
import logging
import pstats
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import torch
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core.campaign_manager import CampaignConfig, CampaignManager
from .core.config import ROOT, FlightSpecs, LoaderConfig, PredictionConfig
from .core.data.census import CensusDataManager
from .core.data.drone_image import DroneImage
from .core.data.utils import get_images_paths
from .core.detection_pipeline import DetectionPipeline, MultiThreadedDetectionPipeline
from .core.visualization.fiftyone_manager import FiftyOneManager
from .core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)

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
    __version__ = "0.0.1"


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


def parse_cls_label_map(cls_label_map: List[str]) -> Dict[int, str]:
    """Parse a list of key:value strings into a dictionary with int keys and str values."""
    result = {}
    for item in cls_label_map:
        key, value = item.split(":", 1)
        result[int(key)] = value
    return result


@app.command()
def detect(
    images: List[str] = typer.Argument(..., help="Image paths or directory"),
    model_path: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to model weights. Otherwise uses WILDETECT_MODEL_PATH environment variable",
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Name of the FiftyOne dataset to save detections to",
    ),
    model_type: str = typer.Option("yolo", "--type", "-t", help="Model type"),
    confidence: float = typer.Option(
        0.2, "--confidence", "-c", help="Confidence threshold"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device to run inference on. auto, cpu or cuda"
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    tile_size: int = typer.Option(800, "--tile-size", help="Tile size for processing"),
    output: Optional[str] = typer.Option(
        "results", "--output", "-o", help="Output directory for results"
    ),
    roi_weights: Optional[str] = typer.Option(
        None,
        "--roi-weights",
        help="Path to ROI weights file. Otherwise uses ROI_MODEL_PATH environment variable",
    ),
    feature_extractor_path: Optional[str] = typer.Option(
        "facebook/dinov2-with-registers-small",
        "--feature-extractor-path",
        help="Path to feature extractor",
    ),
    cls_label_map: List[str] = typer.Option(
        ["0:groundtruth", "1:other"],
        "--cls-label-map",
        help="Label map as key:value pairs",
    ),
    keep_classes: List[str] = typer.Option(
        ["groundtruth"], "--keep-classes", help="Classes to keep"
    ),
    cls_imgsz: int = typer.Option(128, "--cls-imgsz", help="Image size for classifier"),
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
    profile: bool = typer.Option(False, "--profile", help="Enable detailed profiling"),
    memory_profile: bool = typer.Option(
        False, "--memory-profile", help="Enable memory profiling"
    ),
    line_profile: bool = typer.Option(
        False, "--line-profile", help="Enable line-by-line profiling"
    ),
    gpu_profile: bool = typer.Option(
        False, "--gpu-profile", help="Enable GPU profiling (CUDA only)"
    ),
    pipeline_type: str = typer.Option(
        "single",
        "--pipeline-type",
        help="Pipeline type: 'single' or 'multi' for single-threaded vs multi-threaded",
    ),
    queue_size: int = typer.Option(
        3, "--queue-size", help="Queue size for multi-threaded pipeline"
    ),
):
    """Run wildlife detection on images."""
    setup_logging(
        verbose,
        log_file=str(
            ROOT / "logs" / "detect" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    )
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

        # Parse cls_label_map
        label_map_dict = parse_cls_label_map(cls_label_map)

        for class_name in keep_classes:
            if class_name not in label_map_dict.values():
                raise ValueError(
                    f"Class {class_name} not found in cls_label_map. Please check the cls_label_map argument."
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
            keep_classes=keep_classes,
            feature_extractor_path=feature_extractor_path,
            cls_label_map=label_map_dict,
            inference_service_url=inference_service_url,
            verbose=verbose,
            nms_iou=nms_iou,
            overlap_ratio=overlap_ratio,
            pipeline_type=pipeline_type,
            queue_size=queue_size,
        )

        loader_config = LoaderConfig(
            tile_size=tile_size,
            batch_size=batch_size,
            num_workers=0,
            flight_specs=flight_specs,
        )

        console.print(pred_config)

        # Create detection pipeline based on configuration
        if pred_config.pipeline_type == "multi":
            pipeline = MultiThreadedDetectionPipeline(
                config=pred_config,
                loader_config=loader_config,
                queue_size=pred_config.queue_size,
            )
        else:
            pipeline = DetectionPipeline(
                config=pred_config,
                loader_config=loader_config,
            )
        save_path = None
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / "results.json")

        # Profile the detection pipeline
        start_time = time.time()

        if profile:
            # Enable detailed profiling
            profiler = cProfile.Profile()
            profiler.enable()

        if memory_profile:
            try:
                from memory_profiler import profile as memory_profile_decorator

                print("Memory profiling enabled - this will be slower")
            except ImportError:
                print(
                    "memory_profiler not installed. Install with: pip install memory_profiler"
                )
                memory_profile = False

        if line_profile:
            try:
                from line_profiler import LineProfiler

                print("Line profiling enabled - this will be slower")
                line_profiler = LineProfiler()
                # Profile the run_detection method
                line_profiler.add_function(pipeline.run_detection)
                line_profiler.enable_by_count()
            except ImportError:
                print(
                    "line_profiler not installed. Install with: pip install line_profiler"
                )
                line_profile = False

        if gpu_profile:
            try:
                import torch

                if torch.cuda.is_available():
                    print("GPU profiling enabled")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                else:
                    print("CUDA not available, GPU profiling disabled")
                    gpu_profile = False
            except ImportError:
                print("PyTorch not available, GPU profiling disabled")
                gpu_profile = False

        # Run pipeline
        drone_images = pipeline.run_detection(
            image_paths=image_paths,
            image_dir=image_dir,
            save_path=save_path,
        )
        if dataset_name:
            fo_manager = FiftyOneManager(dataset_name, persistent=True)
            fo_manager.add_drone_images(drone_images)
            fo_manager.save_dataset()

        try:
            annot_key = f"{dataset_name}_review"
            fo_manager.send_predictions_to_labelstudio(
                annot_key, dotenv_path=str(Path(ROOT) / ".env")
            )
            logger.info(
                f"Exported FiftyOne dataset to LabelStudio with annot_key: {annot_key}"
            )
        except Exception as e:
            logger.error(f"Error exporting to LabelStudio: {e}")

        if gpu_profile and torch.cuda.is_available():
            print(
                f"GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"GPU Memory Current: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        if line_profile and "line_profiler" in locals():
            line_profiler.disable_by_count()
            line_profiler.print_stats()
            if output:
                line_profile_path = Path(output) / "line_profile_results.txt"
                with open(line_profile_path, "w") as f:
                    line_profiler.print_stats(stream=f)
                print(f"Line profile saved to: {line_profile_path}")

        if profile:
            # Disable profiler and save results
            profiler.disable()
            stats = pstats.Stats(profiler)

            # Save profiling results
            profile_path = (
                Path(output) / "profile_results.prof"
                if output
                else Path("profile_results.prof")
            )
            stats.dump_stats(str(profile_path))

            # Print top 20 functions by cumulative time
            print("\n=== PROFILING RESULTS ===")
            stats.sort_stats("cumulative")
            stats.print_stats(20)
            print(f"Detailed profile saved to: {profile_path}")
            print(
                "To view with snakeviz: pip install snakeviz && snakeviz", profile_path
            )

        end_time = time.time()

        # Log timing information
        execution_time = end_time - start_time
        logger.info(f"Detection pipeline execution time: {execution_time:.2f} seconds")
        print(f"Detection completed in {execution_time:.2f} seconds")

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
        "results", "--output", "-o", help="Output directory for visualizations"
    ),
    create_map: bool = typer.Option(
        True, "--map", help="Create geographic visualization map"
    ),
    show_confidence: bool = typer.Option(
        False, "--show-confidence", help="Show confidence scores in visualization"
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
    def get_pyproject_dependencies(pyproject_path="pyproject.toml"):
        """
        Reads the pyproject.toml file and extracts dependencies and their versions.

        Returns:
            List of tuples: (dependency_name, import_name, version)
        """
        import re
        import tomllib

        # Map package names to import names if they differ
        import_name_map = {
            "Pillow": "PIL",
            "python-dotenv": "dotenv",
            "fiftyone-brain": "fiftyone.brain",
            "spyder-kernels": "spyder_kernels",
        }

        try:
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            deps = pyproject["project"]["dependencies"]
        except Exception:
            traceback.print_exc()
            # Fallback: hardcoded list if file not found or tomllib not available
            deps = [
                "fiftyone>=1.7.0",
                "fiftyone-brain>=0.21.2",
                "folium>=0.20.0",
                "geopy>=2.4.1",
                "huggingface",
                "pillow>=11.3.0",
                "pyproj>=3.7.1",
                "pytest>=8.4.1",
                "ruff>=0.1.6",
                "shapely>=2.1.1",
                "spyder-kernels==3.0.*",
                "torch==2.6.0",
                "torchmetrics>=1.7.4",
                "ultralytics>=8.3.162",
                "utm>=0.8.1",
                "tqdm>=4.65.0",
                "numpy>=1.24.0",
                "torchvision>=0.15.0",
                "python-dotenv>=1.0.0",
                "transformers>=4.53.1",
                "accelerate>=1.8.1",
                "mlflow>=3.1.1",
                "streamlit>=1.46.1",
                "isort>=6.0.1",
            ]

        dep_list = []
        for dep in deps:
            # Extract name and version
            match = re.match(r"([a-zA-Z0-9_\-]+)([<>=!~].*)?", dep)
            if match:
                name = match.group(1)
                version = match.group(2) if match.group(2) else ""
                import_name = import_name_map.get(name.lower(), name.replace("-", "_"))
                dep_list.append((name, import_name, version))
        return dep_list

    dependencies = [
        (name, import_name)
        for name, import_name, version in get_pyproject_dependencies()
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
    roi_weights: Optional[str] = typer.Option(
        None,
        "--roi-weights",
        help="Path to ROI weights file. Otherwise uses ROI_MODEL_PATH environment variable",
    ),
    feature_extractor_path: str = typer.Option(
        "facebook/dinov2-with-registers-small",
        "--feature-extractor-path",
        help="Path to feature extractor",
    ),
    cls_label_map: List[str] = typer.Option(
        ["0:groundtruth", "1:other"],
        "--cls-label-map",
        help="Label map as key:value pairs",
    ),
    keep_classes: List[str] = typer.Option(
        ["groundtruth"], "--keep-classes", help="Classes to keep"
    ),
    confidence: float = typer.Option(
        0.2, "--confidence", "-c", help="Confidence threshold"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to run inference on"
    ),
    inference_service_url: Optional[str] = typer.Option(
        None, "--inference-service-url", help="URL for inference service"
    ),
    nms_iou: float = typer.Option(0.5, "--nms-iou", help="NMS IoU threshold"),
    overlap_ratio: float = typer.Option(
        0.2, "--overlap-ratio", help="Tile overlap ratio"
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    tile_size: int = typer.Option(800, "--tile-size", help="Tile size for processing"),
    cls_imgsz: int = typer.Option(96, "--cls-imgsz", help="Image size for classifier"),
    sensor_height: float = typer.Option(
        24.0, "--sensor-height", help="Sensor height in mm"
    ),
    focal_length: float = typer.Option(
        35.0, "--focal-length", help="Focal length in mm"
    ),
    flight_height: float = typer.Option(
        180.0, "--flight-height", help="Flight height in meters"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    pilot_name: Optional[str] = typer.Option(
        None, "--pilot", help="Pilot name for campaign metadata"
    ),
    target_species: Optional[List[str]] = typer.Option(
        None, "--species", help="Target species for detection"
    ),
    export_to_fiftyone: bool = typer.Option(
        True, "--to-fiftyone", help="Export to FiftyOne"
    ),
    create_map: bool = typer.Option(
        True, "--map", help="Create geographic visualization map"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    pipeline_type: str = typer.Option(
        "single",
        "--pipeline-type",
        help="Pipeline type: 'single' or 'multi' for single-threaded vs multi-threaded",
    ),
    queue_size: int = typer.Option(
        3, "--queue-size", help="Queue size for multi-threaded pipeline"
    ),
):
    """Run wildlife census campaign with enhanced analysis."""
    setup_logging(
        verbose,
        log_file=str(
            ROOT
            / "logs"
            / "census"
            / f"{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    )
    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        console.print(
            f"[bold red]CUDA is not available. It will be very slow...[/bold red]"
        )
    else:
        console.print(f"[bold green]CUDA is available...[/bold green]")

    try:
        console.print(
            f"[bold green]Starting Wildlife Census Campaign: {campaign_id}[/bold green]"
        )

        # Determine if input is directory or file paths
        if len(images) == 1 and Path(images[0]).is_dir():
            image_dir = images[0]
            image_paths = get_images_paths(image_dir)
            console.print(f"[green]Processing directory: {image_dir}[/green]")
        else:
            image_dir = None
            image_paths = images
            console.print(f"[green]Processing {len(images)} images[/green]")

        flight_specs = FlightSpecs(
            sensor_height=sensor_height,
            focal_length=focal_length,
            flight_height=flight_height,
        )

        # Parse cls_label_map
        label_map_dict = parse_cls_label_map(cls_label_map)
        for class_name in keep_classes:
            if class_name not in label_map_dict.values():
                raise ValueError(
                    f"Class {class_name} not found in cls_label_map. Please check the cls_label_map argument."
                )

        # Create campaign metadata
        campaign_metadata = {
            "pilot_info": {"name": pilot_name or "Unknown", "experience": "Unknown"},
            "weather_conditions": {},
            "mission_objectives": [
                "wildlife_survey",
            ],
            "target_species": target_species,
            "flight_parameters": vars(flight_specs),
            "equipment_info": {},
        }

        # Create configurations
        pred_config = PredictionConfig(
            model_path=model_path,
            confidence_threshold=confidence,
            device=device,
            batch_size=batch_size,
            tilesize=tile_size,
            flight_specs=flight_specs,
            roi_weights=roi_weights,
            cls_imgsz=cls_imgsz,
            keep_classes=keep_classes,
            feature_extractor_path=feature_extractor_path,
            cls_label_map=label_map_dict,
            inference_service_url=inference_service_url,
            verbose=verbose,
            nms_iou=nms_iou,
            overlap_ratio=overlap_ratio,
            pipeline_type=pipeline_type,
            queue_size=queue_size,
        )

        loader_config = LoaderConfig(
            tile_size=tile_size,
            batch_size=batch_size,
            num_workers=0,
            flight_specs=flight_specs,
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
                image_paths=image_paths,
                output_dir=output,
                export_to_fiftyone=export_to_fiftyone,
            )

            progress.update(task, completed=True)

        # Display results
        display_census_results(
            campaign_manager.census_manager,
            results["statistics"],
        )

        console.print(
            f"[bold green]Census campaign completed successfully![/bold green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Census campaign failed: {traceback.format_exc()}")
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
    pipeline_type: str = typer.Option(
        "single",
        "--pipeline-type",
        help="Pipeline type: 'single' or 'multi' for single-threaded vs multi-threaded",
    ),
    queue_size: int = typer.Option(
        3, "--queue-size", help="Queue size for multi-threaded pipeline"
    ),
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
        # if create_map and "drone_images" in results:
        # console.print("[green]Creating geographic visualization...[/green]")
        # create_geographic_visualization(results["drone_images"], output_dir)

        # Export analysis report
        export_analysis_report(analysis_results, output_dir)

        console.print(
            f"[bold green]Analysis completed! Results saved to: {output_dir}[/bold green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Analysis failed: {e}")
        raise typer.Exit(1)


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


def create_geographic_visualization(
    drone_images: List[DroneImage], output_dir: Optional[str] = None
):
    """Create geographic visualization of drone images."""
    if not drone_images:
        console.print("[yellow]No drone images available for visualization[/yellow]")
        return

    try:
        # Create visualizer
        config = VisualizationConfig()
        visualizer = GeographicVisualizer(config)
        map_file = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            map_file = str(output_path / "geographic_visualization.html")

        # Create map
        visualizer.create_map(drone_images, save_path=map_file)

        if map_file:
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
        import os
        import subprocess

        # Get the path to the UI module
        ui_path = Path(__file__).parent / "ui" / "main.py"

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

        from .api.main import app

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

        # Import the run_inference_server function
        from .core.detectors.detection_server import run_inference_server

        # Launch the inference server
        run_inference_server(port=port, workers_per_device=workers_per_device)

    except Exception as e:
        console.print(f"[red]Error starting inference server: {e}[/red]")
        logger.error(f"Inference server failed: {e}")
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
    from .utils.cuda_installer import install_cpu_torch, install_cuda_torch

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


if __name__ == "__main__":
    app()
