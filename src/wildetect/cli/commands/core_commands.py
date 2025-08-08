"""
Core detection and analysis commands.
"""

import cProfile
import json
import logging
import os
import pstats
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.campaign_manager import CampaignConfig, CampaignManager
from ...core.config import ROOT, FlightSpecs, LoaderConfig, PredictionConfig
from ...core.config_loader import load_config_from_yaml, load_config_with_pydantic
from ...core.config_models import CensusConfigModel, DetectConfigModel
from ...core.data.census import CensusDataManager
from ...core.data.drone_image import DroneImage
from ...core.data.utils import get_images_paths
from ...core.detection_pipeline import DetectionPipeline, MultiThreadedDetectionPipeline
from ...core.visualization.fiftyone_manager import FiftyOneManager
from ..utils import (
    analyze_detection_results,
    display_analysis_results,
    display_census_results,
    display_results,
    export_analysis_report,
    get_detection_statistics,
    get_geographic_coverage,
    parse_cls_label_map,
    setup_logging,
)

app = typer.Typer(name="detection", help="Detection and analysis commands")
console = Console()


@app.command()
def detect(
    images: List[str] = typer.Option(
        [], "--images", "-i", help="Image paths or directory"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    output: Optional[str] = typer.Option(
        "results", "--output", "-o", help="Override output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    dataset_name: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Name of the FiftyOne dataset to save detections to",
    ),
    # Essential overrides
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override model path"
    ),
    confidence: Optional[float] = typer.Option(
        0.2, "--confidence", help="Override confidence threshold"
    ),
    device: Optional[str] = typer.Option("auto", "--device", help="Override device"),
    batch_size: Optional[int] = typer.Option(
        32, "--batch-size", "-b", help="Override batch size"
    ),
    tile_size: Optional[int] = typer.Option(
        800, "--tile-size", help="Override tile size"
    ),
    roi_weights: Optional[str] = typer.Option(
        None, "--roi-weights", help="Override ROI weights path"
    ),
    # Profiling overrides
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
):
    """Detect wildlife in images using AI models."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration from YAML
        if config:
            loaded_config = load_config_with_pydantic("detect", config)
            # For detect command, we need to get image directory from config or use provided images
            if hasattr(loaded_config, "image_dir") and loaded_config.image_dir:
                images = [loaded_config.image_dir]

            # Apply command-line overrides
            if model_path:
                loaded_config.model.path = model_path
            if confidence is not None:
                loaded_config.model.confidence_threshold = confidence
            if device:
                loaded_config.model.device = device
            if batch_size:
                loaded_config.model.batch_size = batch_size
            if tile_size:
                loaded_config.processing.tile_size = tile_size
            if roi_weights:
                loaded_config.roi_classifier.weights = roi_weights
            if profile:
                loaded_config.profiling.enable = True
            if memory_profile:
                loaded_config.profiling.memory_profile = True
            if line_profile:
                loaded_config.profiling.line_profile = True
            if gpu_profile:
                loaded_config.profiling.gpu_profile = True
            if output:
                loaded_config.output.directory = output
            if dataset_name:
                loaded_config.output.dataset_name = dataset_name

            # Convert to existing dataclasses
            pred_config = loaded_config.to_prediction_config(verbose=verbose)
            loader_config = loaded_config.to_loader_config()

            # Set output directory
            output_dir = loaded_config.output.directory
            save_results = loaded_config.output.save_results
            export_to_fiftyone = loaded_config.output.export_to_fiftyone
            dataset_name = loaded_config.output.dataset_name or "wildlife_detection"
        else:
            # Use command-line parameters directly
            pred_config = PredictionConfig(
                model_path=model_path,
                model_type="yolo",
                confidence_threshold=confidence or 0.2,
                device=device or "auto",
                batch_size=batch_size or 32,
                tilesize=tile_size or 800,
                flight_specs=FlightSpecs(
                    sensor_height=24.0,
                    focal_length=35.0,
                    flight_height=180.0,
                ),
                roi_weights=roi_weights,
                cls_imgsz=128,
                keep_classes=["groundtruth"],
                feature_extractor_path="facebook/dinov2-with-registers-small",
                cls_label_map={0: "groundtruth", 1: "other"},
                inference_service_url=None,
                verbose=verbose,
                nms_iou=0.5,
                overlap_ratio=0.2,
                pipeline_type="single",
                queue_size=32,
            )

            loader_config = LoaderConfig(
                tile_size=tile_size or 800,
                batch_size=batch_size or 32,
                num_workers=0,
                flight_specs=FlightSpecs(
                    sensor_height=24.0,
                    focal_length=35.0,
                    flight_height=180.0,
                ),
            )

            output_dir = output or "results"
            save_results = True
            export_to_fiftyone = True
            dataset_name = dataset_name or "wildlife_detection"

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

        console.print(f"Prediction config: {pred_config}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize detection pipeline
        if pred_config.pipeline_type == "multi":
            pipeline = MultiThreadedDetectionPipeline(
                pred_config, loader_config, queue_size=pred_config.queue_size
            )
        else:
            pipeline = DetectionPipeline(pred_config, loader_config)

        save_path = None
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = str(output_path / "results.json")

        # Profile the detection pipeline
        start_time = time.time()

        if loaded_config.profiling.enable:
            # Enable detailed profiling
            profiler = cProfile.Profile()
            profiler.enable()

        if loaded_config.profiling.memory_profile:
            try:
                from memory_profiler import profile as memory_profile_decorator

                print("Memory profiling enabled - this will be slower")
            except ImportError:
                print(
                    "memory_profiler not installed. Install with: pip install memory_profiler"
                )
                memory_profile = False

        if loaded_config.profiling.line_profile:
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

        if loaded_config.profiling.gpu_profile:
            try:
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
            console.print(f"[red]Error exporting to LabelStudio: {e}[/red]")

        if loaded_config.profiling.gpu_profile and torch.cuda.is_available():
            print(
                f"GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"GPU Memory Current: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        if loaded_config.profiling.line_profile and "line_profiler" in locals():
            line_profiler.disable_by_count()
            line_profiler.print_stats()
            if output:
                line_profile_path = Path(output) / "line_profile_results.txt"
                with open(line_profile_path, "w") as f:
                    line_profiler.print_stats(stream=f)
                print(f"Line profile saved to: {line_profile_path}")

        if loaded_config.profiling.enable:
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

        console.print(f"[bold green]Detection completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]Error during detection: {e}[/red]")
        console.print(traceback.format_exc())


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
        24, "--queue-size", help="Queue size for multi-threaded pipeline"
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

        console.print(f"[bold green]Campaign configuration:[/bold green]")
        console.print(campaign_config)

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

        # Export analysis report
        export_analysis_report(analysis_results, output_dir)

        console.print(
            f"[bold green]Analysis completed! Results saved to: {output_dir}[/bold green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Analysis failed: {e}")
        raise typer.Exit(1)
