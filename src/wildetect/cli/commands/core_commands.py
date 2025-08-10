"""
Core CLI commands for the wildetect application.
"""
import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import typer
from rich.console import Console

from wildetect.cli.utils import (
    analyze_detection_results,
    display_analysis_results,
    display_census_results,
    display_results,
    export_analysis_report,
    setup_logging,
)
from wildetect.core.campaign_manager import CampaignConfig, CampaignManager
from wildetect.core.config_loader import load_config_with_pydantic
from wildetect.core.config_models import FlightSpecs, LoaderConfig, PredictionConfig
from wildetect.core.data.utils import get_images_paths
from wildetect.core.detection_pipeline import (
    AsyncDetectionPipeline,
    DetectionPipeline,
    MultiThreadedDetectionPipeline,
)
from wildetect.core.visualization.fiftyone_manager import FiftyOneManager
from wildetect.utils.profiler import profile_command
from wildetect.utils.utils import ROOT

app = typer.Typer()
console = Console()


@app.command()
def detect(
    images: Optional[List[str]] = typer.Option(
        None, "--images", "-i", help="Image paths or directory"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Override output directory"
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
        16, "--batch-size", "-b", help="Override batch size"
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

    log_file = str(
        ROOT / "logs" / "detect" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    setup_logging(verbose, log_file)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration from YAML
        if config:
            loaded_config = load_config_with_pydantic("detect", config)
            # For detect command, we need to get image directory from config or use provided images
            if images is None and hasattr(loaded_config, "images"):
                images = list(loaded_config.images)

            # Apply command-line overrides
            if model_path and hasattr(loaded_config, "model"):
                loaded_config.model.path = model_path
            if confidence is not None and hasattr(loaded_config, "model"):
                loaded_config.model.confidence_threshold = confidence
            if device and hasattr(loaded_config, "model"):
                loaded_config.model.device = device
            if batch_size and hasattr(loaded_config, "model"):
                loaded_config.model.batch_size = batch_size
            if tile_size and hasattr(loaded_config, "processing"):
                loaded_config.processing.tile_size = tile_size
            if roi_weights and hasattr(loaded_config, "roi_classifier"):
                loaded_config.roi_classifier.weights = roi_weights
            if profile and hasattr(loaded_config, "profiling"):
                loaded_config.profiling.enable = True
            if memory_profile and hasattr(loaded_config, "profiling"):
                loaded_config.profiling.memory_profile = True
            if line_profile and hasattr(loaded_config, "profiling"):
                loaded_config.profiling.line_profile = True
            if gpu_profile and hasattr(loaded_config, "profiling"):
                loaded_config.profiling.gpu_profile = True
            if output and hasattr(loaded_config, "output"):
                loaded_config.output.directory = output
            if dataset_name and hasattr(loaded_config, "output"):
                loaded_config.output.dataset_name = dataset_name

            # Convert to existing dataclasses
            pred_config = loaded_config.to_prediction_config(verbose=verbose)
            loader_config = loaded_config.to_loader_config()

            # Set output directory
            output_dir = (
                Path(loaded_config.output.directory)
                if hasattr(loaded_config, "output")
                else Path(output or "results")
            )
            dataset_name = (
                loaded_config.output.dataset_name
                if hasattr(loaded_config, "output")
                else dataset_name
            )
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
                verbose=verbose,
                nms_iou=0.5,
                overlap_ratio=0.2,
                queue_size=32,
            )

            loader_config = LoaderConfig(
                tile_size=pred_config.tilesize,
                batch_size=pred_config.batch_size,
                num_workers=0,
                flight_specs=pred_config.flight_specs,
            )

            output_dir = Path(output or "results")

        # Determine if input is directory or file paths
        logger.info(f"Processing images: {images}")
        assert isinstance(images, list), "images must be a list"
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
        console.print(f"Loader config: {loader_config}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detection pipeline
        if pred_config.pipeline_type == "multi":
            pipeline = MultiThreadedDetectionPipeline(
                pred_config,
                loader_config,
            )
        elif pred_config.pipeline_type == "async":
            # NEW: Support for async pipeline
            pipeline = AsyncDetectionPipeline(pred_config, loader_config)
        else:
            pipeline = DetectionPipeline(pred_config, loader_config)

        save_path = None
        if output_dir:
            save_path = str(output_dir / "results.json")

        # Determine profiling settings from config or command line
        profiling_enabled = (
            loaded_config.profiling.enable
            if config and hasattr(loaded_config, "profiling")
            else profile
        )
        memory_profiling_enabled = (
            loaded_config.profiling.memory_profile
            if config and hasattr(loaded_config, "profiling")
            else memory_profile
        )
        line_profiling_enabled = (
            loaded_config.profiling.line_profile
            if config and hasattr(loaded_config, "profiling")
            else line_profile
        )
        gpu_profiling_enabled = (
            loaded_config.profiling.gpu_profile
            if config and hasattr(loaded_config, "profiling")
            else gpu_profile
        )

        profiling_dir = Path(log_file).parent / f"{Path(log_file).stem}_profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)

        with profile_command(
            output_dir=profiling_dir,
            profile=profiling_enabled,
            memory_profile=memory_profiling_enabled,
            line_profile=line_profiling_enabled,
            gpu_profile=gpu_profiling_enabled,
        ) as profiler:
            # Run pipeline with profiling
            if pred_config.pipeline_type == "async":

                async def run_async_detection():
                    return await pipeline.run_detection_async(
                        image_paths=image_paths,
                        image_dir=image_dir,
                        save_path=save_path,
                    )

                drone_images = asyncio.run(run_async_detection())
            else:
                # Use line profiling if enabled
                if line_profiling_enabled:
                    drone_images = profiler.profile_function(
                        pipeline.run_detection,
                        image_paths=image_paths,
                        image_dir=image_dir,
                        save_path=save_path,
                    )
                else:
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

        # Display results
        try:
            display_results(drone_images, output_dir)
        except Exception as e:
            console.print(f"[red]Error displaying results: {e}[/red]")

        console.print(f"[bold green]Detection completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]Error during detection: {e}[/red]")
        console.print(traceback.format_exc())


@app.command()
def census(
    campaign_id: str = typer.Option("", help="Campaign identifier"),
    images: Optional[List[str]] = typer.Option(None, help="Image paths or directory"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Override output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    # Essential overrides
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override model path"
    ),
    roi_weights: Optional[str] = typer.Option(
        None, "--roi-weights", help="Override ROI weights path"
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
    pilot_name: Optional[str] = typer.Option(
        None, "--pilot", help="Override pilot name"
    ),
    target_species: Optional[List[str]] = typer.Option(
        None, "--species", help="Override target species"
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
    """Run wildlife census campaign with enhanced analysis."""

    try:
        # Load configuration from YAML
        if config:
            loaded_config = load_config_with_pydantic("census", config)

            campaign_id = loaded_config.campaign.id

            # Apply command-line overrides
            if images is None:
                images = list(loaded_config.detection.images)

            if model_path:
                loaded_config.detection.model.path = model_path
            if roi_weights:
                loaded_config.detection.roi_classifier.weights = roi_weights
            if confidence is not None:
                loaded_config.detection.model.confidence_threshold = confidence
            if device:
                loaded_config.detection.model.device = device
            if batch_size:
                loaded_config.detection.model.batch_size = batch_size
            if tile_size:
                loaded_config.detection.processing.tile_size = tile_size
            if pilot_name:
                loaded_config.campaign.pilot_name = pilot_name
            if target_species:
                loaded_config.campaign.target_species = target_species
            if profile:
                loaded_config.detection.profiling.enable = True
            if memory_profile:
                loaded_config.detection.profiling.memory_profile = True
            if line_profile:
                loaded_config.detection.profiling.line_profile = True
            if gpu_profile:
                loaded_config.detection.profiling.gpu_profile = True
            if output:
                loaded_config.export.output_directory = output

            # Convert to existing dataclasses
            pred_config = loaded_config.detection.to_prediction_config(verbose=verbose)
            loader_config = loaded_config.detection.to_loader_config()

            # Set campaign metadata
            campaign_metadata = {
                "pilot_info": {
                    "name": loaded_config.campaign.pilot_name,
                    "experience": "Unknown",
                },
                "weather_conditions": {},
                "mission_objectives": ["wildlife_survey"],
                "target_species": loaded_config.campaign.target_species,
                "flight_parameters": vars(
                    loaded_config.detection.flight_specs.to_flight_specs()
                ),
                "equipment_info": {},
            }

            # Set output directory
            output_dir = loaded_config.export.output_directory
            export_to_fiftyone = loaded_config.export.to_fiftyone

        else:
            # Use command-line parameters directly (legacy mode)
            flight_specs = FlightSpecs(
                sensor_height=24.0,
                focal_length=35.0,
                flight_height=180.0,
            )

            # Parse cls_label_map
            label_map_dict = {0: "groundtruth", 1: "other"}
            keep_classes = ["groundtruth"]

            # Create campaign metadata
            campaign_metadata = {
                "pilot_info": {
                    "name": pilot_name or "Unknown",
                    "experience": "Unknown",
                },
                "weather_conditions": {},
                "mission_objectives": ["wildlife_survey"],
                "target_species": target_species,
                "flight_parameters": vars(flight_specs),
                "equipment_info": {},
            }

            # Create configurations
            pred_config = PredictionConfig(
                model_path=model_path,
                confidence_threshold=confidence or 0.2,
                device=device or "auto",
                batch_size=batch_size or 32,
                tilesize=tile_size or 800,
                flight_specs=flight_specs,
                roi_weights=roi_weights,
                cls_imgsz=96,
                keep_classes=keep_classes,
                feature_extractor_path="facebook/dinov2-with-registers-small",
                cls_label_map=label_map_dict,
                inference_service_url=None,
                verbose=verbose,
                nms_iou=0.5,
                overlap_ratio=0.2,
                pipeline_type="single",
                queue_size=24,
            )

            loader_config = LoaderConfig(
                tile_size=tile_size or 800,
                batch_size=batch_size or 32,
                num_workers=0,
                flight_specs=flight_specs,
            )

            output_dir = output or f"results/{campaign_id}"
            export_to_fiftyone = True

        console.print(
            f"[bold green]Starting Wildlife Census Campaign: {campaign_id}[/bold green]"
        )

        log_file = str(
            ROOT
            / "logs"
            / "census"
            / f"{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        setup_logging(verbose, log_file)

        if not torch.cuda.is_available():
            console.print(
                f"[bold red]CUDA is not available. It will be very slow...[/bold red]"
            )
        else:
            console.print(f"[bold green]CUDA is available...[/bold green]")

        # Determine if input is directory or file paths
        assert isinstance(images, list), f"images must be a list. Received: {images}"
        if len(images) == 1 and Path(images[0]).is_dir():
            image_dir = images[0]
            image_paths = get_images_paths(image_dir)
            console.print(f"[green]Processing directory: {image_dir}[/green]")
        else:
            image_dir = None
            image_paths = images
            console.print(f"[green]Processing {len(images)} images[/green]")

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

        # Determine profiling settings from config or command line
        if config and hasattr(loaded_config, "profiling"):
            profiling_enabled = loaded_config.profiling.enable
            memory_profiling_enabled = loaded_config.profiling.memory_profile
            line_profiling_enabled = loaded_config.profiling.line_profile
            gpu_profiling_enabled = loaded_config.profiling.gpu_profile
        else:
            # Use command-line parameters directly
            profiling_enabled = profile
            memory_profiling_enabled = memory_profile
            line_profiling_enabled = line_profile
            gpu_profiling_enabled = gpu_profile

        profiling_dir = Path(log_file).parent / f"{Path(log_file).stem}_profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)

        with profile_command(
            output_dir=profiling_dir,
            profile=profiling_enabled,
            memory_profile=memory_profiling_enabled,
            line_profile=line_profiling_enabled,
            gpu_profile=gpu_profiling_enabled,
        ) as profiler:
            # Run campaign with profiling
            if line_profiling_enabled:
                results = profiler.profile_function(
                    campaign_manager.run_complete_campaign,
                    image_paths=image_paths,
                    output_dir=output_dir,
                    export_to_fiftyone=export_to_fiftyone,
                )
            else:
                results = campaign_manager.run_complete_campaign(
                    image_paths=image_paths,
                    output_dir=output_dir,
                    export_to_fiftyone=export_to_fiftyone,
                )

        # Display results
        display_census_results(
            campaign_manager.census_manager,
            results["statistics"],
        )

        console.print(
            f"[bold green]Census campaign completed successfully![/bold green]"
        )

    except Exception:
        console.print(f"[red]Error: {traceback.format_exc()}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    results_path: str = typer.Argument(..., help="Path to detection results"),
    output_dir: str = typer.Option(
        "analysis", "--output", "-o", help="Output directory for analysis"
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

        # Export analysis report
        export_analysis_report(analysis_results, output_dir)

        console.print(
            f"[bold green]Analysis completed! Results saved to: {output_dir}[/bold green]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Analysis failed: {e}")
        raise typer.Exit(1)
