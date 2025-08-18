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

from ...core.campaign_manager import CampaignConfig, CampaignManager
from ...core.config import ROOT
from ...core.config_loader import load_config_from_yaml, load_config_with_pydantic
from ...core.data.utils import get_images_paths
from ...core.detection_pipeline import DetectionPipeline, MultiThreadedDetectionPipeline
from ...core.visualization.fiftyone_manager import FiftyOneManager
from ..utils import (
    analyze_detection_results,
    display_analysis_results,
    display_census_results,
    display_results,
    export_analysis_report,
    setup_logging,
)

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
    setup_logging()
    logger = logging.getLogger(__name__)

    log_file = str(
        ROOT / "logs" / "detect" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(log_file)

    try:
        # Load configuration from YAML
        loaded_config = load_config_with_pydantic("detect", config)
        if images is None:
            images = list(loaded_config.images)

        if profile:
            loaded_config.profiling.enable = True
        if memory_profile:
            loaded_config.profiling.memory_profile = True
        if line_profile:
            loaded_config.profiling.line_profile = True
        if gpu_profile:
            loaded_config.profiling.gpu_profile = True

        # Convert to existing dataclasses
        pred_config = loaded_config.to_prediction_config()
        loader_config = loaded_config.to_loader_config()

        # Set output directory
        output_dir = Path(loaded_config.output.directory)
        dataset_name = loaded_config.output.dataset_name

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

        if loaded_config.profiling.gpu_profile and torch.cuda.is_available():
            print(
                f"GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"GPU Memory Current: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

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
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
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
        loaded_config = load_config_with_pydantic("census", config)
        campaign_id = loaded_config.campaign.id
        images = loaded_config.detection.images

        if profile:
            loaded_config.detection.profiling.enable = True
        if memory_profile:
            loaded_config.detection.profiling.memory_profile = True
        if line_profile:
            loaded_config.detection.profiling.line_profile = True
        if gpu_profile:
            loaded_config.detection.profiling.gpu_profile = True

        # Convert to existing dataclasses
        pred_config = loaded_config.detection.to_prediction_config()
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

        console.print(
            f"[bold green]Starting Wildlife Census Campaign: {campaign_id}[/bold green]"
        )

        log_file = str(
            ROOT
            / "logs"
            / "census"
            / f"{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        setup_logging(
            loaded_config.logging.verbose,
            log_file=log_file,
        )

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
