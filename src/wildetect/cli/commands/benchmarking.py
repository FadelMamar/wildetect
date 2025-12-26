import logging
from pathlib import Path
from typing import Optional

import typer
import yaml

# from ...core.config_loader import ConfigLoader
from ...core.config_models import BenchmarkConfigModel
from ...core.data.utils import get_images_paths
from ...utils.benchmark import BenchmarkPipeline

# Configure logging
logger = logging.getLogger(__name__)

app = typer.Typer(name="benchmarking", help="Benchmarking commands for WildDetect")


@app.command()
def detection(
    config: str = typer.Option(
        ..., "-c", "--config", help="Path to benchmark configuration file"
    ),
):
    """
    Run detection pipeline benchmarking with hyperparameter optimization.

    This command will:
    1. Load the benchmark configuration from the specified YAML file
    2. Find test images in the configured directory
    3. Run Optuna optimization to find the best hyperparameters
    4. Save results and optionally generate performance plots

    Example:
        wildetect benchmarking detection --config config/benchmark.yaml
        wildetect benchmarking detection --config config/benchmark.yaml --trials 50 --timeout 7200
    """

    try:
        # Load and validate configuration
        logger.info(f"Loading benchmark configuration from: {config}")

        config_data = _load_config(config)

        # Validate the final configuration
        benchmark_config = BenchmarkConfigModel(**config_data)

        # Find test images
        image_paths = benchmark_config.get_image_paths()
        if not image_paths:
            typer.echo(
                f"❌ No test images found. Please check your test images configuration. ``{benchmark_config.test_images}``"
            )
            raise typer.Exit(1)

        logger.info(f"Found {len(image_paths)} test images for benchmarking")

        # Create output directory
        output_path = Path(benchmark_config.output.directory)
        output_path.mkdir(parents=True, exist_ok=True)

        benchmark_pipeline = BenchmarkPipeline.from_config(benchmark_config)

        # Run benchmarking
        typer.echo("🚀 Starting detection pipeline benchmarking...")
        typer.echo(
            f"📊 Configuration: {benchmark_config.execution.n_trials} trials, "
            f"{benchmark_config.execution.timeout}s timeout, "
            f"{benchmark_config.execution.direction} direction"
        )
        typer.echo(f"🖼️  Test images: {len(image_paths)} images")
        typer.echo(f"📁 Output directory: {output_path}")

        # Execute the benchmark
        benchmark_pipeline.run(
            n_trials=benchmark_config.execution.n_trials,
            timeout=benchmark_config.execution.timeout,
            direction=benchmark_config.execution.direction,
        )

        # Save results if requested
        if benchmark_config.output.save_results:
            _save_benchmark_results(benchmark_pipeline, benchmark_config, output_path)

        typer.echo("✅ Benchmarking completed successfully!")

    except FileNotFoundError as e:
        typer.echo(f"❌ Configuration file not found: {e}")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        typer.echo(f"❌ Invalid YAML configuration: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"❌ Configuration validation error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during benchmarking: {e}")
        typer.echo(f"❌ Benchmarking failed: {e}")
        raise typer.Exit(1)


def _load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_benchmark_results(
    benchmark_pipeline: BenchmarkPipeline,
    benchmark_config: BenchmarkConfigModel,
    output_path: Path,
):
    """Save benchmark results to the specified output directory."""
    try:
        # This would be implemented to save results in the specified format
        # For now, just log that we would save results
        logger.info(f"Results would be saved to: {output_path}")

        # TODO: Implement actual result saving based on benchmark_config.output.format
        # - JSON results
        # - CSV data
        # - Performance plots
        # - Optimization history

    except Exception as e:
        logger.warning(f"Failed to save benchmark results: {e}")
        typer.echo(f"⚠️  Warning: Failed to save results: {e}")


if __name__ == "__main__":
    app()
