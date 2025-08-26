import logging
from pathlib import Path
from typing import Optional

import typer
import yaml

from ...core.config_loader import ConfigLoader
from ...core.config_models import BenchmarkConfigModel, validate_config_dict
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
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", help="Override output directory for results"
    ),
    trials: Optional[int] = typer.Option(
        None, "--trials", help="Override number of optimization trials"
    ),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", help="Override timeout in seconds"
    ),
    direction: Optional[str] = typer.Option(
        None, "--direction", help="Override optimization direction (minimize/maximize)"
    ),
    save_results: bool = typer.Option(
        True,
        "--save-results/--no-save-results",
        help="Whether to save detailed results",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
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

        # Apply CLI overrides
        config_data = _apply_cli_overrides(
            config_data, output_dir, trials, timeout, direction, save_results, verbose
        )

        # Validate the final configuration
        benchmark_config = validate_config_dict(config_data, "benchmark")

        # Find test images
        image_paths = _find_test_images(benchmark_config)
        if not image_paths:
            typer.echo(
                "❌ No test images found. Please check the test_images.path configuration."
            )
            raise typer.Exit(1)

        logger.info(f"Found {len(image_paths)} test images for benchmarking")

        # Create output directory
        output_path = Path(benchmark_config.output.directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize BenchmarkPipeline
        prediction_config = benchmark_config.to_prediction_config(verbose=verbose)
        loader_config = benchmark_config.to_loader_config()

        benchmark_pipeline = BenchmarkPipeline(
            image_paths=image_paths,
            prediction_config=prediction_config,
            loader_config=loader_config,
        )

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
        if save_results:
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


def _apply_cli_overrides(
    config_data: dict,
    output_dir: Optional[str],
    trials: Optional[int],
    timeout: Optional[int],
    direction: Optional[str],
    save_results: bool,
    verbose: bool,
) -> dict:
    """Apply CLI argument overrides to configuration."""

    # Create a copy to avoid modifying the original
    config = config_data.copy()

    if output_dir:
        if "output" not in config:
            config["output"] = {}
        config["output"]["directory"] = output_dir

    if trials:
        if "execution" not in config:
            config["execution"] = {}
        config["execution"]["n_trials"] = trials

    if timeout:
        if "execution" not in config:
            config["execution"] = {}
        config["execution"]["timeout"] = timeout

    if direction:
        if direction not in ["minimize", "maximize"]:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'minimize' or 'maximize'"
            )
        if "execution" not in config:
            config["execution"] = {}
        config["execution"]["direction"] = direction

    if not save_results:
        if "output" not in config:
            config["output"] = {}
        config["output"]["save_results"] = False

    if verbose:
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["verbose"] = True

    return config


def _find_test_images(benchmark_config: BenchmarkConfigModel) -> list:
    """Find test images based on configuration."""
    test_images_config = benchmark_config.test_images

    # Use the existing utility function to find image files
    # Convert supported formats to glob patterns
    patterns = tuple(test_images_config.supported_formats)

    image_paths = get_images_paths(
        images_dir=test_images_config.path, patterns=patterns
    )

    # Limit to max_images if specified
    if (
        test_images_config.max_images
        and len(image_paths) > test_images_config.max_images
    ):
        image_paths = image_paths[: test_images_config.max_images]

    return image_paths


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
