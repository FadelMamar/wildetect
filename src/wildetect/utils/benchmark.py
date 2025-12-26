import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
import torch
from tqdm import tqdm

from ..core.config import LoaderConfig
from ..core.config_models import (
    BenchmarkConfigModel,
    BenchmarkFormatTypes,
    BenchmarkObjectiveTypes,
)
from ..core.data import DataLoader
from ..core.detectors import get_detection_pipeline
from .profiler import profile_command

logger = logging.getLogger(__name__)


class BenchmarkPipeline(object):
    def __init__(
        self,
        config: BenchmarkConfigModel,
    ):
        self._config = config
        assert isinstance(
            config, BenchmarkConfigModel
        ), f"config must be a BenchmarkConfigModel, but got {type(config)}"

        self.image_paths = config.get_image_paths()
        self.prediction_config = config.to_prediction_config()
        self.loader_config = config.to_loader_config()
        self.hyperparameters = config.hyperparameters
        self.study = None
        self.output_config = config.output
        self.execution_config = config.execution

    def run(
        self,
    ):
        study = optuna.create_study(
            direction=self.execution_config.direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Start the optimization process
        print("Starting Optuna optimization for inference throughput...")
        study.optimize(
            self,
            n_trials=self.execution_config.n_trials,
            timeout=self.execution_config.timeout,
        )

        # Store study for later access (e.g., for saving results)
        self.study = study

        # Output the best result
        best_trial = study.best_trial
        print("\n" + "=" * 50)
        print("Optimization completed!")
        print(f"Best trial: #{best_trial.number}")
        print(f"Best Value (Throughput): {best_trial.value:.2f} img/sec")
        print("Best Params:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # Save benchmark results
        if self.output_config.save_results:
            self._save_benchmark_results()

    def _run_benchmark(self) -> float:
        """Runs a short benchmark and returns the achieved throughput (img/sec)."""
        detection_pipeline = get_detection_pipeline(
            pipeline_type=self.prediction_config.pipeline_type,
            config=self.prediction_config,
            loader_config=self.loader_config,
        )

        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        detection_pipeline.run_detection(
            image_paths=self.image_paths, override_loading_config=False
        )

        end_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.execution_config.objective == BenchmarkObjectiveTypes.THROUGHPUT:
            return len(self.image_paths) / (end_time - start_time)
        elif self.execution_config.objective == BenchmarkObjectiveTypes.LATENCY:
            return (end_time - start_time) / len(self.image_paths)
        else:
            raise ValueError(f"Invalid objective: {self.execution_config.objective}")

    def __call__(self, trial: optuna.Trial):
        # Suggest values for the hyperparameters
        batch_size = trial.suggest_categorical(
            "batch_size", self.hyperparameters.batch_size
        )
        self.loader_config.num_workers = trial.suggest_categorical(
            "num_workers", self.hyperparameters.num_workers
        )

        self.prediction_config.tilesize = trial.suggest_categorical(
            "tile_size", self.hyperparameters.tile_size
        )
        self.prediction_config.overlap_ratio = trial.suggest_categorical(
            "overlap_ratio", self.hyperparameters.overlap_ratio
        )

        self.prediction_config.batch_size = batch_size
        self.loader_config.batch_size = batch_size
        return self._run_benchmark()

    def _save_benchmark_results(
        self,
    ):
        """Save benchmark results to the specified output directory."""
        try:
            # Ensure output directory exists
            output_path = Path(self.output_config.directory)
            output_path.mkdir(parents=True, exist_ok=True)

            # Check if study is available
            if not hasattr(self, "study") or self.study is None:
                logger.warning(
                    "No study found in benchmark pipeline. Results cannot be saved."
                )
                return

            study = self.study
            output_format = self.output_config.format

            # Extract best trial data
            best_trial = study.best_trial
            best_trial_data = {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            }

            # Extract all trials data if requested
            all_trials_data = []
            if self.output_config.include_optimization_history:
                for trial in study.trials:
                    trial_data = {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "state": trial.state.name if trial.state else None,
                    }
                    all_trials_data.append(trial_data)

            # Study metadata
            study_metadata = {
                "direction": study.direction.name,
                "n_trials": len(study.trials),
                "best_trial_number": best_trial.number,
                "best_value": best_trial.value,
            }

            # Save JSON format if requested
            if output_format in (BenchmarkFormatTypes.JSON, BenchmarkFormatTypes.BOTH):
                try:
                    json_data = {
                        "metadata": study_metadata,
                        "best_trial": best_trial_data,
                    }
                    if self.output_config.include_optimization_history:
                        json_data["all_trials"] = all_trials_data

                    json_file = output_path / "benchmark_results.json"
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)

                    logger.info(f"Saved JSON results to: {json_file}")
                except Exception as e:
                    logger.warning(f"Failed to save JSON results: {e}")

            # Save CSV format if requested (CSV always includes all trials as it's tabular data)
            if output_format in (BenchmarkFormatTypes.CSV, BenchmarkFormatTypes.BOTH):
                try:
                    csv_file = output_path / "benchmark_results.csv"
                    with open(csv_file, "w", newline="", encoding="utf-8") as f:
                        all_trials_for_csv = []
                        for trial in study.trials:
                            trial_data = {
                                "number": trial.number,
                                "value": trial.value,
                                "params": trial.params,
                            }
                            all_trials_for_csv.append(trial_data)

                        if all_trials_for_csv:
                            # Write header
                            fieldnames = ["trial_number", "value"]
                            # Add param columns (use first trial to get param names)
                            param_names = list(all_trials_for_csv[0]["params"].keys())
                            fieldnames.extend(param_names)

                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()

                            # Write trial data
                            for trial_data in all_trials_for_csv:
                                row = {
                                    "trial_number": trial_data["number"],
                                    "value": trial_data["value"],
                                }
                                row.update(trial_data["params"])
                                writer.writerow(row)
                        else:
                            # Fallback: just write best trial if no trials available
                            fieldnames = ["trial_number", "value"]
                            param_names = list(best_trial_data["params"].keys())
                            fieldnames.extend(param_names)

                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            row = {
                                "trial_number": best_trial_data["number"],
                                "value": best_trial_data["value"],
                            }
                            row.update(best_trial_data["params"])
                            writer.writerow(row)

                    logger.info(f"Saved CSV results to: {csv_file}")
                except Exception as e:
                    logger.warning(f"Failed to save CSV results: {e}")

            # Generate performance plots if requested
            if self.output_config.save_plots:
                try:
                    # Optimization history plot
                    try:
                        fig = optuna.visualization.plot_optimization_history(study)
                        plot_file = output_path / "optimization_history.html"
                        fig.write_html(str(plot_file))
                        logger.info(f"Saved optimization history plot to: {plot_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate optimization history plot: {e}"
                        )

                    # Parallel coordinate plot
                    try:
                        fig = optuna.visualization.plot_parallel_coordinate(study)
                        plot_file = output_path / "parallel_coordinate.html"
                        fig.write_html(str(plot_file))
                        logger.info(f"Saved parallel coordinate plot to: {plot_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate parallel coordinate plot: {e}"
                        )

                    # Parameter importance plot (may fail if not enough trials)
                    try:
                        if len(study.trials) > 1:
                            fig = optuna.visualization.plot_param_importances(study)
                            plot_file = output_path / "param_importances.html"
                            fig.write_html(str(plot_file))
                            logger.info(
                                f"Saved parameter importance plot to: {plot_file}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate parameter importance plot (may need more trials): {e}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate plots: {e}")

            logger.info(f"Benchmark results saved successfully to: {output_path}")

        except Exception as e:
            logger.warning(f"Failed to save benchmark results: {e}")


class DataLoaderBenchmark:
    """Comprehensive benchmarking class for DataLoader performance profiling."""

    def __init__(
        self,
        image_paths: List[str],
        config: LoaderConfig,
        max_batches: Optional[int] = None,
        memory_check_interval: int = 10,
        use_tile_dataset: bool = True,
    ):
        """
        Initialize DataLoader benchmark.

        Args:
            image_paths: List of image paths to benchmark
            config: LoaderConfig with tile_size, overlap, batch_size, etc.
            max_batches: Maximum number of batches to process (default: None)
            memory_check_interval: Check memory every N batches (default: 10)
            use_tile_dataset: Whether to use TileDataset (default: True)
        """
        self.image_paths = image_paths
        self.config = config
        self.max_batches = max_batches or np.inf
        self.memory_check_interval = memory_check_interval
        self.use_tile_dataset = use_tile_dataset
        self.metrics: Dict[str, float] = {}
        self.memory_readings: List[float] = []

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return sys.getsizeof(gc.get_objects()) / 1024 / 1024

    def run(self, verbose: bool = True, profile: bool = False) -> Dict[str, float]:
        """
        Run comprehensive benchmark and return performance metrics.

        Args:
            verbose: Whether to print detailed progress and report (default: True)

        Returns:
            Dictionary with performance metrics including timing, throughput, and memory
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("DATA LOADER BENCHMARK")
            logger.info("=" * 60)

        # Initialize metrics
        self.metrics = {
            "image_discovery_time": 0.0,
            "dataset_creation_time": 0.0,
            "loader_creation_time": 0.0,
            "iteration_time": 0.0,
            "total_tiles": 0,
            "total_batches": 0,
            "memory_peak": 0.0,
            "memory_start": 0.0,
            "memory_end": 0.0,
            "throughput_tiles_per_sec": 0.0,
            "throughput_batches_per_sec": 0.0,
            "avg_batch_size": 0.0,
            "tile_size": self.config.tile_size,
            "overlap": self.config.overlap,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "num_images": len(self.image_paths),
        }

        # Start profiling
        start_time = time.perf_counter()
        initial_memory = self._get_memory_usage()
        self.metrics["memory_start"] = initial_memory

        if verbose:
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            logger.info("Starting performance profiling...")

        # Phase 1: DataLoader creation
        if verbose:
            logger.info("Phase 1: DataLoader creation")
        phase_start = time.perf_counter()
        loader = DataLoader(
            image_paths=self.image_paths,
            image_dir=None,
            config=self.config,
            use_tile_dataset=self.use_tile_dataset,
        )
        self.metrics["loader_creation_time"] = time.perf_counter() - phase_start
        if verbose:
            logger.info(
                f"Created loader with {len(loader)} batches in {self.metrics['loader_creation_time']:.3f}s"
            )

        # Phase 2: Iteration and data loading
        if verbose:
            logger.info("Phase 2: Data iteration and loading")
        phase_start = time.perf_counter()
        batch_count = 0
        total_tiles_processed = 0
        batch_sizes = []

        # Reset memory readings and batch timings
        self.memory_readings = []
        batch_load_times = []  # Track per-batch loading time

        # Create iterator to manually time batch loading
        iterator = iter(tqdm(loader, desc="Processing batches") if verbose else loader)

        with profile_command(
            output_dir=None,
            profile=profile,
            memory_profile=False,
            line_profile=False,
            gpu_profile=False,
        ):
            while batch_count < self.max_batches:
                # Time the actual batch loading (iterator's __next__ call)
                batch_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                batch_time = time.perf_counter() - batch_start
                batch_load_times.append(batch_time)

                # Process batch
                batch_count += 1
                if self.use_tile_dataset:
                    batch_size = len(batch["tiles"])
                else:
                    batch_size = batch.shape[0]
                # print(f"size: {batch.shape}")
                total_tiles_processed += batch_size
                batch_sizes.append(batch_size)

                # Memory tracking at intervals
                if batch_count % self.memory_check_interval == 0:
                    self.memory_readings.append(self._get_memory_usage())

        self.metrics["iteration_time"] = time.perf_counter() - phase_start
        self.metrics["total_batches"] = batch_count
        self.metrics["total_tiles"] = total_tiles_processed

        # Calculate throughput
        if self.metrics["iteration_time"] > 0:
            self.metrics["throughput_tiles_per_sec"] = (
                total_tiles_processed / self.metrics["iteration_time"]
            )
            self.metrics["throughput_batches_per_sec"] = (
                batch_count / self.metrics["iteration_time"]
            )

        # Calculate average batch size
        if batch_sizes:
            self.metrics["avg_batch_size"] = sum(batch_sizes) / len(batch_sizes)

        # Analyze batch loading times (cache performance diagnostics)
        if batch_load_times:
            batch_times_array = np.array(batch_load_times)
            self.metrics["batch_load_time_mean"] = float(np.mean(batch_times_array))
            self.metrics["batch_load_time_std"] = float(np.std(batch_times_array))
            self.metrics["batch_load_time_min"] = float(np.min(batch_times_array))
            self.metrics["batch_load_time_max"] = float(np.max(batch_times_array))
            self.metrics["batch_load_time_median"] = float(np.median(batch_times_array))

            # Store batch times for detailed analysis
            self.metrics["batch_load_times"] = batch_load_times

            # Identify potential cache hits vs misses
            # Fast batches (likely cache hits) vs slow batches (likely cache misses)
            threshold = (
                self.metrics["batch_load_time_mean"]
                + self.metrics["batch_load_time_std"]
            )
            slow_batches = [i for i, t in enumerate(batch_load_times) if t > threshold]
            fast_batches = [
                i
                for i, t in enumerate(batch_load_times)
                if t < self.metrics["batch_load_time_mean"]
            ]

            self.metrics["num_slow_batches"] = len(slow_batches)
            self.metrics["num_fast_batches"] = len(fast_batches)
            self.metrics["slow_batch_indices"] = slow_batches[
                :10
            ]  # First 10 slow batches
            self.metrics["cache_hit_rate_estimate"] = (
                len(fast_batches) / len(batch_load_times) if batch_load_times else 0
            )

        # Memory analysis
        final_memory = self._get_memory_usage()
        self.metrics["memory_end"] = final_memory
        self.metrics["memory_peak"] = (
            max(self.memory_readings) if self.memory_readings else final_memory
        )

        # Force garbage collection and measure final memory
        gc.collect()
        final_memory_after_gc = self._get_memory_usage()
        self.metrics["memory_after_gc"] = final_memory_after_gc

        # Total time
        self.metrics["total_time"] = time.perf_counter() - start_time

        # Print report if verbose
        if verbose:
            self._print_report()

        return self.metrics

    def _print_report(self):
        """Print comprehensive performance report."""
        logger.info("=" * 60)
        logger.info("PERFORMANCE PROFILE REPORT")
        logger.info("=" * 60)

        logger.info("TIMING BREAKDOWN:")
        logger.info(
            f"  Loader creation:     {self.metrics['loader_creation_time']:.3f}s"
        )
        logger.info(f"  Data iteration:      {self.metrics['iteration_time']:.3f}s")
        logger.info(f"  Total time:          {self.metrics['total_time']:.3f}s")

        logger.info("\nTHROUGHPUT METRICS:")
        logger.info(f"  Tiles processed:     {self.metrics['total_tiles']}")
        logger.info(f"  Batches processed:   {self.metrics['total_batches']}")
        logger.info(
            f"  Tiles per second:    {self.metrics['throughput_tiles_per_sec']:.2f}"
        )
        logger.info(
            f"  Batches per second:  {self.metrics['throughput_batches_per_sec']:.2f}"
        )
        logger.info(f"  Average batch size:  {self.metrics['avg_batch_size']:.1f}")

        logger.info("\nMEMORY USAGE:")
        logger.info(f"  Initial memory:      {self.metrics['memory_start']:.2f} MB")
        logger.info(f"  Peak memory:         {self.metrics['memory_peak']:.2f} MB")
        logger.info(f"  Final memory:        {self.metrics['memory_end']:.2f} MB")
        logger.info(f"  Memory after GC:     {self.metrics['memory_after_gc']:.2f} MB")
        logger.info(
            f"  Memory increase:     {self.metrics['memory_end'] - self.metrics['memory_start']:.2f} MB"
        )

        logger.info("\nCONFIGURATION:")
        logger.info(f"  Tile size:           {self.metrics['tile_size']}")
        logger.info(f"  Overlap:             {self.metrics['overlap']}")
        logger.info(f"  Batch size:          {self.metrics['batch_size']}")
        logger.info(f"  Num workers:         {self.metrics['num_workers']}")
        logger.info(f"  Images processed:    {self.metrics['num_images']}")

        # Cache performance diagnostics
        if "batch_load_time_mean" in self.metrics:
            logger.info("\nCACHE PERFORMANCE DIAGNOSTICS:")
            logger.info(
                f"  Mean batch time:     {self.metrics['batch_load_time_mean']*1000:.2f} ms"
            )
            logger.info(
                f"  Std batch time:      {self.metrics['batch_load_time_std']*1000:.2f} ms"
            )
            logger.info(
                f"  Min batch time:      {self.metrics['batch_load_time_min']*1000:.2f} ms (fastest)"
            )
            logger.info(
                f"  Max batch time:      {self.metrics['batch_load_time_max']*1000:.2f} ms (slowest)"
            )
            logger.info(
                f"  Median batch time:   {self.metrics['batch_load_time_median']*1000:.2f} ms"
            )

            if self.metrics["slow_batch_indices"]:
                logger.info(
                    f"  First slow batches:  {self.metrics['slow_batch_indices']}"
                )

        # Performance recommendations
        logger.info("\nPERFORMANCE RECOMMENDATIONS:")
        recommendations = self._get_recommendations()
        if recommendations:
            for rec in recommendations:
                logger.info(f"  ⚠️  {rec}")
        else:
            logger.info("  ✓ Performance looks good!")

        logger.info("=" * 60)

    def print_batch_timing_details(self, num_batches: int = 20):
        """
        Print detailed timing for individual batches.

        Args:
            num_batches: Number of batches to display (default: 20)
        """
        if "batch_load_times" not in self.metrics:
            logger.warning("No batch timing data available. Run benchmark first.")
            return

        batch_times = self.metrics["batch_load_times"]
        num_to_show = min(num_batches, len(batch_times))

        logger.info("=" * 80)
        logger.info(f"DETAILED BATCH TIMING (first {num_to_show} batches)")
        logger.info("=" * 80)

        mean_time = self.metrics["batch_load_time_mean"]
        std_time = self.metrics["batch_load_time_std"]

        logger.info(f"{'Batch':<8} {'Time (ms)':<12} {'Status':<20} {'Deviation':<15}")
        logger.info("-" * 80)

        for i in range(num_to_show):
            batch_time = batch_times[i] * 1000  # Convert to ms
            deviation = (batch_times[i] - mean_time) / std_time if std_time > 0 else 0

            # Classify batch
            if batch_times[i] > mean_time + std_time:
                status = "SLOW"
            elif batch_times[i] < mean_time - std_time:
                status = "FAST"
            else:
                status = "Normal"

            logger.info(f"{i:<8} {batch_time:<12.2f} {status:<20} {deviation:>+.2f}σ")

        if len(batch_times) > num_to_show:
            logger.info(f"\n... and {len(batch_times) - num_to_show} more batches")

        logger.info("=" * 80)

    def _get_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []

        if self.metrics["throughput_tiles_per_sec"] < 10:
            recommendations.append(
                "Low throughput - consider optimizing image loading or increasing batch size"
            )

        if self.metrics["memory_peak"] > 1000:
            recommendations.append(
                "High memory usage - consider reducing batch size or tile size"
            )

        if self.metrics["num_workers"] == 0 and self.metrics["num_images"] > 10:
            recommendations.append(
                "Using 0 workers - consider increasing num_workers for parallel data loading"
            )

        memory_increase = self.metrics["memory_end"] - self.metrics["memory_start"]
        if memory_increase > 500:
            recommendations.append(
                f"Large memory increase ({memory_increase:.1f} MB) - possible memory leak or cache growth"
            )

        return recommendations

    def compare_configs(
        self, configs: List[LoaderConfig], config_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple configurations and compare results.

        Args:
            configs: List of LoaderConfig objects to benchmark
            config_names: Optional names for each config

        Returns:
            Dictionary mapping config names to their metrics
        """
        if config_names is None:
            config_names = [f"Config_{i+1}" for i in range(len(configs))]

        results = {}

        logger.info("=" * 60)
        logger.info("COMPARING MULTIPLE CONFIGURATIONS")
        logger.info("=" * 60)

        for config, name in zip(configs, config_names):
            logger.info(f"\nBenchmarking: {name}")
            logger.info(
                f"  Tile size: {config.tile_size}, Overlap: {config.overlap}, "
                f"Batch size: {config.batch_size}, Workers: {config.num_workers}"
            )

            # Update config and run benchmark
            self.config = config
            self.metrics["tile_size"] = config.tile_size
            self.metrics["overlap"] = config.overlap
            self.metrics["batch_size"] = config.batch_size
            self.metrics["num_workers"] = config.num_workers

            metrics = self.run(verbose=False)
            results[name] = metrics

            logger.info(
                f"  Throughput: {metrics['throughput_tiles_per_sec']:.2f} tiles/sec"
            )
            logger.info(f"  Total time: {metrics['total_time']:.2f}s")
            logger.info(f"  Peak memory: {metrics['memory_peak']:.2f} MB")

        # Print comparison summary
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)

        # Find best config for different metrics
        best_throughput = max(
            results.items(), key=lambda x: x[1]["throughput_tiles_per_sec"]
        )
        best_memory = min(results.items(), key=lambda x: x[1]["memory_peak"])
        fastest = min(results.items(), key=lambda x: x[1]["total_time"])

        logger.info(
            f"Best throughput:  {best_throughput[0]} ({best_throughput[1]['throughput_tiles_per_sec']:.2f} tiles/sec)"
        )
        logger.info(
            f"Lowest memory:    {best_memory[0]} ({best_memory[1]['memory_peak']:.2f} MB)"
        )
        logger.info(f"Fastest overall:  {fastest[0]} ({fastest[1]['total_time']:.2f}s)")
        logger.info("=" * 60)

        return results
