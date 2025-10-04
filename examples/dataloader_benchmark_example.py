"""
Example script demonstrating how to use the DataLoaderBenchmark class.

This script shows:
1. Basic single-config benchmarking
2. Comparing multiple configurations
3. Finding optimal settings for your use case
"""
import fire
import logging
from pathlib import Path

from wildetect.core.config import LoaderConfig
from wildetect.core.data.utils import get_images_paths
from wildetect.utils.benchmark import DataLoaderBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_single_benchmark(n: int = 10,use_tile_dataset: bool = True, num_workers: int = 0, batch_size: int = 8, profile: bool = True):
    """Example of benchmarking a single configuration."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Single Configuration Benchmark")
    logger.info("=" * 80)

    # Get test images (adjust path to your images)
    image_dir = r"D:\workspace\data\savmap_dataset_v2\raw\images"
    image_paths = get_images_paths(image_dir)[:n]  # Use first 10 images
    
    logger.info(f"Using {len(image_paths)} images for benchmarking")

    # Create a loader configuration
    config = LoaderConfig(
        tile_size=800,
        overlap=0.2,
        batch_size=batch_size,
        num_workers=num_workers 
    )

    # Create benchmark instance
    benchmark = DataLoaderBenchmark(
        image_paths=image_paths,
        config=config,
        max_batches=None,  
        memory_check_interval=10,  
        use_tile_dataset=use_tile_dataset
    )

    # Run benchmark
    metrics = benchmark.run(verbose=True,profile=profile)

    # Access specific metrics
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Throughput: {metrics['throughput_tiles_per_sec']:.2f} tiles/sec")
    logger.info(f"  Peak Memory: {metrics['memory_peak']:.2f} MB")
    logger.info(f"  Total Time: {metrics['total_time']:.2f}s")
    
    # Print detailed batch timing to diagnose cache hits/misses
    logger.info("\n")
    benchmark.print_batch_timing_details(num_batches=30)


def example_compare_configs():
    """Example of comparing multiple configurations."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Comparing Multiple Configurations")
    logger.info("=" * 80)

    # Get test images
    image_dir = r"D:\workspace\data\savmap_dataset_v2\raw\images"
    image_paths = get_images_paths(image_dir)[:10]

    # Define multiple configurations to compare
    configs = [
        LoaderConfig(tile_size=640, overlap=0.2, batch_size=16, num_workers=0),
        LoaderConfig(tile_size=640, overlap=0.2, batch_size=32, num_workers=0),
        LoaderConfig(tile_size=640, overlap=0.2, batch_size=64, num_workers=0),
        LoaderConfig(tile_size=800, overlap=0.2, batch_size=32, num_workers=0),
        LoaderConfig(tile_size=1024, overlap=0.2, batch_size=32, num_workers=0),
    ]

    config_names = [
        "Small Batch (16)",
        "Medium Batch (32)",
        "Large Batch (64)",
        "Large Tiles (800)",
        "Extra Large Tiles (1024)",
    ]

    # Create benchmark instance
    benchmark = DataLoaderBenchmark(
        image_paths=image_paths,
        config=configs[0],  # Initial config (will be updated)
        max_batches=30  # Shorter for comparison
    )

    # Compare configurations
    results = benchmark.compare_configs(configs, config_names)

    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED RESULTS ANALYSIS")
    logger.info("=" * 80)

    for name, metrics in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Throughput:    {metrics['throughput_tiles_per_sec']:.2f} tiles/sec")
        logger.info(f"  Total time:    {metrics['total_time']:.2f}s")
        logger.info(f"  Peak memory:   {metrics['memory_peak']:.2f} MB")
        logger.info(f"  Tiles created: {metrics['total_tiles']}")


def example_find_optimal_batch_size():
    """Example of finding optimal batch size for your system."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Finding Optimal Batch Size")
    logger.info("=" * 80)

    # Get test images
    image_dir = r"D:\workspace\data\savmap_dataset_v2\raw\images"
    image_paths = get_images_paths(image_dir)[:10]

    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]
    
    configs = [
        LoaderConfig(tile_size=800, overlap=0.2, batch_size=bs, num_workers=0)
        for bs in batch_sizes
    ]
    
    config_names = [f"Batch Size {bs}" for bs in batch_sizes]

    # Run comparison
    benchmark = DataLoaderBenchmark(
        image_paths=image_paths,
        config=configs[0],
        max_batches=None
    )

    results = benchmark.compare_configs(configs, config_names)

    # Find optimal based on throughput/memory tradeoff
    logger.info("\n" + "=" * 80)
    logger.info("BATCH SIZE OPTIMIZATION RESULTS")
    logger.info("=" * 80)

    for name, metrics in results.items():
        efficiency = metrics['throughput_tiles_per_sec'] / max(metrics['memory_peak'], 1)
        logger.info(
            f"{name}: {metrics['throughput_tiles_per_sec']:.2f} tiles/sec, "
            f"{metrics['memory_peak']:.2f} MB, "
            f"Efficiency: {efficiency:.4f}"
        )


def example_cache_diagnostics():
    """Example of diagnosing cache performance with different cache sizes."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Cache Performance Diagnostics")
    logger.info("=" * 80)
    
    # Get test images - use more images to stress the cache
    image_dir = r"D:\workspace\data\savmap_dataset_v2\raw\images"
    image_paths = get_images_paths(image_dir)[:20]  # Use 20 images
    
    logger.info(f"Testing cache with {len(image_paths)} images")
    logger.info("Note: SimpleMemoryLRUCache default max_items is 5")
    logger.info("We expect to see cache misses when loading more unique images than cache size\n")
    
    # Single config to test cache behavior
    config = LoaderConfig(
        tile_size=800,
        overlap=0.2,
        batch_size=16,
        num_workers=0
    )
    
    # Run benchmark
    benchmark = DataLoaderBenchmark(
        image_paths=image_paths,
        config=config,
        max_batches=100  # Run more batches to see cache effects
    )
    
    metrics = benchmark.run(verbose=True)
    
    # Show detailed batch timings
    logger.info("\n")
    benchmark.print_batch_timing_details(num_batches=50)
    
    # Analyze cache effectiveness
    logger.info("\n" + "=" * 80)
    logger.info("CACHE EFFECTIVENESS ANALYSIS")
    logger.info("=" * 80)
    
    cache_hit_rate = metrics.get('cache_hit_rate_estimate', 0) * 100
    logger.info(f"Estimated cache hit rate: {cache_hit_rate:.1f}%")
    
    if cache_hit_rate < 50:
        logger.info("⚠️  Low cache hit rate! Consider:")
        logger.info("   - Increasing SimpleMemoryLRUCache max_items (currently 5)")
        logger.info("   - Reducing number of unique images processed")
        logger.info("   - Sorting images to improve locality")
    elif cache_hit_rate > 80:
        logger.info("✓ Good cache hit rate! Cache is working effectively.")
    else:
        logger.info("ℹ️  Moderate cache hit rate. Some room for improvement.")


if __name__ == "__main__":
    fire.Fire()

