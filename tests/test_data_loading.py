"""
Test script for data loading functionality.
"""

import gc
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from wildetect.core.config import FlightSpecs, LoaderConfig

# Add the project root to the Python path
from wildetect.core.data import DataLoader, TileDataset, load_images_as_drone_images
from wildetect.core.data.utils import get_images_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

IMAGES_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"


def create_test_images(num_images: int = -1) -> List[str]:
    """Create test images for testing.

    Args:
        test_dir (Path): Directory to create test images in
        num_images (int): Number of test images to create

    Returns:
        List[str]: List of created image paths
    """
    image_paths = get_images_paths(IMAGES_DIR)
    return image_paths[: min(num_images, len(image_paths))]


def test_loader_config():
    """Test LoaderConfig functionality."""
    logger.info("=" * 60)
    logger.info("TESTING LOADER CONFIG")
    logger.info("=" * 60)

    # Test basic config creation
    config = LoaderConfig(tile_size=640, overlap=0.2, batch_size=4, recursive=True)

    logger.info(
        f"Config created: tile_size={config.tile_size}, overlap={config.overlap}"
    )
    logger.info(f"Batch size: {config.batch_size}, recursive: {config.recursive}")

    # Test with flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    config_with_flight = LoaderConfig(
        tile_size=640, overlap=0.2, flight_specs=flight_specs
    )

    assert config_with_flight.flight_specs is not None
    logger.info(f"Flight specs: {config_with_flight.flight_specs}")
    logger.info("✓ LoaderConfig test passed")


def test_image_tile_dataset():
    """Test ImageTileDataset functionality."""
    logger.info("=" * 60)
    logger.info("TESTING IMAGE TILE DATASET")
    logger.info("=" * 60)

    # Create test images
    logger.info("Creating test images...")
    image_paths = create_test_images(num_images=3)
    logger.info(f"Created {len(image_paths)} test images")

    # Create config
    logger.info("Creating config...")
    config = LoaderConfig(
        tile_size=320,  # Smaller tiles for testing
        overlap=0.2,
        batch_size=2,
    )
    logger.info("Config created successfully")

    # Create dataset
    logger.info("Creating dataset...")
    try:
        dataset = TileDataset(config=config, image_paths=image_paths)
        logger.info("Dataset created successfully")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        import traceback

        traceback.print_exc()
        raise

    logger.info(f"Dataset created with {len(dataset.tiles)} tiles")
    logger.info(f"Number of images: {len(dataset.image_paths)}")

    # Test dataset iteration
    logger.info("Testing dataset iteration...")
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            logger.info(
                f"Item {i}: tile_id={item['tile_id']}, shape={item['image'].shape if hasattr(item['image'], 'shape') else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Failed to get item {i}: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Test dataset length
    assert len(dataset) > 0, "Dataset should not be empty"
    logger.info(f"Dataset length: {len(dataset)}")
    logger.info("✓ TileDataset test passed")


def test_data_loader():
    """Test DataLoader functionality."""
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADER")
    logger.info("=" * 60)

    # Create test images
    image_paths = create_test_images(num_images=4)

    # Create config
    config = LoaderConfig(
        tile_size=320,
        overlap=0.2,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
    )

    # Create data loader
    dataloader = DataLoader(image_paths=image_paths, image_dir=None, config=config)

    logger.info(f"DataLoader created with {len(dataloader)} batches")

    # Test batch iteration
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        logger.info(f"Batch {batch_count}: {len(batch['tiles'])} tiles")

        if batch_count >= 2:  # Test first 2 batches
            break

    assert batch_count > 0, "Should have at least one batch"
    logger.info("✓ DataLoader test passed")


def test_load_images_as_drone_images():
    """Test load_images_as_drone_images function."""
    logger.info("=" * 60)
    logger.info("TESTING LOAD_IMAGES_AS_DRONE_IMAGES")
    logger.info("=" * 60)

    # Create test images
    image_paths = create_test_images(num_images=3)

    # Create config
    config = LoaderConfig(
        tile_size=640,
        overlap=0.2,
        flight_specs=FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        ),
    )

    # Load images as drone images
    drone_images = load_images_as_drone_images(
        image_paths=image_paths, flight_specs=config.flight_specs
    )

    logger.info(f"Loaded {len(drone_images)} drone images")

    # Test drone image properties
    for i, drone_image in enumerate(drone_images):
        logger.info(
            f"DroneImage {i}: id={drone_image.id}, tiles={len(drone_image.tiles)}"
        )
        assert drone_image.id is not None, "DroneImage should have an ID"
        assert len(drone_image.tiles) > 0, "DroneImage should have tiles"

    logger.info("✓ load_images_as_drone_images test passed")


def test_create_loader():
    """Test create_loader function with comprehensive profiling."""
    logger.info("=" * 60)
    logger.info("TESTING CREATE_LOADER WITH PROFILING")
    logger.info("=" * 60)

    # Memory tracking function
    def get_memory_usage():
        """Get current memory usage in MB."""
        return sys.getsizeof(gc.get_objects()) / 1024 / 1024  # Convert to MB

    # Performance tracking
    performance_metrics: Dict[str, float] = {
        "image_discovery_time": 0.0,
        "dataset_creation_time": 0.0,
        "loader_creation_time": 0.0,
        "iteration_time": 0.0,
        "total_tiles": 0.0,
        "total_batches": 0.0,
        "memory_peak": 0.0,
        "memory_start": 0.0,
        "memory_end": 0.0,
        "throughput_tiles_per_sec": 0.0,
        "throughput_batches_per_sec": 0.0,
        "avg_batch_size": 0.0,
        "tile_size": 0.0,
        "overlap": 0.0,
        "batch_size": 0.0,
    }

    # Start profiling
    start_time = time.time()
    initial_memory = get_memory_usage()
    performance_metrics["memory_start"] = initial_memory

    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    logger.info("Starting performance profiling...")

    # Phase 1: Image discovery
    logger.info("Phase 1: Image discovery")
    phase_start = time.time()
    image_paths = create_test_images(num_images=-1)
    performance_metrics["image_discovery_time"] = time.time() - phase_start
    logger.info(
        f"Found {len(image_paths)} images in {performance_metrics['image_discovery_time']:.3f}s"
    )

    # Phase 2: Configuration
    logger.info("Phase 2: Configuration setup")
    config = LoaderConfig(tile_size=800, overlap=0.2, batch_size=32)
    performance_metrics["tile_size"] = config.tile_size
    performance_metrics["overlap"] = config.overlap
    performance_metrics["batch_size"] = config.batch_size

    # Phase 3: Dataset creation
    logger.info("Phase 3: Dataset creation")
    phase_start = time.time()
    dataset = TileDataset(config=config, image_paths=image_paths)
    performance_metrics["dataset_creation_time"] = time.time() - phase_start
    performance_metrics["total_tiles"] = len(dataset)
    logger.info(
        f"Created dataset with {len(dataset)} tiles in {performance_metrics['dataset_creation_time']:.3f}s"
    )

    # Phase 4: DataLoader creation
    logger.info("Phase 4: DataLoader creation")
    phase_start = time.time()
    loader = DataLoader(image_paths=image_paths, image_dir=None, config=config)
    performance_metrics["loader_creation_time"] = time.time() - phase_start
    performance_metrics["total_batches"] = len(loader)
    logger.info(
        f"Created loader with {len(loader)} batches in {performance_metrics['loader_creation_time']:.3f}s"
    )

    # Phase 5: Iteration and data loading
    logger.info("Phase 5: Data iteration and loading")
    phase_start = time.time()
    batch_count = 0
    total_tiles_processed = 0
    batch_sizes = []

    # Track memory during iteration
    memory_readings = []

    for batch in tqdm(loader, desc="Processing batches"):
        batch_count += 1
        batch_size = len(batch["tiles"])
        total_tiles_processed += batch_size
        batch_sizes.append(batch_size)

        # Memory tracking every 10 batches
        if batch_count % 10 == 0:
            memory_readings.append(get_memory_usage())

        # Log batch info
        # logger.info(f"Batch {batch_count}: {batch_size} tiles, images shape: {batch['images'].shape}")

        # Limit to first 50 batches for profiling
        if batch_count >= 50:
            break

    performance_metrics["iteration_time"] = time.time() - phase_start
    performance_metrics["total_batches"] = batch_count
    performance_metrics["total_tiles"] = total_tiles_processed

    # Calculate throughput
    total_time = performance_metrics["iteration_time"]
    if total_time > 0:
        performance_metrics["throughput_tiles_per_sec"] = (
            total_tiles_processed / total_time
        )
        performance_metrics["throughput_batches_per_sec"] = batch_count / total_time

    # Calculate average batch size
    if batch_sizes:
        performance_metrics["avg_batch_size"] = sum(batch_sizes) / len(batch_sizes)

    # Memory analysis
    final_memory = get_memory_usage()
    performance_metrics["memory_end"] = final_memory
    performance_metrics["memory_peak"] = (
        max(memory_readings) if memory_readings else final_memory
    )

    # Force garbage collection and measure final memory
    gc.collect()
    final_memory_after_gc = get_memory_usage()

    # Total time
    total_time = time.time() - start_time

    # Print comprehensive performance report
    logger.info("=" * 60)
    logger.info("PERFORMANCE PROFILE REPORT")
    logger.info("=" * 60)

    logger.info("TIMING BREAKDOWN:")
    logger.info(
        f"  Image discovery:     {performance_metrics['image_discovery_time']:.3f}s"
    )
    logger.info(
        f"  Dataset creation:    {performance_metrics['dataset_creation_time']:.3f}s"
    )
    logger.info(
        f"  Loader creation:     {performance_metrics['loader_creation_time']:.3f}s"
    )
    logger.info(f"  Data iteration:      {performance_metrics['iteration_time']:.3f}s")
    logger.info(f"  Total time:          {total_time:.3f}s")

    logger.info("\nTHROUGHPUT METRICS:")
    logger.info(f"  Tiles processed:     {performance_metrics['total_tiles']}")
    logger.info(f"  Batches processed:   {performance_metrics['total_batches']}")
    logger.info(
        f"  Tiles per second:    {performance_metrics['throughput_tiles_per_sec']:.2f}"
    )
    logger.info(
        f"  Batches per second:  {performance_metrics['throughput_batches_per_sec']:.2f}"
    )
    logger.info(f"  Average batch size:  {performance_metrics['avg_batch_size']:.1f}")

    logger.info("\nMEMORY USAGE:")
    logger.info(f"  Initial memory:      {performance_metrics['memory_start']:.2f} MB")
    logger.info(f"  Peak memory:         {performance_metrics['memory_peak']:.2f} MB")
    logger.info(f"  Final memory:        {performance_metrics['memory_end']:.2f} MB")
    logger.info(f"  Memory after GC:     {final_memory_after_gc:.2f} MB")
    logger.info(
        f"  Memory increase:     {performance_metrics['memory_end'] - performance_metrics['memory_start']:.2f} MB"
    )

    logger.info("\nCONFIGURATION:")
    logger.info(f"  Tile size:           {performance_metrics['tile_size']}")
    logger.info(f"  Overlap:             {performance_metrics['overlap']}")
    logger.info(f"  Batch size:          {performance_metrics['batch_size']}")
    logger.info(f"  Images processed:    {len(image_paths)}")

    # Performance recommendations
    logger.info("\nPERFORMANCE RECOMMENDATIONS:")
    if performance_metrics["throughput_tiles_per_sec"] < 10:
        logger.info("  ⚠️  Low throughput - consider optimizing image loading")
    if performance_metrics["memory_peak"] > 1000:
        logger.info(
            "  ⚠️  High memory usage - consider reducing batch size or tile size"
        )
    if (
        performance_metrics["dataset_creation_time"]
        > performance_metrics["iteration_time"] * 0.5
    ):
        logger.info(
            "  ⚠️  Dataset creation is slow - consider caching tile calculations"
        )

    logger.info("=" * 60)
    logger.info("✓ create_loader test with profiling completed")
    logger.info("=" * 60)

    # Assertions for test validation
    assert batch_count > 0, "Should have at least one batch"
    assert total_tiles_processed > 0, "Should have processed at least one tile"
    assert (
        performance_metrics["throughput_tiles_per_sec"] > 0
    ), "Should have positive throughput"


def run_all_tests():
    """Run all data loading tests."""
    logger.info("Starting data loading tests...")

    try:
        test_loader_config()
        test_image_tile_dataset()
        test_data_loader()
        test_load_images_as_drone_images()
        test_create_loader()

        logger.info("=" * 60)
        logger.info("ALL DATA LOADING TESTS PASSED! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
