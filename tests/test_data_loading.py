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
    print("=" * 60)
    print("TESTING LOADER CONFIG")
    print("=" * 60)

    # Test basic config creation
    config = LoaderConfig(tile_size=640, overlap=0.2, batch_size=4, recursive=True)

    print(f"Config created: tile_size={config.tile_size}, overlap={config.overlap}")
    print(f"Batch size: {config.batch_size}, recursive: {config.recursive}")

    # Test with flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    config_with_flight = LoaderConfig(
        tile_size=640, overlap=0.2, flight_specs=flight_specs
    )

    assert config_with_flight.flight_specs is not None
    print(f"Flight specs: {config_with_flight.flight_specs}")
    print("✓ LoaderConfig test passed")


def test_image_tile_dataset():
    """Test ImageTileDataset functionality."""
    print("=" * 60)
    print("TESTING IMAGE TILE DATASET")
    print("=" * 60)

    # Create test images
    print("Creating test images...")
    image_paths = create_test_images(num_images=3)
    print(f"Created {len(image_paths)} test images")

    # Create config
    print("Creating config...")
    config = LoaderConfig(
        tile_size=320,  # Smaller tiles for testing
        overlap=0.2,
        batch_size=2,
    )
    print("Config created successfully")

    # Create dataset
    print("Creating dataset...")
    try:
        dataset = TileDataset(config=config, image_paths=image_paths)
        print("Dataset created successfully")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        import traceback

        traceback.print_exc()
        raise

    print(f"Dataset created with {len(dataset.tiles)} tiles")
    print(f"Number of images: {len(dataset.image_paths)}")

    # Test dataset iteration
    print("Testing dataset iteration...")
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            print(
                f"Item {i}: tile_id={item['tile_id']}, shape={item['image'].shape if hasattr(item['image'], 'shape') else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Failed to get item {i}: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Test dataset length
    assert len(dataset) > 0, "Dataset should not be empty"
    print(f"Dataset length: {len(dataset)}")
    print("✓ TileDataset test passed")


def test_data_loader():
    """Test DataLoader functionality."""
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)

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

    print(f"DataLoader created with {len(dataloader)} batches")

    # Test batch iteration
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        print(f"Batch {batch_count}: {len(batch['tiles'])} tiles")

        if batch_count >= 2:  # Test first 2 batches
            break

    assert batch_count > 0, "Should have at least one batch"
    print("✓ DataLoader test passed")


def test_load_images_as_drone_images():
    """Test load_images_as_drone_images function."""
    print("=" * 60)
    print("TESTING LOAD_IMAGES_AS_DRONE_IMAGES")
    print("=" * 60)

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

    print(f"Loaded {len(drone_images)} drone images")

    # Test drone image properties
    for i, drone_image in enumerate(drone_images):
        print(f"DroneImage {i}: id={drone_image.id}, tiles={len(drone_image.tiles)}")
        assert drone_image.id is not None, "DroneImage should have an ID"
        assert len(drone_image.tiles) > 0, "DroneImage should have tiles"

    print("✓ load_images_as_drone_images test passed")


def test_create_loader():
    """Test create_loader function with comprehensive profiling."""
    print("=" * 60)
    print("TESTING CREATE_LOADER WITH PROFILING")
    print("=" * 60)

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

    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print("Starting performance profiling...")

    # Phase 1: Image discovery
    print("Phase 1: Image discovery")
    phase_start = time.time()
    image_paths = create_test_images(num_images=-1)
    performance_metrics["image_discovery_time"] = time.time() - phase_start
    print(
        f"Found {len(image_paths)} images in {performance_metrics['image_discovery_time']:.3f}s"
    )

    # Phase 2: Configuration
    print("Phase 2: Configuration setup")
    config = LoaderConfig(tile_size=800, overlap=0.2, batch_size=32, num_workers=4)
    performance_metrics["tile_size"] = config.tile_size
    performance_metrics["overlap"] = config.overlap
    performance_metrics["batch_size"] = config.batch_size

    # Phase 3: DataLoader creation
    print("Phase 3: DataLoader creation")
    phase_start = time.time()
    loader = DataLoader(image_paths=image_paths, image_dir=None, config=config)
    performance_metrics["loader_creation_time"] = time.time() - phase_start
    performance_metrics["total_batches"] = len(loader)
    print(
        f"Created loader with {len(loader)} batches in {performance_metrics['loader_creation_time']:.3f}s"
    )

    # Phase 4: Iteration and data loading
    print("Phase 4: Data iteration and loading")
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

        # Limit to first 50 batches for profiling
        # if batch_count >= 50:
        #    break

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
    print("=" * 60)
    print("PERFORMANCE PROFILE REPORT")
    print("=" * 60)

    print("TIMING BREAKDOWN:")
    print(f"  Image discovery:     {performance_metrics['image_discovery_time']:.3f}s")
    print(f"  Dataset creation:    {performance_metrics['dataset_creation_time']:.3f}s")
    print(f"  Loader creation:     {performance_metrics['loader_creation_time']:.3f}s")
    print(f"  Data iteration:      {performance_metrics['iteration_time']:.3f}s")
    print(f"  Total time:          {total_time:.3f}s")

    print("\nTHROUGHPUT METRICS:")
    print(f"  Tiles processed:     {performance_metrics['total_tiles']}")
    print(f"  Batches processed:   {performance_metrics['total_batches']}")
    print(
        f"  Tiles per second:    {performance_metrics['throughput_tiles_per_sec']:.2f}"
    )
    print(
        f"  Batches per second:  {performance_metrics['throughput_batches_per_sec']:.2f}"
    )
    print(f"  Average batch size:  {performance_metrics['avg_batch_size']:.1f}")

    print("\nMEMORY USAGE:")
    print(f"  Initial memory:      {performance_metrics['memory_start']:.2f} MB")
    print(f"  Peak memory:         {performance_metrics['memory_peak']:.2f} MB")
    print(f"  Final memory:        {performance_metrics['memory_end']:.2f} MB")
    print(f"  Memory after GC:     {final_memory_after_gc:.2f} MB")
    print(
        f"  Memory increase:     {performance_metrics['memory_end'] - performance_metrics['memory_start']:.2f} MB"
    )

    print("\nCONFIGURATION:")
    print(f"  Tile size:           {performance_metrics['tile_size']}")
    print(f"  Overlap:             {performance_metrics['overlap']}")
    print(f"  Batch size:          {performance_metrics['batch_size']}")
    print(f"  Images processed:    {len(image_paths)}")

    # Performance recommendations
    print("\nPERFORMANCE RECOMMENDATIONS:")
    if performance_metrics["throughput_tiles_per_sec"] < 10:
        print("  ⚠️  Low throughput - consider optimizing image loading")
    if performance_metrics["memory_peak"] > 1000:
        print("  ⚠️  High memory usage - consider reducing batch size or tile size")
    if (
        performance_metrics["dataset_creation_time"]
        > performance_metrics["iteration_time"] * 0.5
    ):
        print("  ⚠️  Dataset creation is slow - consider caching tile calculations")

    print("=" * 60)
    print("✓ create_loader test with profiling completed")
    print("=" * 60)

    # Assertions for test validation
    assert batch_count > 0, "Should have at least one batch"
    assert total_tiles_processed > 0, "Should have processed at least one tile"
    assert (
        performance_metrics["throughput_tiles_per_sec"] > 0
    ), "Should have positive throughput"


def run_all_tests():
    """Run all data loading tests."""
    print("Starting data loading tests...")

    try:
        test_loader_config()
        test_image_tile_dataset()
        test_data_loader()
        test_load_images_as_drone_images()
        test_create_loader()

        print("=" * 60)
        print("ALL DATA LOADING TESTS PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
