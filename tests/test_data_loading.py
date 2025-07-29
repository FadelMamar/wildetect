"""
Test script for data loading functionality.
"""

import logging
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

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
    image_paths = create_test_images(num_images=3)

    # Create config
    config = LoaderConfig(
        tile_size=320,  # Smaller tiles for testing
        overlap=0.2,
        batch_size=2,
    )

    # Create dataset
    dataset = TileDataset(config, image_paths=image_paths)

    logger.info(f"Dataset created with {len(dataset.tiles)} tiles")
    logger.info(f"Number of images: {len(dataset.image_paths)}")

    # Test dataset iteration
    for i in range(len(dataset)):
        item = dataset[i]
        logger.info(
            f"Item {i}: tile_id={item['tile_id']}, shape={item['image'].shape if hasattr(item['image'], 'shape') else 'N/A'}"
        )

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
        config=config, image_paths=image_paths, max_images=2
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
    """Test create_loader function."""
    logger.info("=" * 60)
    logger.info("TESTING CREATE_LOADER")
    logger.info("=" * 60)

    # Create test images
    image_paths = create_test_images(num_images=-1)

    # Create config
    config = LoaderConfig(tile_size=320, overlap=0.2, batch_size=2)

    # Create loader
    loader = DataLoader(image_paths=image_paths, image_dir=None, config=config)

    logger.info(f"Created loader with {len(loader)} batches")
    assert len(loader) > 0, "Loader should have batches"

    # Test loader iteration
    batch_count = 0
    for _ in loader:
        batch_count += 1
    assert batch_count > 0, "Should have at least one batch"
    logger.info("✓ create_loader test passed")


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
