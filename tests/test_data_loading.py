"""
Test script for data loading functionality.
"""

import logging
import os
import shutil
import sys
import tempfile
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


def create_test_images(test_dir: Path, num_images: int = 5) -> List[str]:
    """Create test images for testing.

    Args:
        test_dir (Path): Directory to create test images in
        num_images (int): Number of test images to create

    Returns:
        List[str]: List of created image paths
    """
    import numpy as np
    from PIL import Image

    test_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for i in range(num_images):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Save with different formats
        if i % 2 == 0:
            img_path = test_dir / f"test_image_{i}.jpg"
        else:
            img_path = test_dir / f"test_image_{i}.png"

        img.save(img_path)
        image_paths.append(str(img_path))

    logger.info(f"Created {len(image_paths)} test images in {test_dir}")
    return image_paths


def test_image_discovery():
    """Test image discovery functionality."""
    logger.info("=" * 60)
    logger.info("TESTING IMAGE DISCOVERY")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=5)

        # Test get_images_paths function
        discovered_paths = get_images_paths(temp_dir)
        discovered_paths = [str(p) for p in discovered_paths]

        logger.info(f"Expected images: {len(image_paths)}")
        logger.info(f"Discovered images: {len(discovered_paths)}")

        # Check if all expected images were discovered
        missing_images = set(image_paths) - set(discovered_paths)
        extra_images = set(discovered_paths) - set(image_paths)

        if missing_images:
            logger.error(f"Missing images: {missing_images}")
        if extra_images:
            logger.warning(f"Extra images found: {extra_images}")

        assert len(missing_images) == 0, f"Missing images: {missing_images}"
        logger.info("✓ Image discovery test passed")


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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=3)

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
        for i in range(min(2, len(dataset))):
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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=4)

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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=3)

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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=3)

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
        test_image_discovery()
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
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
