"""
Test script for data loading functionality.
"""

import logging
import os
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
RASTER_PATH = r"C:\Users\FADELCO\Downloads\day5_ebenhazer_burnt_geier_merged_archive\transparent_day_5_ebenhazer_burnt_geier_merged_mosaic_group1.tif"

if not os.path.exists(RASTER_PATH):
    raise FileNotFoundError(f"Raster path {RASTER_PATH} does not exist")

if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Images directory {IMAGES_DIR} does not exist")


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


def test_ortho_dataset():
    """Test OrthoDataset functionality."""
    print("=" * 60)
    print("TESTING ORTHO DATASET")
    print("=" * 60)

    # Create config
    config = LoaderConfig(
        tile_size=800,
        overlap=0.2,
        batch_size=4,
        num_workers=0,
    )

    # Create dataset
    loader = DataLoader(raster_path=RASTER_PATH, config=config)

    for batch in loader:
        print(f"{len(batch)} tiles")

    print(f"Dataset created with {len(loader)} tiles")

    # Test dataset iteration


def run_all_tests():
    """Run all data loading tests."""
    print("Starting data loading tests...")

    try:
        test_loader_config()
        test_image_tile_dataset()
        test_data_loader()
        test_load_images_as_drone_images()
        test_ortho_dataset()

        print("=" * 60)
        print("ALL DATA LOADING TESTS PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
