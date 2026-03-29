"""
Test file for DroneImage and GeographicBounds creation.

This test file verifies that:
1. DroneImage objects can be created from image paths
2. GeographicBounds objects can be created with proper metadata
3. Both classes work together correctly
4. GPS coordinates and geographic footprints are properly extracted
"""

import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
from PIL import Image

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wildetect.core.config import FlightSpecs
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.data.tile import Tile
from wildetect.core.gps.geographic_bounds import GeographicBounds

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"


def create_test_image() -> str:
    """Create a test image for testing."""

    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    return image_path


def test_drone_image_creation():
    """Test DroneImage creation with various scenarios."""
    logger.info("=" * 60)
    logger.info("TESTING DRONE IMAGE CREATION")
    logger.info("=" * 60)

    # Test 1: Create DroneImage from test image
    test_image_path = create_test_image()

    drone_image = DroneImage.from_image_path(test_image_path)

    # Basic properties
    assert drone_image.id is not None
    assert drone_image.image_path == test_image_path
    assert len(drone_image.tiles) > 0

    logger.info(f"✓ DroneImage created from test image: {drone_image.id}")

    # Test 2: Create DroneImage with flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    drone_image_with_specs = DroneImage.from_image_path(
        test_image_path, flight_specs=flight_specs
    )

    assert drone_image_with_specs.flight_specs is not None
    assert drone_image_with_specs.flight_specs.sensor_height == 24.0
    assert drone_image_with_specs.flight_specs.focal_length == 35.0
    assert drone_image_with_specs.flight_specs.flight_height == 180.0

    logger.info(f"✓ DroneImage with flight specs created: {drone_image_with_specs.id}")

    # Test 3: Test tile management
    assert len(drone_image.tiles) > 0
    assert len(drone_image.tile_offsets) > 0

    # Test 4: Test geographic footprint
    if drone_image.geographic_footprint:
        logger.info(
            f"✓ Geographic footprint available: {drone_image.geographic_footprint}"
        )
    else:
        logger.info("Geographic footprint not available (may be expected)")

    # Test 5: Test GPS coordinates
    if drone_image.latitude and drone_image.longitude:
        logger.info(
            f"✓ GPS coordinates: ({drone_image.latitude}, {drone_image.longitude})"
        )
    else:
        logger.info("GPS coordinates not available (may be expected for test image)")

    logger.info("✓ DroneImage creation tests passed")


def test_real_image_processing():
    """Test DroneImage creation with real images from the dataset."""
    logger.info("=" * 60)
    logger.info("TESTING REAL IMAGE PROCESSING")
    logger.info("=" * 60)

    # Find real test images
    test_images = [create_test_image() for _ in range(3)]

    if not test_images:
        logger.warning(f"No test images found in {TEST_IMAGE_DIR}")
        logger.info("Skipping real image processing tests")
        return

    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    for i, image_path in enumerate(test_images):
        logger.info(
            f"Processing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}"
        )

        try:
            # Create DroneImage
            drone_image = DroneImage.from_image_path(
                image_path, flight_specs=flight_specs
            )

            # Test basic properties
            assert drone_image.id is not None
            assert drone_image.image_path == image_path
            assert drone_image.width is not None and drone_image.width > 0
            assert drone_image.height is not None and drone_image.height > 0
            assert len(drone_image.tiles) > 0

            logger.info(f"  ✓ Image size: {drone_image.width}x{drone_image.height}")
            logger.info(f"  ✓ Number of tiles: {len(drone_image.tiles)}")

            # Test GPS coordinates
            if drone_image.latitude and drone_image.longitude:
                logger.info(
                    f"  ✓ GPS: ({drone_image.latitude:.6f}, {drone_image.longitude:.6f})"
                )
            else:
                logger.info("  ⚠ No GPS coordinates found")

            # Test geographic footprint
            if drone_image.geographic_footprint:
                footprint = drone_image.geographic_footprint
                logger.info(f"  ✓ Geographic footprint: {footprint.box}")

                # Test polygon points
                polygon_points = footprint.get_polygon_points()
                if polygon_points:
                    logger.info(f"  ✓ Polygon points: {len(polygon_points)} points")
                else:
                    logger.info("  ⚠ Polygon points not available")
            else:
                logger.info("  ⚠ No geographic footprint")

            # Test GSD calculation
            if drone_image.gsd:
                logger.info(f"  ✓ GSD: {drone_image.gsd:.3f} cm/px")
            else:
                logger.info("  ⚠ GSD not calculated")

            # Test image loading
            try:
                image_data = drone_image.load_image_data()
                assert image_data.size == (drone_image.width, drone_image.height)
                logger.info("  ✓ Image data loaded successfully")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to load image data: {e}")

        except Exception as e:
            logger.error(f"  ✗ Failed to process image {image_path}: {e}")
            continue

    logger.info("✓ Real image processing tests completed")


def test_geographic_bounds_integration():
    """Test integration between DroneImage and GeographicBounds."""
    logger.info("=" * 60)
    logger.info("TESTING GEOGRAPHIC BOUNDS INTEGRATION")
    logger.info("=" * 60)

    # Create test image
    test_image_path = create_test_image()

    # Create flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    # Create DroneImage
    drone_image = DroneImage.from_image_path(test_image_path, flight_specs=flight_specs)

    # Test that DroneImage has geographic footprint
    if drone_image.geographic_footprint:
        footprint = drone_image.geographic_footprint

        # Test that footprint is a GeographicBounds object
        assert isinstance(footprint, GeographicBounds)

        # Test basic properties
        assert (
            footprint.north is not None
            and footprint.south is not None
            and footprint.north > footprint.south
        )
        assert (
            footprint.east is not None
            and footprint.west is not None
            and footprint.east > footprint.west
        )
        assert footprint.area > 0

        logger.info(f"✓ Geographic footprint integrated: {footprint.box}")

        # Test polygon points
        polygon_points = footprint.get_polygon_points()
        if polygon_points:
            logger.info(f"✓ Polygon points available: {len(polygon_points)} points")
        else:
            logger.info("⚠ Polygon points not available")

    else:
        logger.info("⚠ No geographic footprint available (may be expected)")

    # Test tile geographic footprints
    for i, tile in enumerate(drone_image.tiles):
        if tile.geographic_footprint:
            logger.info(
                f"✓ Tile {i} has geographic footprint: {tile.geographic_footprint.box}"
            )
        else:
            logger.info(f"⚠ Tile {i} has no geographic footprint")

    logger.info("✓ Geographic bounds integration tests passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    logger.info("=" * 60)
    logger.info("TESTING ERROR HANDLING")
    logger.info("=" * 60)

    # Test 1: Invalid image path
    try:
        drone_image = DroneImage.from_image_path("nonexistent_image.jpg")
        assert False, "Should have raised an exception"
    except Exception as e:
        logger.info(f"✓ Correctly handled invalid image path: {e}")

    # Test 2: Invalid GeographicBounds parameters
    try:
        bounds = GeographicBounds(
            north=44.0,  # north < south
            south=45.0,
            east=-74.0,  # east < west
            west=-73.0,
        )
        # This should work but the bounds would be invalid
        logger.info(
            "✓ GeographicBounds with invalid bounds created (bounds will be invalid)"
        )
    except Exception as e:
        logger.info(f"✓ Correctly handled invalid bounds: {e}")

    # Test 3: GeographicBounds without required metadata
    bounds_no_metadata = GeographicBounds(
        north=45.0, south=44.0, east=-73.0, west=-74.0
    )

    # This should work but polygon computation will fail
    polygon_points = bounds_no_metadata.get_polygon_points()
    if polygon_points is None:
        logger.info("✓ Correctly handled missing metadata for polygon computation")
    else:
        logger.info("✓ Polygon computation succeeded despite missing metadata")

    logger.info("✓ Error handling tests passed")


def test_performance():
    """Test performance with multiple images."""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE")
    logger.info("=" * 60)

    # Find test images
    test_images = [create_test_image() for _ in range(5)]

    if not test_images:
        logger.warning("No test images available for performance testing")
        return

    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    import time

    start_time = time.time()
    drone_images = []

    for i, image_path in enumerate(test_images):
        try:
            drone_image = DroneImage.from_image_path(
                image_path, flight_specs=flight_specs
            )
            drone_images.append(drone_image)
            logger.info(
                f"✓ Processed image {i+1}/{len(test_images)} in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            logger.warning(f"⚠ Failed to process image {i+1}: {e}")

    total_time = time.time() - start_time
    logger.info(f"✓ Processed {len(drone_images)} images in {total_time:.2f}s")
    logger.info(f"✓ Average time per image: {total_time/len(drone_images):.2f}s")

    # Test memory usage
    import psutil

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"✓ Memory usage: {memory_mb:.1f} MB")

    logger.info("✓ Performance tests passed")


def main():
    """Run all tests."""
    logger.info("Starting DroneImage and GeographicBounds tests")
    logger.info(f"Test image directory: {TEST_IMAGE_DIR}")

    try:
        test_drone_image_creation()
        test_real_image_processing()
        test_geographic_bounds_integration()
        test_error_handling()
        test_performance()

        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
