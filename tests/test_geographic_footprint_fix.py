#!/usr/bin/env python3
"""
Test script to verify the geographic footprint fix for Detection objects.
"""

import logging
import os
import random
import sys

from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.tile import Tile
from wildetect.core.gps.gps_service import GPSDetectionService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST_GEO_FOOTPRINT")

# Create a test image

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)


def test_geographic_footprint_fix():
    """Test that geographic footprints are properly computed for detections."""
    logger.info("Testing geographic footprint fix...")

    # Create a test detection
    detection = Detection(
        bbox=[100, 150, 200, 250],  # [x1, y1, x2, y2]
        confidence=0.9,
        class_id=0,
        class_name="person",
    )

    # Create a mock tile (in real usage, this would be created from an actual image)
    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    tile = Tile(
        image_path=image_path,  # Mock path
        flight_specs=FLIGHT_SPECS,
    )

    # Test the fix
    logger.info("Before GPS update:")
    logger.info(f"  Detection geo_box: {detection.geo_box}")
    logger.info(f"  Detection geographic_footprint: {detection.geographic_footprint}")

    # Update detection with GPS data
    try:
        GPSDetectionService.update_detection_gps(detection, tile)

        logger.info("After GPS update:")
        logger.info(f"  Detection geo_box: {detection.geo_box}")
        logger.info(
            f"  Detection geographic_footprint: {detection.geographic_footprint}"
        )

        if detection.geographic_footprint is not None:
            logger.info("✓ Geographic footprint computed successfully!")
            logger.info(f"  North: {detection.geographic_footprint.north}")
            logger.info(f"  South: {detection.geographic_footprint.south}")
            logger.info(f"  East: {detection.geographic_footprint.east}")
            logger.info(f"  West: {detection.geographic_footprint.west}")
        else:
            logger.warning("✗ Geographic footprint computation failed")

    except Exception as e:
        logger.error(f"✗ Error during GPS update: {e}")


def test_validation_improvements():
    """Test the validation improvements in the geographic footprint computation."""
    logger.info("Testing validation improvements...")

    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    tile = Tile(
        image_path=image_path,
        flight_specs=FLIGHT_SPECS,
    )

    # Test with detection outside tile bounds
    detection_outside = Detection(
        bbox=[tile.width - 10, 150, tile.width + 50, 300],  # Outside tile bounds
        confidence=0.9,
        class_id=0,
        class_name="person",
    )

    # This should log a warning about bbox being outside bounds
    GPSDetectionService.update_detection_gps(detection_outside, tile)

    if detection_outside.geographic_footprint is None:
        logger.info("✓ Correctly rejected detection outside tile bounds")
    else:
        logger.warning("✗ Should have rejected detection outside tile bounds")
        assert False


if __name__ == "__main__":
    test_geographic_footprint_fix()
    test_validation_improvements()
    logger.info("Test completed!")
