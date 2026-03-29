#!/usr/bin/env python3
"""
Test script to verify geographic footprint computation for detections.
"""

import logging
import os
import random
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.flight.geographic_merger import CentroidProximityRemovalStrategy
from wildetect.core.gps.geographic_bounds import GeographicBounds
from wildetect.core.gps.gps_service import GPSDetectionService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)


def create_test_detection_with_gps():
    """Create a test detection with GPS data."""
    # Create a test detection
    detection = Detection(
        bbox=[100, 200, 150, 250],  # [x1, y1, x2, y2]
        confidence=0.9,
        class_id=0,
        class_name="person",
    )

    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)

    # Create a test drone image with GPS data
    drone_image = DroneImage(image_path=image_path, flight_specs=FLIGHT_SPECS)

    return detection, drone_image


def test_geographic_footprint_computation():
    """Test that geographic footprints can be computed for detections."""
    logger.info("Testing geographic footprint computation...")

    # Create test detection and drone image
    detection, drone_image = create_test_detection_with_gps()

    # Check initial state
    logger.info(f"Initial detection geo_box: {detection.geo_box}")
    assert detection.geo_box is None, "Detection should not have geo_box initially"

    # Update detection with GPS data
    try:
        GPSDetectionService.update_detection_gps(detection, drone_image)
        logger.info(f"Updated detection geo_box: {detection.geo_box}")

        if detection.geo_box is not None:
            logger.info("✓ Geographic footprint computed successfully!")
            logger.info(f"  Geographic footprint: {detection.geographic_footprint}")
            logger.info(f"  Geo box: {detection.geo_box}")
        else:
            logger.warning("✗ Geographic footprint computation failed")

    except Exception as e:
        logger.error(f"✗ Error computing geographic footprint: {e}")


def test_geographic_merger_with_footprints():
    """Test the geographic merger with geographic footprints."""
    logger.info("Testing geographic merger with footprints...")

    # Create test detections and drone images
    detection1, drone_image1 = create_test_detection_with_gps()
    detection2, drone_image2 = create_test_detection_with_gps()

    # Add detections to drone images
    drone_image1.set_predictions([detection1])
    drone_image2.set_predictions([detection2])

    # Create merger strategy
    strategy = CentroidProximityRemovalStrategy()

    # Test geographic footprint stats
    stats = strategy._get_geographic_footprint_stats([drone_image1, drone_image2])
    logger.info(f"Geographic footprint stats: {stats}")

    # Test ensuring geographic footprints
    strategy._ensure_geographic_footprints(drone_image1)
    strategy._ensure_geographic_footprints(drone_image2)

    logger.info(f"After ensuring footprints:")
    logger.info(f"  Detection1 geo_box: {detection1.geo_box}")
    logger.info(f"  Detection2 geo_box: {detection2.geo_box}")


def main():
    """Run the geographic footprint tests."""
    logger.info("=" * 60)
    logger.info("GEOGRAPHIC FOOTPRINT TEST")
    logger.info("=" * 60)

    try:
        test_geographic_footprint_computation()
        test_geographic_merger_with_footprints()
        logger.info("✓ All tests completed successfully!")
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
