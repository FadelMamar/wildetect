#!/usr/bin/env python3
"""
Test script for the updated GeographicVisualizer with GPSOverlapStrategy.
"""

import logging
import os
import sys

from wildetect.core.config import FlightSpecs

# Add the src directory to the Python path
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.data.utils import get_images_paths
from wildetect.core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DATA_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)


def create_test_drone_images() -> list[DroneImage]:
    """Create test DroneImage instances with geographic footprints."""
    drone_images = []

    for image_path in get_images_paths(TEST_DATA_DIR):
        drone_image = DroneImage.from_image_path(image_path, flight_specs=FLIGHT_SPECS)

        drone_images.append(drone_image)

    return drone_images


def test_geographic_visualizer():
    """Test the updated GeographicVisualizer."""
    logger.info("Creating test drone images...")
    drone_images = create_test_drone_images()

    logger.info(f"Created {len(drone_images)} test drone images")

    # Check geographic footprints
    footprints_count = sum(
        1 for img in drone_images if img.geographic_footprint is not None
    )
    logger.info(
        f"Images with geographic footprints: {footprints_count}/{len(drone_images)}"
    )

    # Calculate mean latitude and longitude from drone images
    valid_coordinates = []
    for drone_image in drone_images:
        if drone_image.latitude is not None and drone_image.longitude is not None:
            valid_coordinates.append((drone_image.latitude, drone_image.longitude))

    if valid_coordinates:
        mean_lat = sum(coord[0] for coord in valid_coordinates) / len(valid_coordinates)
        mean_lon = sum(coord[1] for coord in valid_coordinates) / len(valid_coordinates)
        map_center = (mean_lat, mean_lon)
        logger.info(f"Map centered at mean coordinates: {mean_lat:.6f}, {mean_lon:.6f}")
    else:
        map_center = (40.7128, -74.0060)  # Fallback to New York coordinates
        logger.warning("No valid GPS coordinates found, using default center")

    # Custom configuration
    config = VisualizationConfig(
        map_center=map_center,
        zoom_start=12,
        tiles="OpenStreetMap",  # Use OpenStreetMap instead of Stamen Terrain
        image_bounds_color="purple",
        image_center_color="orange",
        overlap_color="red",
        show_image_path=False,
        show_detection_count=True,
        show_gps_info=True,
        show_image_bounds=True,
        show_image_centers=True,
        show_statistics=True,
    )

    # Create visualizer
    visualizer = GeographicVisualizer(config)

    logger.info("Testing overlap detection...")
    overlap_map = visualizer._find_overlaps_using_strategy(drone_images)
    logger.info(f"Found {len(overlap_map)} overlapping image pairs")

    for image_path, overlapping_paths in overlap_map.items():
        logger.debug(f"Image {image_path} overlaps with: {overlapping_paths}")

    # Test coverage statistics
    logger.info("Testing coverage statistics...")
    stats = visualizer.get_coverage_statistics(drone_images)
    logger.info(f"Coverage statistics: {stats}")

    # Create map
    logger.info("Creating map visualization...")
    map_obj = visualizer.create_map(drone_images)

    # Save map
    output_path = "test_geographic_visualization.html"
    visualizer.save_map(drone_images, output_path)
    logger.info(f"Map saved to: {output_path}")

    return True


if __name__ == "__main__":
    try:
        success = test_geographic_visualizer()
        if success:
            logger.info("✅ Test completed successfully!")
        else:
            logger.error("❌ Test failed!")
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
