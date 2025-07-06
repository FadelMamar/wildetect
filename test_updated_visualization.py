#!/usr/bin/env python3
"""
Test script for the updated GeographicVisualizer with GPSOverlapStrategy.
"""

import logging
import os
import sys
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wildetect.core.data.drone_image import DroneImage
from wildetect.core.gps.geographic_bounds import GeographicBounds
from wildetect.core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_drone_images():
    """Create mock DroneImage instances with geographic footprints."""
    drone_images = []

    # Create test images with overlapping footprints
    test_data = [
        {
            "image_path": "test_image_1.jpg",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "width": 1920,
            "height": 1080,
            "gsd": 5.0,
            "predictions": [],
        },
        {
            "image_path": "test_image_2.jpg",
            "latitude": 40.7130,
            "longitude": -74.0058,
            "width": 1920,
            "height": 1080,
            "gsd": 5.0,
            "predictions": [],
        },
        {
            "image_path": "test_image_3.jpg",
            "latitude": 40.7140,
            "longitude": -74.0070,
            "width": 1920,
            "height": 1080,
            "gsd": 5.0,
            "predictions": [],
        },
    ]

    for i, data in enumerate(test_data):
        # Create a mock DroneImage without requiring actual image files
        drone_image = Mock(spec=DroneImage)

        # Set basic properties
        drone_image.image_path = data["image_path"]
        drone_image.width = data["width"]
        drone_image.height = data["height"]
        drone_image.gsd = data["gsd"]
        drone_image.latitude = data["latitude"]
        drone_image.longitude = data["longitude"]
        drone_image.tile_gps_loc = f"{data['latitude']},{data['longitude']},100"
        drone_image.predictions = data["predictions"]
        drone_image.id = f"test_image_{i+1}"

        # Create geographic footprint
        bounds = GeographicBounds.from_image_metadata(
            lat_center=data["latitude"],
            lon_center=data["longitude"],
            width_px=data["width"],
            height_px=data["height"],
            gsd=data["gsd"],
        )

        if bounds:
            drone_image.geographic_footprint = bounds
            # Mock the geo_polygon_points property
            polygon_points = bounds.get_polygon_points()
            drone_image.geo_polygon_points = polygon_points if polygon_points else []
        else:
            drone_image.geographic_footprint = None
            drone_image.geo_polygon_points = []

        drone_images.append(drone_image)

    return drone_images


def test_geographic_visualizer():
    """Test the updated GeographicVisualizer."""
    logger.info("Creating mock drone images...")
    drone_images = create_mock_drone_images()

    logger.info(f"Created {len(drone_images)} mock drone images")

    # Check geographic footprints
    footprints_count = sum(
        1 for img in drone_images if img.geographic_footprint is not None
    )
    logger.info(
        f"Images with geographic footprints: {footprints_count}/{len(drone_images)}"
    )

    # Create visualization config
    config = VisualizationConfig(
        show_overlaps=True, show_statistics=True, use_polygons=True
    )

    # Create visualizer
    visualizer = GeographicVisualizer(config)

    logger.info("Testing overlap detection...")
    overlap_map = visualizer._find_overlaps_using_strategy(drone_images)
    logger.info(f"Found {len(overlap_map)} overlapping image pairs")

    for image_path, overlapping_paths in overlap_map.items():
        logger.info(f"Image {image_path} overlaps with: {overlapping_paths}")

    # Test coverage statistics
    logger.info("Testing coverage statistics...")
    stats = visualizer.get_coverage_statistics(drone_images)
    logger.info(f"Coverage statistics: {stats}")

    # Test map creation (without saving to avoid file system issues)
    logger.info("Testing map creation...")
    try:
        map_obj = visualizer.create_map(drone_images)
        logger.info("✅ Map creation successful!")

        # Test the map object has the expected attributes
        if hasattr(map_obj, "_name") and map_obj._name == "folium":
            logger.info("✅ Map object is valid Folium map!")
        else:
            logger.warning("⚠️ Map object may not be a valid Folium map")

    except Exception as e:
        logger.error(f"❌ Map creation failed: {e}")
        return False

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
