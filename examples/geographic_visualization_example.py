#!/usr/bin/env python3
"""
Example script demonstrating geographic bounds visualization for drone images.

This script shows how to:
1. Load drone images with GPS data
2. Visualize their geographic footprints on an interactive map
3. Analyze coverage and overlap statistics
"""

import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
from wildetect.core.config import FlightSpecs
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.data.utils import get_images_paths
from wildetect.core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
    visualize_geographic_bounds,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example configuration
EXAMPLE_IMAGE_DIR = (
    r"D:\workspace\data\savmap_dataset_v2\raw\images"  # Update this path
)
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)


def create_sample_drone_images() -> list[DroneImage]:
    """Create sample drone images for demonstration.

    In a real scenario, you would load actual drone images with GPS data.
    """
    drone_images = []

    # Example: Create drone images from a directory
    if os.path.exists(EXAMPLE_IMAGE_DIR):
        image_files = get_images_paths(EXAMPLE_IMAGE_DIR)

        for image_file in image_files:  # Limit to first 5 images for demo
            try:
                drone_image = DroneImage(
                    image_path=image_file, flight_specs=FLIGHT_SPECS
                )
                drone_images.append(drone_image)
                logger.debug(f"Loaded drone image: {image_file}")
            except Exception as e:
                logger.warning(f"Failed to load {image_file}: {e}")

    return drone_images


def demonstrate_custom_configuration():
    """Demonstrate custom visualization configuration."""
    logger.info("=== Custom Configuration Demo ===")

    drone_images = create_sample_drone_images()

    if not drone_images:
        return

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
    custom_config = VisualizationConfig(
        map_center=map_center,
        zoom_start=12,
        tiles="OpenStreetMap",  # Use OpenStreetMap instead of Stamen Terrain
        image_bounds_color="purple",
        image_center_color="orange",
        overlap_color="red",
        show_image_path=False,
        show_image_bounds=True,
        show_detection_count=True,
        show_gps_info=True,
        show_image_centers=True,
        show_statistics=True,
    )

    # Use convenience function
    map_obj = visualize_geographic_bounds(
        drone_images=drone_images,
        output_path="custom_visualization.html",
        config=custom_config,
    )

    logger.info("Custom visualization created with terrain tiles and custom colors")


def demonstrate_statistics_analysis():
    """Demonstrate coverage statistics analysis."""
    logger.info("=== Statistics Analysis Demo ===")

    drone_images = create_sample_drone_images()

    if not drone_images:
        return

    visualizer = GeographicVisualizer()
    stats = visualizer.get_coverage_statistics(drone_images)

    # Analyze coverage
    total_area = stats["coverage_area"]
    overlap_area = sum(stats["overlap_areas"])
    overlap_percentage = stats["overlap_percentage"]

    logger.info("Coverage Analysis:")
    logger.info(f"  Total Coverage Area: {total_area:.2f} m²")
    logger.info(f"  Total Overlap Area: {overlap_area:.2f} m²")
    logger.info(f"  Overlap Percentage: {overlap_percentage:.2f}%")
    logger.info(f"  Average Overlap Area: {stats['average_overlap_area']:.2f} m²")

    # GPS coverage analysis
    gps_coverage = (stats["images_with_gps"] / stats["total_images"]) * 100
    footprint_coverage = (stats["images_with_footprints"] / stats["total_images"]) * 100

    logger.info("GPS Coverage Analysis:")
    logger.info(
        f"  Images with GPS: {stats['images_with_gps']}/{stats['total_images']} ({gps_coverage:.1f}%)"
    )
    logger.info(
        f"  Images with Footprints: {stats['images_with_footprints']}/{stats['total_images']} ({footprint_coverage:.1f}%)"
    )


def main():
    """Main demonstration function."""
    logger.info("Starting Geographic Bounds Visualization Demo")

    try:
        # Run demonstrations
        demonstrate_custom_configuration()
        demonstrate_statistics_analysis()

        logger.info("Demo completed successfully!")
        logger.info("Check the generated HTML files for interactive maps.")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info("Make sure to:")
        logger.info("1. Update EXAMPLE_IMAGE_DIR to point to your drone images")
        logger.info("2. Install required dependencies: pip install folium pyproj")
        logger.info("3. Ensure your images have GPS EXIF data")


if __name__ == "__main__":
    main()
