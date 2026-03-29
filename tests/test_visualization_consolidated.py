#!/usr/bin/env python3
"""
Consolidated test script for GeographicVisualizer with detection visualization.

This test combines functionality from:
- test_updated_visualization.py
- test_visualization_implementation.py

Tests include:
1. Basic GeographicVisualizer functionality
2. Detection visualization
3. Empty detection filtering
4. Real DroneImage with detections
5. Map creation and saving
"""

import logging
import os
import random
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wildetect.core.config import FlightSpecs
from src.wildetect.core.data import DroneImage
from src.wildetect.core.data.detection import Detection
from src.wildetect.core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)
NUM_IMAGES = 5  # Reduced for faster testing


def create_test_image() -> str:
    """Create a test image path for testing."""
    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    return image_path


def create_test_drone_images() -> list[DroneImage]:
    """Create test DroneImage instances with geographic footprints."""
    drone_images = []

    for _ in range(NUM_IMAGES):
        try:
            image_path = create_test_image()
            drone_image = DroneImage.from_image_path(
                image_path, flight_specs=FLIGHT_SPECS
            )
            drone_images.append(drone_image)
        except Exception as e:
            logger.warning(f"Failed to create drone image: {e}")
            continue

    return drone_images


def create_test_detections(drone_image: DroneImage) -> list[Detection]:
    """Create test detections for a drone image."""
    detections = []

    # Create some test detections (non-empty)
    test_detections = [
        Detection(
            bbox=[10, 20, 50, 80],
            confidence=0.85,
            class_id=1,
            class_name="deer",
            parent_image=drone_image.image_path,
        ),
        Detection(
            bbox=[100, 150, 200, 250],
            confidence=0.92,
            class_id=2,
            class_name="elephant",
            parent_image=drone_image.image_path,
        ),
        Detection(
            bbox=[300, 400, 350, 450],
            confidence=0.78,
            class_id=3,
            class_name="giraffe",
            parent_image=drone_image.image_path,
        ),
    ]

    # Create an empty detection
    empty_detection = Detection.empty(drone_image.image_path)

    return test_detections + [empty_detection]


def test_basic_geographic_visualizer():
    """Test basic GeographicVisualizer functionality."""
    logger.info("=== Testing Basic GeographicVisualizer ===")

    # Create test drone images
    drone_images = create_test_drone_images()
    logger.info(f"Created {len(drone_images)} test drone images")

    if not drone_images:
        logger.warning("No drone images created, skipping test")
        return False

    # Check geographic footprints
    footprints_count = sum(
        1 for img in drone_images if img.geographic_footprint is not None
    )
    logger.info(
        f"Images with geographic footprints: {footprints_count}/{len(drone_images)}"
    )

    # Calculate mean coordinates
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
        map_center = (40.7128, -74.0060)  # Fallback coordinates
        logger.warning("No valid GPS coordinates found, using default center")

    # Create visualizer with custom configuration
    config = VisualizationConfig(
        map_center=map_center,
        zoom_start=12,
        tiles="OpenStreetMap",
        image_bounds_color="purple",
        image_center_color="orange",
        detection_color="green",
        show_image_path=False,
        show_detection_count=True,
        show_gps_info=True,
        show_image_bounds=True,
        show_image_centers=True,
        show_statistics=True,
        show_detections=True,  # Enable detection visualization
    )

    visualizer = GeographicVisualizer(config)
    logger.info("✓ GeographicVisualizer created successfully")

    # Test overlap detection
    logger.info("Testing overlap detection...")
    overlap_map = visualizer._find_overlaps_using_strategy(drone_images)
    logger.info(f"Found {len(overlap_map)} overlapping image pairs")

    # Test coverage statistics
    logger.info("Testing coverage statistics...")
    stats = visualizer.get_coverage_statistics(drone_images)
    logger.info(f"Coverage statistics: {stats}")

    # Create map
    logger.info("Creating map visualization...")
    output_path = "test_basic_geographic_visualization.html"
    visualizer.create_map(drone_images, output_path)
    logger.info("✓ Map created successfully")

    logger.info(f"✓ Map saved to: {output_path}")

    return True


def test_detection_visualization():
    """Test detection visualization functionality."""
    logger.info("=== Testing Detection Visualization ===")

    # Create a real test image
    image_path = create_test_image()
    logger.info(f"Using test image: {image_path}")

    # Create a DroneImage
    drone_image = DroneImage.from_image_path(image_path, flight_specs=FLIGHT_SPECS)

    # Create test detections
    detections = create_test_detections(drone_image)
    drone_image.predictions = detections

    logger.info(f"Created {len(detections)} detections (including empty)")

    # Test filtering empty detections
    all_detections = drone_image.predictions
    non_empty_detections = [det for det in all_detections if not det.is_empty]

    logger.info(f"Total detections: {len(all_detections)}")
    logger.info(f"Non-empty detections: {len(non_empty_detections)}")
    logger.info(f"Empty detections: {len(all_detections) - len(non_empty_detections)}")

    # Verify empty detection is correctly identified
    empty_detection = next((det for det in all_detections if det.is_empty), None)
    assert empty_detection is not None, "Should have an empty detection"
    assert (
        empty_detection.is_empty == True
    ), "Empty detection should be identified as empty"
    assert len(non_empty_detections) == 3, "Should have 3 non-empty detections"

    logger.info("✓ Empty detection filtering works correctly")

    # Test detection visualization
    config = VisualizationConfig(
        show_detections=True,
        detection_color="red",
        show_image_bounds=True,
        show_image_centers=True,
        show_statistics=True,
    )

    visualizer = GeographicVisualizer(config)

    # Only include images with non-empty detections
    images_with_detections = []
    if non_empty_detections:
        images_with_detections.append(drone_image)

    if images_with_detections:
        output_path = Path("test_detection_visualization.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        map_obj = visualizer.create_map(images_with_detections)
        visualizer.save_map(map_obj, str(output_path))
        logger.info(f"✓ Detection visualization saved to: {output_path}")
    else:
        logger.warning("✗ No images with non-empty detections for visualization")
        return False

    return True


def test_consolidated_functionality():
    """Test consolidated functionality with real data and detections."""
    logger.info("=== Testing Consolidated Functionality ===")

    # Create test drone images
    drone_images = create_test_drone_images()

    if not drone_images:
        logger.warning("No drone images created, skipping test")
        return False

    # Add test detections to each drone image
    for drone_image in drone_images:
        detections = create_test_detections(drone_image)
        drone_image.predictions = detections

    # Filter images with detections
    images_with_detections = [
        drone_image
        for drone_image in drone_images
        if any(not det.is_empty for det in drone_image.get_all_predictions())
    ]

    logger.info(f"Images with detections: {len(images_with_detections)}")

    # Create comprehensive configuration
    config = VisualizationConfig(
        zoom_start=12,
        tiles="OpenStreetMap",
        image_bounds_color="purple",
        image_center_color="orange",
        detection_color="green",
        show_image_path=False,
        show_detection_count=True,
        show_gps_info=True,
        show_image_bounds=True,
        show_image_centers=True,
        show_statistics=True,
        show_detections=True,
    )

    # Create visualizer and test all features
    visualizer = GeographicVisualizer(config)

    # Test map creation
    map_obj = visualizer.create_map(images_with_detections)
    logger.info("✓ Map created with all features")

    # Test statistics
    stats = visualizer.get_coverage_statistics(images_with_detections)
    logger.info(f"Coverage statistics: {stats}")

    # Save comprehensive visualization
    output_path = "test_consolidated_visualization.html"
    visualizer.save_map(map_obj, output_path)
    logger.info(f"✓ Consolidated visualization saved to: {output_path}")

    return True


def main():
    """Run all consolidated tests."""
    logger.info("Starting Consolidated Visualization Tests")

    try:
        # Test 1: Basic functionality
        success1 = test_basic_geographic_visualizer()

        # Test 2: Detection visualization
        success2 = test_detection_visualization()

        # Test 3: Consolidated functionality
        success3 = test_consolidated_functionality()

        if success1 and success2 and success3:
            logger.info("=" * 60)
            logger.info("✅ ALL TESTS PASSED")
            logger.info("=" * 60)
            logger.info("Generated files:")
            logger.info("  - test_basic_geographic_visualization.html")
            logger.info("  - test_detection_visualization.html")
            logger.info("  - test_consolidated_visualization.html")
            return True
        else:
            logger.error("❌ Some tests failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
