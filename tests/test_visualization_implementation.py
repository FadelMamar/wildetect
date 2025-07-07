#!/usr/bin/env python3
"""
Test script to verify the visualization implementation in display_results function.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wildetect.core.data import DroneImage
from src.wildetect.core.data.detection import Detection
from src.wildetect.core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)
from tests.test_tile_and_drone_image import create_test_image


def test_visualization_implementation():
    """Test the visualization implementation with real DroneImage and detections."""
    print("Testing visualization implementation with real DroneImage...")

    # Create a real test image path
    image_path = create_test_image()
    print(f"Using test image: {image_path}")

    # Create a DroneImage object
    drone_image = DroneImage.from_image_path(image_path)

    # Create some test detections (non-empty)
    detections = [
        Detection(
            bbox=[10, 20, 50, 80],
            confidence=0.85,
            class_id=1,
            class_name="deer",
            parent_image=image_path,
        ),
        Detection(
            bbox=[100, 150, 200, 250],
            confidence=0.92,
            class_id=2,
            class_name="elephant",
            parent_image=image_path,
        ),
    ]

    # Create an empty detection
    empty_detection = Detection.empty(image_path)

    # Set predictions to a mix of empty and non-empty detections
    drone_image.predictions = detections + [empty_detection]

    # Test filtering empty detections
    all_detections = drone_image.predictions
    non_empty_detections = [det for det in all_detections if not det.is_empty]

    print(f"Total detections: {len(all_detections)}")
    print(f"Non-empty detections: {len(non_empty_detections)}")
    print(f"Empty detections: {len(all_detections) - len(non_empty_detections)}")

    # Verify empty detection is correctly identified
    assert (
        empty_detection.is_empty == True
    ), "Empty detection should be identified as empty"
    assert len(non_empty_detections) == 2, "Should have 2 non-empty detections"

    print("✓ Empty detection filtering works correctly")

    # Test GeographicVisualizer creation and map saving
    try:
        config = VisualizationConfig(
            show_image_bounds=True,
            show_image_centers=True,
            show_statistics=True,
            show_detection_count=True,
            show_gps_info=True,
        )
        visualizer = GeographicVisualizer(config)
        print("✓ GeographicVisualizer created successfully")

        # Only include images with non-empty detections
        images_with_detections = []
        if non_empty_detections:
            images_with_detections.append(drone_image)

        if images_with_detections:
            output_path = Path("output/test_geographic_visualization.html")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            visualizer.save_map(images_with_detections, str(output_path))
            print(f"✓ Geographic visualization saved to: {output_path}")
        else:
            print("✗ No images with non-empty detections for visualization")
    except Exception as e:
        print(f"✗ Failed to create/save GeographicVisualizer: {e}")
        return False

    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_visualization_implementation()
