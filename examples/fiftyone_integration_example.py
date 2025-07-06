"""
Example demonstrating FiftyOne integration with WildDetect.

This example shows how to use the FiftyOneManager to visualize
wildlife detection results from DroneImage and Detection objects.
"""

import logging
from pathlib import Path
from typing import List

from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.visualization.fiftyone_manager import FiftyOneManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_detections() -> List[Detection]:
    """Create sample Detection objects for demonstration."""
    detections = []

    # Sample detection 1
    detection1 = Detection(
        bbox=[100, 150, 200, 250],  # [x1, y1, x2, y2]
        confidence=0.85,
        class_id=1,
        class_name="deer",
        gps_loc="40.7128,-74.0060",  # Sample GPS coordinates
        parent_image="/path/to/sample_image.jpg",
    )
    detections.append(detection1)

    # Sample detection 2
    detection2 = Detection(
        bbox=[300, 400, 400, 500],
        confidence=0.72,
        class_id=2,
        class_name="bear",
        gps_loc="40.7128,-74.0060",
        parent_image="/path/to/sample_image.jpg",
    )
    detections.append(detection2)

    return detections


def create_sample_drone_image() -> DroneImage:
    """Create a sample DroneImage for demonstration."""
    # Create a drone image with sample data
    drone_image = DroneImage.from_image_path(
        image_path="/path/to/drone_image.jpg",
        latitude=40.7128,
        longitude=-74.0060,
        gsd=0.1,  # Ground sample distance in meters
        timestamp="2024-01-15T10:30:00Z",
    )

    # Add some sample detections to the drone image
    detections = create_sample_detections()
    for detection in detections:
        detection.parent_image = drone_image.image_path

    # Add detections to the drone image (this would normally come from model predictions)
    drone_image.predictions = detections

    return drone_image


def main():
    """Main example function."""
    logger.info("Starting FiftyOne integration example")

    # Initialize FiftyOne manager
    fiftyone_manager = FiftyOneManager("wildlife_detection_dataset")

    # Create sample data
    drone_image = create_sample_drone_image()
    detections = create_sample_detections()

    # Add drone image to dataset
    logger.info("Adding drone image to FiftyOne dataset")
    fiftyone_manager.add_drone_image(drone_image)

    # Add individual detections to dataset
    logger.info("Adding individual detections to FiftyOne dataset")
    fiftyone_manager.add_detections(detections, "/path/to/another_image.jpg")

    # Get dataset statistics
    stats = fiftyone_manager.get_annotation_stats()
    logger.info(f"Dataset statistics: {stats}")

    # Get samples with GPS data
    gps_samples = fiftyone_manager.get_detections_with_gps()
    logger.info(f"Found {len(gps_samples)} samples with GPS data")

    # Get dataset info
    info = fiftyone_manager.get_dataset_info()
    logger.info(f"Dataset info: {info}")

    # Save the dataset
    fiftyone_manager.save_dataset()
    logger.info("Dataset saved successfully")

    # Launch FiftyOne app (uncomment to launch)
    # logger.info("Launching FiftyOne app...")
    # fiftyone_manager.launch_app()

    # Close the dataset
    fiftyone_manager.close()
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
