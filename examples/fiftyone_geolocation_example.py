"""
Example demonstrating FiftyOne native geolocation support.

This example shows how to use the updated FiftyOneManager with native
FiftyOne geolocation features using fo.GeoLocation.
"""

import os
import random
import logging
from pathlib import Path
from typing import List

import fiftyone as fo

from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.visualization.fiftyone_manager import FiftyOneManager
from wildetect.core.config import FlightSpecs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"

def load_image_path():
    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    return image_path

def create_sample_detections() -> List[Detection]:
    """Create sample Detection objects for demonstration."""
    detections = []

    # Sample detection 1
    detection1 = Detection(
        bbox=[100, 150, 200, 250],  # [x1, y1, x2, y2]
        confidence=0.85,
        class_id=1,
        class_name="giraffe",
    )
    detections.append(detection1)

    # Sample detection 2
    detection2 = Detection(
        bbox=[300, 400, 400, 500],
        confidence=0.72,
        class_id=2,
        class_name="elephant",
    )
    detections.append(detection2)

    return detections


def create_sample_drone_image() -> DroneImage:
    """Create a sample DroneImage for demonstration."""
    # Create a drone image with sample data
    drone_image = DroneImage.from_image_path(
        image_path=load_image_path(),
        flight_specs=FlightSpecs(
            sensor_height=24.0,
            focal_length=35.0,
            flight_height=180.0,
        ),
    )

    # Add some sample detections to the drone image
    detections = create_sample_detections()
    drone_image.set_predictions(detections,update_gps=True)

    print(drone_image.get_non_empty_predictions())

    return drone_image


def demonstrate_native_geolocation():
    """Demonstrate FiftyOne's native geolocation features."""

    # Create a sample with native FiftyOne geolocation
    sample = fo.Sample(filepath=load_image_path())

    # Add native FiftyOne geolocation
    sample["location"] = fo.GeoLocation(
        point=[-74.0060, 40.7128],  # [longitude, latitude]
        polygon=[
            [
                [-74.0160, 40.7028],  # Bottom-left
                [-73.9960, 40.7028],  # Bottom-right
                [-73.9960, 40.7228],  # Top-right
                [-74.0160, 40.7228],  # Top-left
                [-74.0160, 40.7028],  # Close polygon
            ]
        ],
    )

    logger.info("Created sample with native FiftyOne geolocation:")
    logger.info(f"Point: {sample['location'].point}")
    logger.info(f"Polygon: {sample['location'].polygon}")


def main():
    """Main example function."""
    logger.info("Starting FiftyOne native geolocation example")

    # Demonstrate native FiftyOne geolocation
    demonstrate_native_geolocation()

    # Initialize FiftyOne manager
    fiftyone_manager = FiftyOneManager("wildlife_geolocation_dataset_example")

    # Create sample data
    drone_image = create_sample_drone_image()

    # Add drone image to dataset (now uses native geolocation)
    logger.info("Adding drone image with native geolocation to FiftyOne dataset")
    fiftyone_manager.add_drone_images([drone_image])
 
    # Get samples with GPS data (now uses native location field)
    gps_samples = fiftyone_manager.get_detections_with_gps()
    logger.info(f"Found {len(gps_samples)} samples with native geolocation data")

    # Get dataset statistics
    stats = fiftyone_manager.get_annotation_stats()
    logger.info(f"Dataset statistics: {stats}")

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
    logger.info("Native geolocation example completed successfully")


if __name__ == "__main__":
    main()
