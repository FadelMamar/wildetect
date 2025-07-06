"""
Example demonstrating IoU logging during duplicate removal.
"""

import logging

from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.flight.geographic_merger import GeographicMerger

# Configure logging to see IoU information
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_overlapping_detections():
    """Create test detections with known overlap."""
    # Create detections with overlapping bounding boxes
    det1 = Detection(
        bbox=[100, 100, 200, 200],  # x1, y1, x2, y2
        class_id=0,
        class_name="person",
        confidence=0.9,
    )

    det2 = Detection(
        bbox=[150, 150, 250, 250],  # Overlapping with det1
        class_id=0,
        class_name="person",
        confidence=0.85,
    )

    det3 = Detection(
        bbox=[300, 300, 400, 400],  # Non-overlapping
        class_id=0,
        class_name="car",
        confidence=0.8,
    )

    return det1, det2, det3


def main():
    """Demonstrate IoU logging during duplicate removal."""
    print("🚁 IoU Logging Example for Duplicate Removal")
    print("=" * 60)

    # Initialize merger
    merger = GeographicMerger(merge_distance_threshold_m=50.0)

    # Create test drone images with overlapping detections
    det1, det2, det3 = create_overlapping_detections()

    # Create drone images (you would normally load these from files)
    # For this example, we'll create mock images
    drone_image_1 = DroneImage.from_image_path(
        image_path="test_image_1.jpg",
        flight_specs=FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        ),
    )

    drone_image_2 = DroneImage.from_image_path(
        image_path="test_image_2.jpg",
        flight_specs=FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        ),
    )

    # Set predictions
    drone_image_1.set_predictions([det1, det3])  # Person and car
    drone_image_2.set_predictions([det2])  # Overlapping person

    print(f"\n📊 Initial detections:")
    print(f"  Image 1: {len(drone_image_1.predictions)} detections")
    for i, det in enumerate(drone_image_1.predictions):
        print(f"    {i+1}. {det.class_name} (confidence: {det.confidence:.3f})")

    print(f"  Image 2: {len(drone_image_2.predictions)} detections")
    for i, det in enumerate(drone_image_2.predictions):
        print(f"    {i+1}. {det.class_name} (confidence: {det.confidence:.3f})")

    # Create overlap map
    overlap_map = {
        str(drone_image_1.image_path): [str(drone_image_2.image_path)],
        str(drone_image_2.image_path): [str(drone_image_1.image_path)],
    }

    print(f"\n🔍 Running duplicate removal with IoU logging...")
    print(f"  IoU threshold: 0.5")
    print(f"  Overlapping images: {len(overlap_map)} pairs")

    # Run duplicate removal
    stats = merger.duplicate_removal_strategy.remove_duplicates(
        [drone_image_1, drone_image_2], overlap_map, iou_threshold=0.5
    )

    print(f"\n✅ Duplicate removal completed!")
    print(f"\n📈 Statistics:")
    print(f"  Total image pairs processed: {stats['total_image_pairs_processed']}")
    print(f"  Total detections removed: {stats['total_detections_removed']}")
    print(f"  Duplicate removal rate: {stats['duplicate_removal_rate']:.2%}")

    print(f"\n📊 Final detections:")
    print(f"  Image 1: {len(drone_image_1.predictions)} detections")
    for i, det in enumerate(drone_image_1.predictions):
        print(f"    {i+1}. {det.class_name} (confidence: {det.confidence:.3f})")

    print(f"  Image 2: {len(drone_image_2.predictions)} detections")
    for i, det in enumerate(drone_image_2.predictions):
        print(f"    {i+1}. {det.class_name} (confidence: {det.confidence:.3f})")

    print(f"\n💡 What the logs show:")
    print(f"  • IoU values for each duplicate pair")
    print(f"  • Confidence scores of competing detections")
    print(f"  • Centroid distances used for decision making")
    print(f"  • Which detection was kept and why")
    print(f"  • Summary of how many detections were removed from each tile")


if __name__ == "__main__":
    main()
