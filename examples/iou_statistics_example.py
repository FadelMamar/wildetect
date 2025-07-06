"""
Example demonstrating how to access and analyze IoU statistics from duplicate removal.
"""

import json
import logging
from typing import Any, Dict, List

from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.flight.geographic_merger import GeographicMerger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_iou_statistics(stats: Dict[str, Any]) -> None:
    """Analyze and display IoU statistics from duplicate removal."""

    print("=" * 60)
    print("IoU STATISTICS ANALYSIS")
    print("=" * 60)

    if "iou_statistics" not in stats:
        print("No IoU statistics available")
        return

    iou_stats_list = stats["iou_statistics"]

    print(f"\n📊 IoU Statistics Summary:")
    print(f"  • Total tile pairs processed: {len(iou_stats_list)}")

    # Aggregate statistics across all tile pairs
    all_ious = []
    all_duplicate_pairs = []
    total_ious_computed = 0

    for i, tile_pair_stats in enumerate(iou_stats_list):
        print(f"\n  Tile Pair {i+1}:")
        print(f"    • IoU matrix shape: {tile_pair_stats['iou_matrix_shape']}")
        print(f"    • Total IoUs computed: {tile_pair_stats['total_ious_computed']}")
        print(
            f"    • Above threshold count: {tile_pair_stats['above_threshold_count']}"
        )

        # IoU range
        iou_range = tile_pair_stats["iou_range"]
        print(f"    • IoU range: {iou_range['min']:.3f} to {iou_range['max']:.3f}")
        print(f"    • Mean IoU: {iou_range['mean']:.3f}")

        # Duplicate pairs
        duplicate_pairs = tile_pair_stats["duplicate_pairs"]
        print(f"    • Duplicate pairs found: {len(duplicate_pairs)}")

        for j, pair in enumerate(duplicate_pairs):
            print(f"      Pair {j+1}:")
            print(f"        - IoU: {pair['iou']:.3f}")
            print(f"        - Class: {pair['class_name']}")
            print(f"        - Det1 confidence: {pair['det1_confidence']:.3f}")
            print(f"        - Det2 confidence: {pair['det2_confidence']:.3f}")
            print(f"        - Det1 kept: {pair['det1_kept']}")
            print(f"        - Det2 kept: {pair['det2_kept']}")

        # Collect for overall analysis
        all_duplicate_pairs.extend(duplicate_pairs)
        total_ious_computed += tile_pair_stats["total_ious_computed"]

    # Overall analysis
    print(f"\n📈 OVERALL IoU ANALYSIS:")
    print(f"  • Total IoUs computed across all pairs: {total_ious_computed}")
    print(f"  • Total duplicate pairs found: {len(all_duplicate_pairs)}")

    if all_duplicate_pairs:
        # IoU distribution
        ious = [pair["iou"] for pair in all_duplicate_pairs]
        print(f"  • Overall IoU range: {min(ious):.3f} to {max(ious):.3f}")
        print(f"  • Average IoU: {sum(ious)/len(ious):.3f}")

        # Class distribution
        class_counts = {}
        for pair in all_duplicate_pairs:
            class_name = pair["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"  • Duplicate pairs by class:")
        for class_name, count in class_counts.items():
            print(f"    - {class_name}: {count} pairs")

        # Decision analysis
        det1_kept = sum(1 for pair in all_duplicate_pairs if pair["det1_kept"])
        det2_kept = sum(1 for pair in all_duplicate_pairs if pair["det2_kept"])
        print(f"  • Decision distribution:")
        print(f"    - Det1 kept: {det1_kept} times")
        print(f"    - Det2 kept: {det2_kept} times")

        # Confidence analysis
        det1_confidences = [pair["det1_confidence"] for pair in all_duplicate_pairs]
        det2_confidences = [pair["det2_confidence"] for pair in all_duplicate_pairs]
        print(f"  • Confidence analysis:")
        print(
            f"    - Det1 avg confidence: {sum(det1_confidences)/len(det1_confidences):.3f}"
        )
        print(
            f"    - Det2 avg confidence: {sum(det2_confidences)/len(det2_confidences):.3f}"
        )


def save_iou_statistics(
    stats: Dict[str, Any], filename: str = "iou_statistics.json"
) -> None:
    """Save IoU statistics to a JSON file."""

    if "iou_statistics" not in stats:
        print("No IoU statistics to save")
        return

    # Create a clean version for JSON serialization
    clean_stats = {
        "overall_stats": {
            "total_image_pairs_processed": stats.get("total_image_pairs_processed", 0),
            "total_detections_removed": stats.get("total_detections_removed", 0),
            "duplicate_removal_rate": stats.get("duplicate_removal_rate", 0.0),
        },
        "iou_statistics": stats["iou_statistics"],
    }

    with open(filename, "w") as f:
        json.dump(clean_stats, f, indent=2)

    print(f"\n💾 IoU statistics saved to: {filename}")


def create_test_scenario():
    """Create a test scenario with overlapping detections."""

    # Create detections with known overlap
    det1 = Detection(
        bbox=[100, 100, 200, 200], class_id=0, class_name="person", confidence=0.9
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

    det4 = Detection(
        bbox=[350, 350, 450, 450],  # Overlapping with det3
        class_id=0,
        class_name="car",
        confidence=0.75,
    )

    return det1, det2, det3, det4


def main():
    """Main example demonstrating IoU statistics analysis."""

    print("🚁 IoU Statistics Analysis Example")
    print("=" * 60)

    # Initialize merger
    merger = GeographicMerger(merge_distance_threshold_m=50.0)

    # Create test detections
    det1, det2, det3, det4 = create_test_scenario()

    # Create drone images
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
    drone_image_2.set_predictions([det2, det4])  # Overlapping person and car

    print(f"\n📊 Initial setup:")
    print(f"  Image 1: {len(drone_image_1.predictions)} detections")
    print(f"  Image 2: {len(drone_image_2.predictions)} detections")

    # Create overlap map
    overlap_map = {
        str(drone_image_1.image_path): [str(drone_image_2.image_path)],
        str(drone_image_2.image_path): [str(drone_image_1.image_path)],
    }

    print(f"\n🔍 Running duplicate removal with IoU tracking...")

    # Run duplicate removal
    stats = merger.duplicate_removal_strategy.remove_duplicates(
        [drone_image_1, drone_image_2], overlap_map, iou_threshold=0.5
    )

    print(f"\n✅ Duplicate removal completed!")

    # Analyze IoU statistics
    analyze_iou_statistics(stats)

    # Save statistics
    save_iou_statistics(stats)

    print(f"\n💡 Key insights from IoU statistics:")
    print(f"  • Track IoU values for each duplicate pair")
    print(f"  • Analyze confidence patterns in competing detections")
    print(f"  • Understand decision-making process")
    print(f"  • Identify classes with most duplicates")
    print(f"  • Monitor IoU threshold effectiveness")


if __name__ == "__main__":
    main()
