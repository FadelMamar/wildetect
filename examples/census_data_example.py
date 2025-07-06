"""
Example demonstrating the CensusData functionality for drone image analysis campaigns.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
from datetime import datetime

from src.wildetect.core.data import CampaignMetadata, CensusData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_census_data():
    """Demonstrate the CensusData functionality."""

    # Create campaign metadata
    metadata = CampaignMetadata(
        campaign_id="flight_2024_01",
        flight_date=datetime.now(),
        pilot_info={"name": "John Doe", "license": "PPL"},
        weather_conditions={"visibility": "good", "wind_speed": "5 knots"},
        mission_objectives=["wildlife survey", "habitat mapping"],
        target_species=["deer", "birds", "small mammals"],
        flight_parameters={"altitude": 100, "speed": 15, "overlap": 0.7},
        equipment_info={"drone": "DJI Phantom 4", "camera": "20MP RGB"},
    )

    # Initialize CensusData
    census = CensusData(
        campaign_id="flight_2024_01",
        metadata={
            "campaign_metadata": metadata,
            "flight_specs": None,  # You would add actual flight specs here
        },
    )

    print("=== CensusData Campaign Setup ===")
    print(f"Campaign ID: {census.campaign_id}")
    print(f"Metadata: {census.metadata}")

    # Simulate adding images (in practice, you'd have actual image paths)
    print("\n=== Adding Images ===")

    # Example image paths (replace with actual paths)
    example_image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.jpg",
    ]

    # Add images from paths
    census.add_images_from_paths(example_image_paths)
    print(f"Added {len(census.image_paths)} image paths")

    # Add images from directory (commented out for demo)
    # census.add_images_from_directory("/path/to/images", recursive=True)

    # Create DroneImages
    print("\n=== Creating DroneImages ===")
    census.create_drone_images(tile_size=640, overlap=0.2)
    print(f"Created {len(census.drone_images)} DroneImages")

    # Get campaign statistics
    print("\n=== Campaign Statistics ===")
    stats = census.get_campaign_statistics()
    print(f"Total images: {stats['total_images']}")
    print(f"Tile configuration: {stats['tile_configuration']}")

    # Simulate detection campaign
    print("\n=== Detection Campaign ===")

    # Mock detector function
    def mock_detector(image, **kwargs):
        """Mock detector that returns empty detections."""
        return []

    # Run detection campaign
    results = census.run_detection_campaign(mock_detector)

    if results:
        print(f"Detection Results:")
        print(f"  Total images processed: {results.total_images}")
        print(f"  Total detections: {results.total_detections}")
        print(f"  Processing time: {results.processing_time:.2f}s")
        print(f"  Detections by class: {results.detection_by_class}")
        print(f"  Geographic coverage: {results.geographic_coverage}")

    # Export report
    print("\n=== Exporting Report ===")
    census.export_detection_report("campaign_report.json")
    print("Report exported to campaign_report.json")

    # Demonstrate DroneImage access
    print("\n=== DroneImage Access ===")
    if census.drone_images:
        drone_image = census.drone_images[0]
        print(f"First DroneImage: {drone_image.image_path}")
        print(f"Number of tiles: {len(drone_image.tiles)}")
        print(f"GPS coordinates: ({drone_image.latitude}, {drone_image.longitude})")

    print("\n=== CensusData Features ===")
    print("✓ Campaign management with metadata")
    print("✓ Image ingestion from paths and directories")
    print("✓ DroneImage creation with tiling")
    print("✓ Detection campaign orchestration")
    print("✓ Geographic coverage analysis")
    print("✓ Comprehensive reporting")
    print("✓ Statistics and analytics")


def demonstrate_advanced_features():
    """Demonstrate advanced CensusData features."""

    print("\n=== Advanced Features ===")
    print("1. Flight Path Analysis:")
    print("   - Analyze GPS patterns across images")
    print("   - Detect overlapping regions")
    print("   - Calculate coverage efficiency")

    print("\n2. Geographic Merging:")
    print("   - Merge detections in overlapping areas")
    print("   - Handle GPS uncertainty")
    print("   - Create coverage maps")

    print("\n3. Performance Optimization:")
    print("   - Batch processing")
    print("   - Parallel detection")
    print("   - Memory management")

    print("\n4. Quality Control:")
    print("   - Filter false positives")
    print("   - Confidence thresholding")
    print("   - Manual review integration")


if __name__ == "__main__":
    print("=== CensusData Demonstration ===\n")

    demonstrate_census_data()
    demonstrate_advanced_features()

    print("\n=== Usage Example ===")
    print(
        """
# Simple usage
census = CensusData("my_campaign")
census.add_images_from_directory("/path/to/images")
census.create_drone_images(tile_size=640, overlap=0.2)
results = census.run_detection_campaign(detector)
census.export_detection_report("results.json")

# Advanced usage
stats = census.get_campaign_statistics()
drone_image = census.get_drone_image_by_path("/path/to/image.jpg")
all_detections = census.get_all_detections()
    """
    )
