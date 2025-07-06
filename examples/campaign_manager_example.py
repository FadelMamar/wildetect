"""
Example demonstrating the CampaignManager for complete wildlife detection campaigns.

This example shows how to use the new modular system to run complete campaigns
from data ingestion through detection, analysis, and reporting.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.wildetect.core.campaign_manager import CampaignConfig, CampaignManager
from src.wildetect.core.config import FlightSpecs, LoaderConfig, PredictionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_campaign_config(
    campaign_id: str, model_path: str, image_paths: List[str]
) -> CampaignConfig:
    """Create a campaign configuration.

    Args:
        campaign_id: Unique identifier for the campaign
        model_path: Path to the detection model
        image_paths: List of image paths to process

    Returns:
        CampaignConfig: Configuration for the campaign
    """
    # Data loading configuration
    loader_config = LoaderConfig(
        tile_size=640,
        overlap=0.2,
        batch_size=4,
        flight_specs=FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        ),
    )

    # Detection configuration
    prediction_config = PredictionConfig(
        model_path=model_path,
        model_type="yolo",
        confidence_threshold=0.25,
        device="cpu",
        tilesize=640,
        cls_imgsz=224,
        verbose=True,
    )

    # Campaign metadata
    metadata = {
        "pilot_info": {"name": "John Doe", "experience": "5 years"},
        "weather_conditions": {
            "temperature": 25,
            "wind_speed": 5,
            "visibility": "good",
        },
        "mission_objectives": ["wildlife_survey", "habitat_mapping"],
        "target_species": ["elephant", "giraffe", "zebra", "lion"],
        "flight_parameters": {"altitude": 100, "speed": 15, "overlap": 0.7},
        "equipment_info": {"drone": "DJI Phantom 4", "camera": "20MP RGB"},
    }

    return CampaignConfig(
        campaign_id=campaign_id,
        loader_config=loader_config,
        prediction_config=prediction_config,
        metadata=metadata,
        fiftyone_dataset_name=f"campaign_{campaign_id}",
    )


def run_basic_campaign():
    """Run a basic campaign with minimal configuration."""
    logger.info("=" * 60)
    logger.info("RUNNING BASIC CAMPAIGN")
    logger.info("=" * 60)

    # Example image paths (replace with actual paths)
    image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", "/path/to/image3.jpg"]

    # Model path (replace with actual model path)
    model_path = "/path/to/model.pt"

    # Create campaign configuration
    config = create_campaign_config("basic_campaign_2024", model_path, image_paths)

    # Initialize campaign manager
    campaign_manager = CampaignManager(config)

    try:
        # Run complete campaign
        results = campaign_manager.run_complete_campaign(
            image_paths=image_paths,
            output_dir="./campaign_results",
            tile_size=640,
            overlap=0.2,
            run_flight_analysis=True,
            run_geographic_merging=True,
            create_visualization=True,
            export_to_fiftyone=False,
        )

        # Display results
        print(f"\n📊 Campaign Results:")
        print(f"  Campaign ID: {results['campaign_id']}")
        print(f"  Total Images: {results['detection_results'].total_images}")
        print(f"  Total Detections: {results['detection_results'].total_detections}")
        print(f"  Processing Time: {results['detection_results'].processing_time:.2f}s")

        if results["detection_results"].detection_by_class:
            print(f"  Species Detected:")
            for species, count in results[
                "detection_results"
            ].detection_by_class.items():
                print(f"    {species}: {count}")

        if results["flight_efficiency"]:
            print(f"  Flight Efficiency:")
            print(
                f"    Distance: {results['flight_efficiency'].total_distance_km:.2f} km"
            )
            print(
                f"    Area Covered: {results['flight_efficiency'].total_area_covered_sqkm:.2f} sq km"
            )

        print(f"  Visualization: {results['visualization_path']}")

    except Exception as e:
        logger.error(f"Campaign failed: {e}")
        raise


def run_advanced_campaign():
    """Run an advanced campaign with all features enabled."""
    logger.info("=" * 60)
    logger.info("RUNNING ADVANCED CAMPAIGN")
    logger.info("=" * 60)

    # Example image paths (replace with actual paths)
    image_paths = [
        "/path/to/advanced/image1.jpg",
        "/path/to/advanced/image2.jpg",
        "/path/to/advanced/image3.jpg",
        "/path/to/advanced/image4.jpg",
        "/path/to/advanced/image5.jpg",
    ]

    # Model path (replace with actual model path)
    model_path = "/path/to/advanced_model.pt"

    # Create campaign configuration
    config = create_campaign_config("advanced_campaign_2024", model_path, image_paths)

    # Initialize campaign manager
    campaign_manager = CampaignManager(config)

    try:
        # Step-by-step campaign execution
        print("Step 1: Adding images...")
        campaign_manager.add_images_from_paths(image_paths)

        print("Step 2: Preparing data...")
        campaign_manager.prepare_data(tile_size=640, overlap=0.2)

        print("Step 3: Running detection...")
        detection_results = campaign_manager.run_detection(
            save_results=True, output_dir="./advanced_campaign_results"
        )

        print("Step 4: Analyzing flight path...")
        flight_path = campaign_manager.analyze_flight_path()

        if flight_path:
            print("Step 5: Calculating flight efficiency...")
            flight_efficiency = campaign_manager.calculate_flight_efficiency()

        print("Step 6: Merging detections geographically...")
        merged_images = campaign_manager.merge_detections_geographically(
            iou_threshold=0.8
        )

        print("Step 7: Creating visualization...")
        visualization_path = campaign_manager.create_geographic_visualization(
            "./advanced_campaign_results/visualization.html"
        )

        print("Step 8: Exporting to FiftyOne...")
        campaign_manager.export_to_fiftyone()

        print("Step 9: Exporting final report...")
        campaign_manager.export_detection_report(
            "./advanced_campaign_results/final_report.json"
        )

        # Get comprehensive statistics
        stats = campaign_manager.get_campaign_statistics()

        print(f"\n📊 Advanced Campaign Results:")
        print(f"  Campaign ID: {stats['campaign_id']}")
        print(f"  Total Images: {stats['total_images']}")

        if "detection_results" in stats:
            detection_stats = stats["detection_results"]
            print(f"  Total Detections: {detection_stats['total_detections']}")
            print(f"  Processing Time: {detection_stats['processing_time']:.2f}s")

        if "flight_analysis" in stats:
            flight_stats = stats["flight_analysis"]
            print(f"  Images with GPS: {flight_stats['num_images_with_gps']}")
            print(f"  Total Waypoints: {flight_stats['total_waypoints']}")
            print(f"  Total Distance: {flight_stats['total_distance_km']:.2f} km")

        if "flight_efficiency" in stats:
            efficiency_stats = stats["flight_efficiency"]
            print(
                f"  Coverage Efficiency: {efficiency_stats['coverage_efficiency']:.2f}"
            )
            print(f"  Overlap Percentage: {efficiency_stats['overlap_percentage']:.1%}")

        print(f"  Visualization: {visualization_path}")

    except Exception as e:
        logger.error(f"Advanced campaign failed: {e}")
        raise


def demonstrate_modular_usage():
    """Demonstrate modular usage of the campaign manager."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING MODULAR USAGE")
    logger.info("=" * 60)

    # Example image paths (replace with actual paths)
    image_paths = ["/path/to/modular/image1.jpg", "/path/to/modular/image2.jpg"]

    # Model path (replace with actual model path)
    model_path = "/path/to/modular_model.pt"

    # Create campaign configuration
    config = create_campaign_config("modular_campaign_2024", model_path, image_paths)

    # Initialize campaign manager
    campaign_manager = CampaignManager(config)

    try:
        # Add images
        campaign_manager.add_images_from_paths(image_paths)
        print(f"✓ Added {len(image_paths)} images")

        # Prepare data
        campaign_manager.prepare_data(tile_size=640, overlap=0.2)
        print(
            f"✓ Prepared {len(campaign_manager.census_manager.drone_images)} drone images"
        )

        # Run detection only
        detection_results = campaign_manager.run_detection(save_results=False)
        print(f"✓ Detection completed: {detection_results.total_detections} detections")

        # Get all detections
        all_detections = campaign_manager.get_all_detections()
        print(f"✓ Retrieved {len(all_detections)} total detections")

        # Get campaign statistics
        stats = campaign_manager.get_campaign_statistics()
        print(f"✓ Generated comprehensive statistics")

        # Export detection report
        campaign_manager.export_detection_report("./modular_campaign_report.json")
        print(f"✓ Exported detection report")

        print(f"\n📊 Modular Campaign Summary:")
        print(f"  Campaign ID: {stats['campaign_id']}")
        print(f"  Total Images: {stats['total_images']}")
        print(f"  Total Detections: {detection_results.total_detections}")
        print(f"  Processing Time: {detection_results.processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Modular campaign failed: {e}")
        raise


def main():
    """Run all campaign examples."""
    print("🚁 WildDetect Campaign Manager Examples")
    print("=" * 60)

    try:
        # Run basic campaign
        run_basic_campaign()

        print("\n" + "=" * 60)

        # Run advanced campaign
        run_advanced_campaign()

        print("\n" + "=" * 60)

        # Demonstrate modular usage
        demonstrate_modular_usage()

        print("\n" + "=" * 60)
        print("✅ All campaign examples completed successfully!")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        print(f"❌ Example execution failed: {e}")


if __name__ == "__main__":
    main()
