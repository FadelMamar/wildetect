"""
Phase 2 Analysis Example

This script demonstrates the advanced features of WildDetect Phase 2:
- Flight path analysis and efficiency metrics
- Geographic detection merging
- Enhanced metadata and reporting
- Performance optimizations

Usage:
    python examples/phase2_analysis_example.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wildetect.core.config import FlightSpecs, LoaderConfig
from src.wildetect.core.data import (
    CensusData,
    FlightEfficiency,
    FlightPath,
    FlightPathAnalyzer,
    GeographicDataset,
    GeographicMerger,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_flight_specs():
    """Create sample flight specifications."""
    return FlightSpecs(
        sensor_height=24.0,  # mm
        focal_length=35.0,  # mm
        flight_height=180.0,  # meters
    )


def create_sample_loader_config(image_dir: str):
    """Create sample loader configuration."""
    return LoaderConfig(
        image_dir=image_dir,
        tile_size=640,
        overlap=0.2,
        batch_size=4,
        recursive=True,
        flight_specs=create_sample_flight_specs(),
        extract_gps=True,
    )


def demonstrate_flight_path_analysis(census_data: CensusData):
    """Demonstrate flight path analysis capabilities."""
    logger.info("=" * 60)
    logger.info("FLIGHT PATH ANALYSIS DEMONSTRATION")
    logger.info("=" * 60)

    # Analyze flight path
    flight_path = census_data.analyze_flight_path()

    if flight_path and flight_path.coordinates:
        logger.info(f"Flight path analysis completed successfully!")
        logger.info(f"  Total waypoints: {len(flight_path.coordinates)}")
        logger.info(f"  Total images: {len(flight_path.image_paths)}")

        # Display flight metrics
        if flight_path.metadata:
            logger.info("Flight metrics:")
            logger.info(
                f"  Total distance: {flight_path.metadata.get('total_distance_km', 0):.2f} km"
            )
            logger.info(
                f"  Average altitude: {flight_path.metadata.get('average_altitude_m', 0):.1f} m"
            )
            logger.info(
                f"  Max altitude: {flight_path.metadata.get('max_altitude_m', 0):.1f} m"
            )
            logger.info(
                f"  Min altitude: {flight_path.metadata.get('min_altitude_m', 0):.1f} m"
            )

        # Calculate flight efficiency
        efficiency = census_data.calculate_flight_efficiency()
        if efficiency:
            logger.info("Flight efficiency metrics:")
            logger.info(f"  Coverage efficiency: {efficiency.coverage_efficiency:.2f}")
            logger.info(f"  Overlap percentage: {efficiency.overlap_percentage:.1%}")
            logger.info(
                f"  Image density: {efficiency.image_density_per_sqkm:.1f} images/sq km"
            )
            logger.info(
                f"  Flight duration: {efficiency.flight_duration_hours:.1f} hours"
            )
    else:
        logger.warning("No GPS data available for flight path analysis")


def demonstrate_overlap_detection(census_data: CensusData):
    """Demonstrate overlap detection capabilities."""
    logger.info("=" * 60)
    logger.info("OVERLAP DETECTION DEMONSTRATION")
    logger.info("=" * 60)

    # Detect overlapping regions
    overlapping_regions = census_data.detect_overlapping_regions(overlap_threshold=0.1)

    if overlapping_regions:
        logger.info(f"Detected {len(overlapping_regions)} overlapping regions")

        for i, region in enumerate(overlapping_regions[:5]):  # Show first 5
            logger.info(f"Region {i+1}:")
            logger.info(f"  Center: ({region.center_lat:.6f}, {region.center_lon:.6f})")
            logger.info(f"  Overlap: {region.overlap_percentage:.1%}")
            logger.info(f"  Area: {region.overlap_area_sqm:.0f} sq m")
            logger.info(f"  Images: {len(region.image_paths)}")
    else:
        logger.info("No overlapping regions detected")


def demonstrate_geographic_merging(census_data: CensusData):
    """Demonstrate geographic detection merging capabilities."""
    logger.info("=" * 60)
    logger.info("GEOGRAPHIC MERGING DEMONSTRATION")
    logger.info("=" * 60)

    # Merge detections geographically
    geographic_dataset = census_data.merge_detections_geographically(
        merge_distance_threshold_m=50.0
    )

    if geographic_dataset and geographic_dataset.merged_detections:
        logger.info(f"Geographic merging completed successfully!")
        logger.info(
            f"  Total merged detections: {len(geographic_dataset.merged_detections)}"
        )
        logger.info(
            f"  Total area covered: {geographic_dataset.total_area_covered_sqkm:.2f} sq km"
        )
        logger.info(
            f"  Detection density: {geographic_dataset.detection_density_per_sqkm:.1f} detections/sq km"
        )

        # Show geographic bounds
        bounds = geographic_dataset.geographic_bounds
        logger.info("Geographic bounds:")
        logger.info(f"  Latitude: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}")
        logger.info(f"  Longitude: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}")

        # Show sample merged detections
        logger.info("Sample merged detections:")
        for i, detection in enumerate(geographic_dataset.merged_detections[:3]):
            logger.info(f"  Detection {i+1}:")
            logger.info(f"    Class: {detection.class_name}")
            logger.info(
                f"    Position: ({detection.center_lat:.6f}, {detection.center_lon:.6f})"
            )
            logger.info(f"    Confidence: {detection.confidence:.2f}")
            logger.info(f"    Source images: {len(detection.source_images)}")
            logger.info(f"    Detection count: {detection.detection_count}")
    else:
        logger.warning("No detections available for geographic merging")


def demonstrate_enhanced_statistics(census_data: CensusData):
    """Demonstrate enhanced statistics and reporting."""
    logger.info("=" * 60)
    logger.info("ENHANCED STATISTICS DEMONSTRATION")
    logger.info("=" * 60)

    # Get enhanced campaign statistics
    stats = census_data.get_enhanced_campaign_statistics()

    logger.info("Enhanced campaign statistics:")
    logger.info(f"  Campaign ID: {stats['campaign_id']}")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Total image paths: {stats['total_image_paths']}")

    # Flight analysis stats
    if "flight_analysis" in stats:
        flight_stats = stats["flight_analysis"]
        logger.info("Flight analysis:")
        logger.info(f"  Total waypoints: {flight_stats['total_waypoints']}")
        logger.info(f"  Total distance: {flight_stats['total_distance_km']:.2f} km")
        logger.info(f"  Images with GPS: {flight_stats['num_images_with_gps']}")

    # Flight efficiency stats
    if "flight_efficiency" in stats:
        efficiency_stats = stats["flight_efficiency"]
        logger.info("Flight efficiency:")
        logger.info(
            f"  Coverage efficiency: {efficiency_stats['coverage_efficiency']:.2f}"
        )
        logger.info(
            f"  Overlap percentage: {efficiency_stats['overlap_percentage']:.1%}"
        )
        logger.info(
            f"  Image density: {efficiency_stats['image_density_per_sqkm']:.1f} images/sq km"
        )

    # Geographic merging stats
    if "geographic_merging" in stats:
        geo_stats = stats["geographic_merging"]
        logger.info("Geographic merging:")
        logger.info(f"  Merged detections: {geo_stats['total_merged_detections']}")
        logger.info(f"  Area covered: {geo_stats['total_area_covered_sqkm']:.2f} sq km")
        logger.info(
            f"  Detection density: {geo_stats['detection_density_per_sqkm']:.1f} detections/sq km"
        )


def demonstrate_complete_phase2_analysis(census_data: CensusData):
    """Demonstrate complete Phase 2 analysis workflow."""
    logger.info("=" * 60)
    logger.info("COMPLETE PHASE 2 ANALYSIS DEMONSTRATION")
    logger.info("=" * 60)

    # Run complete Phase 2 analysis
    results = census_data.run_complete_phase2_analysis(
        overlap_threshold=0.1, merge_distance_threshold_m=50.0
    )

    logger.info("Complete Phase 2 analysis results:")
    logger.info(f"  Campaign ID: {results['campaign_id']}")

    phase2_results = results["phase2_analysis"]

    # Flight path results
    if "flight_path" in phase2_results:
        flight_path = phase2_results["flight_path"]
        logger.info(f"  Flight path waypoints: {flight_path['total_waypoints']}")

    # Flight efficiency results
    if "flight_efficiency" in phase2_results:
        efficiency = phase2_results["flight_efficiency"]
        logger.info(f"  Total distance: {efficiency['total_distance_km']:.2f} km")
        logger.info(
            f"  Area covered: {efficiency['total_area_covered_sqkm']:.2f} sq km"
        )
        logger.info(f"  Coverage efficiency: {efficiency['coverage_efficiency']:.2f}")

    # Overlapping regions results
    if "overlapping_regions" in phase2_results:
        overlap = phase2_results["overlapping_regions"]
        logger.info(f"  Overlapping regions: {overlap['count']}")

    # Geographic merging results
    if "geographic_merging" in phase2_results:
        geo_merging = phase2_results["geographic_merging"]
        logger.info(f"  Merged detections: {geo_merging['total_merged_detections']}")
        logger.info(
            f"  Detection density: {geo_merging['detection_density_per_sqkm']:.1f} detections/sq km"
        )


def export_results(census_data: CensusData, output_dir: str):
    """Export analysis results to files."""
    logger.info("=" * 60)
    logger.info("EXPORTING RESULTS")
    logger.info("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export detection report
    detection_report_path = output_path / "detection_report.json"
    census_data.export_detection_report(str(detection_report_path))

    # Export geographic dataset if available
    if census_data.geographic_dataset:
        geo_dataset_path = output_path / "geographic_dataset.json"
        census_data.export_geographic_dataset(str(geo_dataset_path))

    # Export enhanced statistics
    stats = census_data.get_enhanced_campaign_statistics()
    import json

    stats_path = output_path / "enhanced_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"Results exported to: {output_path}")
    logger.info(f"  Detection report: {detection_report_path}")
    if census_data.geographic_dataset:
        logger.info(f"  Geographic dataset: {geo_dataset_path}")
    logger.info(f"  Enhanced statistics: {stats_path}")


def main():
    """Main demonstration function."""
    logger.info("Starting Phase 2 Analysis Demonstration")
    logger.info("=" * 80)

    # Configuration
    image_dir = "data/sample_images"  # Update this path to your image directory
    campaign_id = "phase2_demo_campaign"

    # Check if image directory exists
    if not Path(image_dir).exists():
        logger.error(f"Image directory not found: {image_dir}")
        logger.info(
            "Please update the image_dir variable to point to your image directory"
        )
        return

    try:
        # Create loader configuration
        loader_config = create_sample_loader_config(image_dir)

        # Create campaign metadata
        campaign_metadata = {
            "flight_specs": create_sample_flight_specs(),
            "pilot_info": {"name": "Demo Pilot", "experience": "5 years"},
            "weather_conditions": {
                "temperature": 25,
                "wind_speed": 5,
                "visibility": "good",
            },
            "mission_objectives": ["wildlife_survey", "habitat_mapping"],
            "target_species": ["elephant", "giraffe", "zebra", "lion"],
        }

        # Initialize CensusData
        logger.info(f"Initializing CensusData for campaign: {campaign_id}")
        census_data = CensusData(
            campaign_id=campaign_id,
            loading_config=loader_config,
            metadata=campaign_metadata,
        )

        # Add images from directory
        logger.info(f"Adding images from directory: {image_dir}")
        census_data.add_images_from_directory(image_dir, recursive=True)

        # Create drone images
        logger.info("Creating DroneImage instances...")
        census_data.create_drone_images()

        if not census_data.drone_images:
            logger.warning("No drone images created. Check your image directory.")
            return

        logger.info(f"Created {len(census_data.drone_images)} DroneImage instances")

        # Demonstrate Phase 2 features
        demonstrate_flight_path_analysis(census_data)
        demonstrate_overlap_detection(census_data)
        demonstrate_geographic_merging(census_data)
        demonstrate_enhanced_statistics(census_data)
        demonstrate_complete_phase2_analysis(census_data)

        # Export results
        export_results(census_data, "output/phase2_analysis")

        logger.info("=" * 80)
        logger.info("Phase 2 Analysis Demonstration Completed Successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during Phase 2 analysis demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
