"""
Test script for CensusData functionality.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wildetect.core.config import FlightSpecs, LoaderConfig
from src.wildetect.core.data import CensusData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_images(test_dir: Path, num_images: int = 5) -> List[str]:
    """Create test images for testing."""
    import numpy as np
    from PIL import Image

    test_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for i in range(num_images):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Save with different formats
        if i % 2 == 0:
            img_path = test_dir / f"test_image_{i}.jpg"
        else:
            img_path = test_dir / f"test_image_{i}.png"

        img.save(img_path)
        image_paths.append(str(img_path))

    logger.info(f"Created {len(image_paths)} test images in {test_dir}")
    return image_paths


def test_census_data_initialization():
    """Test CensusData initialization."""
    logger.info("=" * 60)
    logger.info("TESTING CENSUS DATA INITIALIZATION")
    logger.info("=" * 60)

    # Create loader config
    loader_config = LoaderConfig(
        tile_size=640,
        overlap=0.2,
        batch_size=4,
        flight_specs=FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        ),
    )

    # Create campaign metadata
    campaign_metadata = {
        "pilot_info": {"name": "Test Pilot", "experience": "3 years"},
        "weather_conditions": {"temperature": 25, "wind_speed": 5},
        "mission_objectives": ["wildlife_survey", "habitat_mapping"],
        "target_species": ["elephant", "giraffe", "zebra"],
    }

    # Initialize CensusData
    census_data = CensusData(
        campaign_id="test_campaign_2024",
        loading_config=loader_config,
        metadata=campaign_metadata,
    )

    # Test basic properties
    assert census_data.campaign_id == "test_campaign_2024"
    assert census_data.loading_config is not None
    assert census_data.metadata is not None
    assert len(census_data.drone_images) == 0
    assert len(census_data.image_paths) == 0

    logger.info(f"Campaign ID: {census_data.campaign_id}")
    logger.info(f"Loading config: tile_size={census_data.loading_config.tile_size}")
    logger.info(f"Metadata keys: {list(census_data.metadata.keys())}")
    logger.info("✓ CensusData initialization test passed")


def test_add_images_from_paths():
    """Test adding images from paths."""
    logger.info("=" * 60)
    logger.info("TESTING ADD IMAGES FROM PATHS")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=3)

        # Create CensusData
        loader_config = LoaderConfig(tile_size=640, overlap=0.2)
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images from paths
        census_data.add_images_from_paths(image_paths)

        # Test that images were added
        assert len(census_data.image_paths) == 3
        logger.info(f"Added {len(census_data.image_paths)} image paths")

        # Test with invalid paths
        invalid_paths = ["/nonexistent/image1.jpg", "/nonexistent/image2.png"]
        census_data.add_images_from_paths(invalid_paths)

        # Should still have only valid paths
        assert len(census_data.image_paths) == 3
        logger.info("✓ Add images from paths test passed")


def test_add_images_from_directory():
    """Test adding images from directory."""
    logger.info("=" * 60)
    logger.info("TESTING ADD IMAGES FROM DIRECTORY")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=4)

        # Create CensusData
        loader_config = LoaderConfig(tile_size=640, overlap=0.2)
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images from directory
        census_data.add_images_from_directory(temp_dir)

        # Test that images were added
        assert len(census_data.image_paths) == 4
        logger.info(f"Added {len(census_data.image_paths)} images from directory")

        # Test with empty directory
        empty_dir = temp_path / "empty"
        empty_dir.mkdir()
        census_data.add_images_from_directory(str(empty_dir))

        # Should still have only the original images
        assert len(census_data.image_paths) == 4
        logger.info("✓ Add images from directory test passed")


def test_create_drone_images():
    """Test creating drone images."""
    logger.info("=" * 60)
    logger.info("TESTING CREATE DRONE IMAGES")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=3)

        # Create CensusData
        loader_config = LoaderConfig(
            tile_size=320,  # Smaller tiles for testing
            overlap=0.2,
            flight_specs=FlightSpecs(
                sensor_height=24.0, focal_length=35.0, flight_height=180.0
            ),
        )
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images
        census_data.add_images_from_paths(image_paths)

        # Create drone images
        census_data.create_drone_images()

        # Test that drone images were created
        assert len(census_data.drone_images) == 3
        logger.info(f"Created {len(census_data.drone_images)} drone images")

        # Test drone image properties
        for i, drone_image in enumerate(census_data.drone_images):
            logger.info(
                f"DroneImage {i}: id={drone_image.id}, tiles={len(drone_image.tiles)}"
            )
            assert drone_image.id is not None
            assert len(drone_image.tiles) > 0

        logger.info("✓ Create drone images test passed")


def test_campaign_statistics():
    """Test campaign statistics."""
    logger.info("=" * 60)
    logger.info("TESTING CAMPAIGN STATISTICS")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=2)

        # Create CensusData
        loader_config = LoaderConfig(tile_size=320, overlap=0.2)
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images and create drone images
        census_data.add_images_from_paths(image_paths)
        census_data.create_drone_images()

        # Get statistics
        stats = census_data.get_campaign_statistics()

        # Test statistics
        assert stats["campaign_id"] == "test_campaign"
        assert stats["total_images"] == 2
        assert stats["total_image_paths"] == 2
        assert "tile_configuration" in stats
        assert "metadata" in stats

        logger.info(f"Campaign statistics: {stats}")
        logger.info("✓ Campaign statistics test passed")


def test_get_drone_image_by_path():
    """Test getting drone image by path."""
    logger.info("=" * 60)
    logger.info("TESTING GET DRONE IMAGE BY PATH")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=2)

        # Create CensusData
        loader_config = LoaderConfig(tile_size=320, overlap=0.2)
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images and create drone images
        census_data.add_images_from_paths(image_paths)
        census_data.create_drone_images()

        # Test getting drone image by path
        test_path = image_paths[0]
        drone_image = census_data.get_drone_image_by_path(test_path)

        assert drone_image is not None
        assert drone_image.image_path == test_path

        # Test with non-existent path
        non_existent = census_data.get_drone_image_by_path("/nonexistent/path.jpg")
        assert non_existent is None

        logger.info(f"Found drone image for path: {test_path}")
        logger.info("✓ Get drone image by path test passed")


def test_get_all_detections():
    """Test getting all detections."""
    logger.info("=" * 60)
    logger.info("TESTING GET ALL DETECTIONS")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=2)

        # Create CensusData
        loader_config = LoaderConfig(tile_size=320, overlap=0.2)
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images and create drone images
        census_data.add_images_from_paths(image_paths)
        census_data.create_drone_images()

        # Get all detections (should be empty initially)
        all_detections = census_data.get_all_detections()

        assert len(all_detections) == 0  # No detections yet

        logger.info(f"All detections: {len(all_detections)}")
        logger.info("✓ Get all detections test passed")


def test_export_detection_report():
    """Test exporting detection report."""
    logger.info("=" * 60)
    logger.info("TESTING EXPORT DETECTION REPORT")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        image_paths = create_test_images(temp_path, num_images=2)

        # Create CensusData
        loader_config = LoaderConfig(tile_size=320, overlap=0.2)
        census_data = CensusData(
            campaign_id="test_campaign", loading_config=loader_config
        )

        # Add images and create drone images
        census_data.add_images_from_paths(image_paths)
        census_data.create_drone_images()

        # Export detection report
        report_path = temp_path / "detection_report.json"
        census_data.export_detection_report(str(report_path))

        # Test that report was created
        assert report_path.exists()
        logger.info(f"Detection report exported to: {report_path}")

        # Test with no detection results
        census_data.detection_results = None
        empty_report_path = temp_path / "empty_report.json"
        census_data.export_detection_report(str(empty_report_path))

        # Should not create file when no results
        assert not empty_report_path.exists()

        logger.info("✓ Export detection report test passed")


def test_campaign_metadata():
    """Test campaign metadata functionality."""
    logger.info("=" * 60)
    logger.info("TESTING CAMPAIGN METADATA")
    logger.info("=" * 60)

    # Create CensusData with metadata
    loader_config = LoaderConfig(tile_size=640, overlap=0.2)
    campaign_metadata = {
        "pilot_info": {"name": "Test Pilot", "experience": "5 years"},
        "weather_conditions": {
            "temperature": 25,
            "wind_speed": 5,
            "visibility": "good",
        },
        "mission_objectives": ["wildlife_survey", "habitat_mapping"],
        "target_species": ["elephant", "giraffe", "zebra", "lion"],
        "flight_specs": FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        ),
    }

    census_data = CensusData(
        campaign_id="metadata_test_campaign",
        loading_config=loader_config,
        metadata=campaign_metadata,
    )

    # Test metadata access
    assert census_data.metadata["pilot_info"]["name"] == "Test Pilot"
    assert len(census_data.metadata["target_species"]) == 4
    assert census_data.metadata["weather_conditions"]["temperature"] == 25

    logger.info(f"Campaign metadata: {census_data.metadata}")
    logger.info("✓ Campaign metadata test passed")


def run_all_tests():
    """Run all CensusData tests."""
    logger.info("Starting CensusData tests...")

    try:
        test_census_data_initialization()
        test_add_images_from_paths()
        test_add_images_from_directory()
        test_create_drone_images()
        test_campaign_statistics()
        test_get_drone_image_by_path()
        test_get_all_detections()
        test_export_detection_report()
        test_campaign_metadata()

        logger.info("=" * 60)
        logger.info("ALL CENSUS DATA TESTS PASSED! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
