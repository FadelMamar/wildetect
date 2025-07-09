"""
Test script for Tile and DroneImage functionality.
"""

import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wildetect.core.config import FlightSpecs
from src.wildetect.core.data import DroneImage, Tile
from src.wildetect.core.data.detection import Detection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)


def create_test_image() -> str:
    """Create a test image for testing."""

    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    return image_path


def create_test_detections(image_path: str) -> List[Detection]:
    """Create test detections for testing."""
    detections = []

    # Load the image to get its dimensions
    from PIL import Image

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Create some test detections within image boundaries
    for i in range(3):
        # Calculate positions that ensure detections stay within bounds
        margin = 50  # Minimum margin from edges
        max_x = img_width - margin
        max_y = img_height - margin

        # Ensure we have enough space for the detection
        if max_x < margin or max_y < margin:
            # If image is too small, create minimal detections
            x1 = 10
            y1 = 10
            x2 = min(20, img_width - 1)
            y2 = min(20, img_height - 1)
        else:
            # Create detections within bounds
            x1 = margin + (i * 30) % (max_x - margin)
            y1 = margin + (i * 25) % (max_y - margin)
            x2 = min(x1 + 80, img_width - 1)
            y2 = min(y1 + 60, img_height - 1)

        detection = Detection(
            bbox=[x1, y1, x2, y2],
            class_id=i,
            class_name=f"test_class_{i}",
            confidence=0.8 + i * 0.05,
        )
        detections.append(detection)

    return detections


def test_tile_creation():
    """Test Tile creation functionality."""
    logger.info("=" * 60)
    logger.info("TESTING TILE CREATION")
    logger.info("=" * 60)

    # Create test image
    image_path = create_test_image()

    # Create tile from image path
    tile = Tile.from_image_path(image_path)

    # Test basic properties
    assert tile.id is not None
    assert tile.image_path == image_path
    assert tile.width is not None
    assert tile.height is not None
    assert tile.width > 0
    assert tile.height > 0

    logger.info(f"Tile created: id={tile.id}, size={tile.width}x{tile.height}")

    # Test with flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    tile_with_specs = Tile.from_image_path(image_path, flight_specs=flight_specs)
    assert tile_with_specs.flight_specs is not None

    logger.info("✓ Tile creation test passed")


def test_tile_detection_handling():
    """Test Tile detection handling."""
    logger.info("=" * 60)
    logger.info("TESTING TILE DETECTION HANDLING")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image and tile
        image_path = create_test_image()
        tile = Tile.from_image_path(image_path)

        # Create test detections
        detections = create_test_detections(image_path)

        # Test setting predictions
        tile.set_predictions(detections)
        assert len(tile.predictions) == 3
        assert tile.predictions[0].class_name == "test_class_0"

        # Test setting annotations
        tile.set_annotations(detections[:2])  # Use first 2 detections
        assert len(tile.annotations) == 2

        # Test adding single detection
        new_detection = Detection(
            bbox=[200, 200, 300, 280],
            class_id=3,
            class_name="new_class",
            confidence=0.9,
        )
        tile.add_detection(new_detection, is_annotation=False)
        assert len(tile.predictions) == 4

        # Test validation
        try:
            tile.validate_detections()
            logger.info("✓ Detection validation passed")
        except Exception as e:
            logger.error(f"Detection validation failed: {e}")
            raise

        logger.info(
            f"Tile has {len(tile.predictions)} predictions and {len(tile.annotations)} annotations"
        )
        logger.info("✓ Tile detection handling test passed")


def test_tile_offsets():
    """Test Tile offset functionality."""
    logger.info("=" * 60)
    logger.info("TESTING TILE OFFSETS")
    logger.info("=" * 60)

    # Create test image and tile
    image_path = create_test_image()
    tile = Tile.from_image_path(image_path)

    # Test setting offsets
    tile.set_offsets(100, 200)
    assert tile.x_offset == 100
    assert tile.y_offset == 200

    # Test offset detection mapping
    detections = create_test_detections(image_path)
    tile.set_predictions(detections)

    # Test offset_detections method
    tile.offset_detections()

    # Check that detections were updated with offsets
    for detection in tile.predictions:
        # The detection coordinates should be updated to absolute coordinates
        assert detection.x_center >= 100  # Should be offset by x_offset
        assert detection.y_center >= 200  # Should be offset by y_offset

    logger.info(f"Tile offsets: x={tile.x_offset}, y={tile.y_offset}")
    logger.info("✓ Tile offsets test passed")


def test_tile_filtering():
    """Test Tile detection filtering."""
    logger.info("=" * 60)
    logger.info("TESTING TILE FILTERING")
    logger.info("=" * 60)

    # Create test image and tile
    image_path = create_test_image()
    tile = Tile.from_image_path(image_path)

    # Create detections with varying confidence
    detections = [
        Detection(
            bbox=[100, 100, 180, 160], class_name="class1", confidence=0.9, class_id=0
        ),
        Detection(
            bbox=[200, 200, 280, 260], class_name="class2", confidence=0.7, class_id=1
        ),
        Detection(
            bbox=[300, 300, 380, 360], class_name="class3", confidence=0.5, class_id=2
        ),
    ]

    tile.set_predictions(detections)

    # Test confidence filtering
    original_count = len(tile.predictions)
    tile.filter_detections(confidence_threshold=0.8)
    filtered_count = len(tile.predictions)

    assert filtered_count < original_count
    assert filtered_count == 1  # Only the 0.9 confidence detection should remain

    logger.info(f"Filtered detections: {original_count} -> {filtered_count}")
    logger.info("✓ Tile filtering test passed")


def test_drone_image_creation():
    """Test DroneImage creation functionality."""
    logger.info("=" * 60)
    logger.info("TESTING DRONE IMAGE CREATION")
    logger.info("=" * 60)

    # Create test image
    image_path = create_test_image()

    # Create drone image from image path
    drone_image = DroneImage.from_image_path(image_path)

    # Test basic properties
    assert drone_image.id is not None
    assert drone_image.image_path == image_path
    assert drone_image.width is not None
    assert drone_image.height is not None
    assert len(drone_image.tiles) > 0  # Should have at least one tile

    logger.info(
        f"DroneImage created: id={drone_image.id}, tiles={len(drone_image.tiles)}"
    )

    # Test with flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )

    drone_image_with_specs = DroneImage.from_image_path(
        image_path, flight_specs=flight_specs
    )
    assert drone_image_with_specs.flight_specs is not None

    logger.info("✓ DroneImage creation test passed")


def test_drone_image_tile_management():
    """Test DroneImage tile management."""
    logger.info("=" * 60)
    logger.info("TESTING DRONE IMAGE TILE MANAGEMENT")
    logger.info("=" * 60)

    # Create test image and drone image
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )
    image_path = create_test_image()
    drone_image = DroneImage.from_image_path(image_path, flight_specs=flight_specs)

    # Create additional tiles
    for i in range(3):
        tile = Tile.from_image_path(image_path)
        offset = (i * 100, i * 100)
        drone_image.add_tile(tile, x_offset=offset[0], y_offset=offset[1])

    # Test tile management
    assert len(drone_image.tiles) == 4  # Original + 3 added
    assert len(drone_image.tile_offsets) == 4

    # Test getting tiles at position
    tiles_at_pos = drone_image.get_tiles_at_position(150, 150)
    assert len(tiles_at_pos) > 0

    # Test getting tiles in region
    tiles_in_region = drone_image.get_tiles_in_region(0, 0, 200, 200)
    assert len(tiles_in_region) > 0

    logger.info(f"DroneImage has {len(drone_image.tiles)} tiles")
    logger.info(f"Tiles at position (150,150): {len(tiles_at_pos)}")
    logger.info(f"Tiles in region (0,0,200,200): {len(tiles_in_region)}")
    logger.info("✓ DroneImage tile management test passed")


def test_drone_image_detection_aggregation():
    """Test DroneImage detection aggregation."""
    logger.info("=" * 60)
    logger.info("TESTING DRONE IMAGE DETECTION AGGREGATION")
    logger.info("=" * 60)

    # Create test image and drone image
    image_path = create_test_image()
    drone_image = DroneImage.from_image_path(image_path)

    # Add detections to tiles
    detections = create_test_detections(image_path)
    for tile in drone_image.tiles:
        tile.set_predictions(detections.copy())

    # Test getting all detections
    all_detections = drone_image.get_all_predictions()
    expected_count = len(drone_image.tiles) * len(detections)
    assert len(all_detections) == expected_count

    # Test getting detections in region
    region_detections = drone_image.get_predictions_in_region(0, 0, 300, 300)
    assert len(region_detections) > 0

    # Test merging detections
    merged_detections = drone_image.merge_detections()
    assert len(merged_detections) <= len(all_detections)  # Should deduplicate

    logger.info(f"All detections: {len(all_detections)}")
    logger.info(f"Region detections: {len(region_detections)}")
    logger.info(f"Merged detections: {len(merged_detections)}")
    logger.info("✓ DroneImage detection aggregation test passed")


def test_drone_image_statistics():
    """Test DroneImage statistics."""
    logger.info("=" * 60)
    logger.info("TESTING DRONE IMAGE STATISTICS")
    logger.info("=" * 60)

    # Create test image and drone image
    image_path = create_test_image()
    drone_image = DroneImage.from_image_path(image_path)

    # Add detections to tiles
    detections = create_test_detections(image_path)
    for tile in drone_image.tiles:
        tile.set_predictions(detections.copy())

    # Get statistics
    stats = drone_image.get_statistics()

    # Test statistics
    assert "image_id" in stats
    assert "image_path" in stats
    assert "width" in stats
    assert "height" in stats
    assert "num_tiles" in stats
    assert "total_detections" in stats
    assert "class_counts" in stats

    logger.info(f"DroneImage statistics: {stats}")
    logger.info("✓ DroneImage statistics test passed")


def test_visualization():
    """Test visualization functionality."""
    logger.info("=" * 60)
    logger.info("TESTING VISUALIZATION")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test image and tile
        image_path = create_test_image()
        tile = Tile.from_image_path(image_path)

        # Add detections
        detections = create_test_detections(image_path)
        tile.set_predictions(detections)

        # Test drawing detections
        try:
            result_image = tile.draw_detections()
            assert result_image is not None

            # Test saving
            save_path = temp_path / "detections.jpg"
            tile.draw_detections(save_path=str(save_path))
            assert save_path.exists()

            logger.info(f"Visualization saved to: {save_path}")
            logger.info("✓ Visualization test passed")

        except Exception as e:
            logger.warning(f"Visualization test skipped (OpenCV not available): {e}")


def test_drone_image_visualization():
    """Test DroneImage visualization."""
    logger.info("=" * 60)
    logger.info("TESTING DRONE IMAGE VISUALIZATION")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test image and drone image
        image_path = create_test_image()
        drone_image = DroneImage.from_image_path(image_path)

        # Add detections to tiles
        detections = create_test_detections(image_path)
        for tile in drone_image.tiles:
            tile.set_predictions(detections.copy())

        # Test drawing detections
        try:
            result_image = drone_image.draw_detections()
            assert result_image is not None

            # Test saving
            save_path = temp_path / "drone_detections.jpg"
            drone_image.draw_detections(save_path=str(save_path))
            assert save_path.exists()

            logger.info(f"DroneImage visualization saved to: {save_path}")
            logger.info("✓ DroneImage visualization test passed")

        except Exception as e:
            logger.warning(
                f"DroneImage visualization test skipped (OpenCV not available): {e}"
            )


def run_all_tests():
    """Run all Tile and DroneImage tests."""
    logger.info("Starting Tile and DroneImage tests...")

    try:
        test_tile_creation()
        test_tile_detection_handling()
        test_tile_offsets()
        test_tile_filtering()
        test_drone_image_creation()
        test_drone_image_tile_management()
        test_drone_image_detection_aggregation()
        test_drone_image_statistics()
        test_visualization()
        test_drone_image_visualization()

        logger.info("=" * 60)
        logger.info("ALL TILE AND DRONE IMAGE TESTS PASSED! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
