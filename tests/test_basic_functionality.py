"""
Basic functionality test for wildetect core components.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all core modules can be imported."""
    logger.info("=" * 60)
    logger.info("TESTING IMPORTS")
    logger.info("=" * 60)

    try:
        # Test core imports
        from src.wildetect.core.config import FlightSpecs, LoaderConfig
        from src.wildetect.core.data import DroneImage, Tile
        from src.wildetect.core.data.dataset import CensusData
        from src.wildetect.core.data.detection import Detection
        from src.wildetect.core.data.loader import DataLoader, TileDataset
        from src.wildetect.core.data.utils import TileUtils, get_images_paths

        logger.info("✓ All core imports successful")

        # Test that classes can be instantiated
        config = LoaderConfig(tile_size=640, overlap=0.2)
        assert config.tile_size == 640
        assert config.overlap == 0.2

        flight_specs = FlightSpecs(
            sensor_height=24.0, focal_length=35.0, flight_height=180.0
        )
        assert flight_specs.sensor_height == 24.0

        logger.info("✓ Config classes instantiated successfully")

    except ImportError as e:
        logger.error(f"Import failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during import test: {e}")
        raise


def test_basic_operations():
    """Test basic operations with core classes."""
    logger.info("=" * 60)
    logger.info("TESTING BASIC OPERATIONS")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a test image
        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        image_path = temp_path / "test_image.jpg"
        img.save(image_path)

        # Test Tile creation
        from src.wildetect.core.data import Tile

        tile = Tile.from_image_path(str(image_path))
        assert tile.id is not None
        assert tile.width == 640
        assert tile.height == 480

        # Test Detection creation
        from src.wildetect.core.data.detection import Detection

        detection = Detection(
            bbox=[100, 100, 180, 160],  # [x1, y1, x2, y2]
            class_id=0,
            class_name="test_class",
            confidence=0.8,
        )
        assert detection.class_name == "test_class"
        assert detection.confidence == 0.8

        # Test DroneImage creation
        from src.wildetect.core.data import DroneImage

        drone_image = DroneImage.from_image_path(str(image_path))
        assert drone_image.id is not None
        assert len(drone_image.tiles) > 0

        logger.info("✓ Basic operations test passed")


def test_utils_functions():
    """Test utility functions."""
    logger.info("=" * 60)
    logger.info("TESTING UTILITY FUNCTIONS")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images
        import numpy as np
        from PIL import Image

        for i in range(3):
            img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = temp_path / f"test_image_{i}.jpg"
            img.save(img_path)

        # Test get_images_paths
        from src.wildetect.core.data.utils import get_images_paths

        image_paths = get_images_paths(temp_dir)
        assert len(image_paths) == 3

        # Test TileUtils validation
        from src.wildetect.core.data.utils import TileUtils

        is_valid = TileUtils.validate_patch_parameters(
            image_shape=(3, 480, 640), patch_size=320, stride=256
        )
        assert is_valid is True

        logger.info("✓ Utility functions test passed")


def test_config_validation():
    """Test configuration validation."""
    logger.info("=" * 60)
    logger.info("TESTING CONFIGURATION VALIDATION")
    logger.info("=" * 60)

    from src.wildetect.core.config import FlightSpecs, LoaderConfig

    # Test valid config
    config = LoaderConfig(tile_size=640, overlap=0.2, batch_size=4, recursive=True)
    assert config.tile_size == 640
    assert config.overlap == 0.2
    assert config.batch_size == 4
    assert config.recursive is True

    # Test flight specs
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )
    assert flight_specs.sensor_height == 24.0
    assert flight_specs.focal_length == 35.0
    assert flight_specs.flight_height == 180.0

    logger.info("✓ Configuration validation test passed")


def test_census_data_basic():
    """Test basic CensusData functionality."""
    logger.info("=" * 60)
    logger.info("TESTING CENSUS DATA BASIC")
    logger.info("=" * 60)

    from src.wildetect.core.config import LoaderConfig
    from src.wildetect.core.data.dataset import CensusData

    # Create CensusData
    config = LoaderConfig(tile_size=640, overlap=0.2)
    census_data = CensusData(campaign_id="test_campaign", loading_config=config)

    assert census_data.campaign_id == "test_campaign"
    assert census_data.loading_config is not None
    assert len(census_data.drone_images) == 0
    assert len(census_data.image_paths) == 0

    # Test adding image paths (using mock paths that will be filtered out)
    test_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
    census_data.add_images_from_paths(test_paths)
    # The method filters out non-existent paths, so we expect 0
    assert len(census_data.image_paths) == 0

    # Test with a valid path (create a temporary file)
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        census_data.add_images_from_paths([tmp_path])
        assert len(census_data.image_paths) == 1
    finally:
        os.unlink(tmp_path)

    logger.info("✓ CensusData basic test passed")


def run_all_basic_tests():
    """Run all basic functionality tests."""
    logger.info("Starting basic functionality tests...")

    try:
        test_imports()
        test_basic_operations()
        test_utils_functions()
        test_config_validation()
        test_census_data_basic()

        logger.info("=" * 60)
        logger.info("ALL BASIC FUNCTIONALITY TESTS PASSED! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_basic_tests()
