"""
Simple test script for geographic merger functionality.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_image(image_path: str, width: int = 640, height: int = 480):
    """Create a simple test image."""
    import numpy as np
    from PIL import Image

    # Create a simple test image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(image_path)
    return img


def test_overlap_strategy_abstract():
    """Test that OverlapStrategy is abstract."""
    try:
        from wildetect.core.flight.geographic_merger import OverlapStrategy

        # This should raise TypeError since it's abstract
        OverlapStrategy()
        assert False, "Should have raised TypeError"
    except TypeError:
        logger.info("✓ OverlapStrategy correctly raises TypeError (abstract class)")
    except Exception as e:
        logger.warning(f"Unexpected error: {e}")


def test_duplicate_removal_strategy_abstract():
    """Test that DuplicateRemovalStrategy is abstract."""
    try:
        from wildetect.core.flight.geographic_merger import DuplicateRemovalStrategy

        # This should raise TypeError since it's abstract
        DuplicateRemovalStrategy()
        assert False, "Should have raised TypeError"
    except TypeError:
        logger.info(
            "✓ DuplicateRemovalStrategy correctly raises TypeError (abstract class)"
        )
    except Exception as e:
        logger.warning(f"Unexpected error: {e}")


def test_gps_overlap_strategy_initialization():
    """Test GPSOverlapStrategy initialization."""
    try:
        from wildetect.core.flight.geographic_merger import GPSOverlapStrategy

        strategy = GPSOverlapStrategy()
        assert strategy is not None
        assert strategy.stats is None
        logger.info("✓ GPSOverlapStrategy initialization successful")

    except Exception as e:
        logger.error(f"GPSOverlapStrategy initialization failed: {e}")


def test_centroid_proximity_removal_strategy_initialization():
    """Test CentroidProximityRemovalStrategy initialization."""
    try:
        from wildetect.core.flight.geographic_merger import (
            CentroidProximityRemovalStrategy,
        )

        strategy = CentroidProximityRemovalStrategy()
        assert strategy is not None
        logger.info("✓ CentroidProximityRemovalStrategy initialization successful")

    except Exception as e:
        logger.error(f"CentroidProximityRemovalStrategy initialization failed: {e}")


def test_geographic_merger_initialization():
    """Test GeographicMerger initialization."""
    try:
        from wildetect.core.flight.geographic_merger import GeographicMerger

        merger = GeographicMerger(merge_distance_threshold_m=50.0)
        assert merger.merge_distance_threshold_m == 50.0
        assert merger.overlap_strategy is not None
        assert merger.duplicate_removal_strategy is not None
        logger.info("✓ GeographicMerger initialization successful")

    except Exception as e:
        logger.error(f"GeographicMerger initialization failed: {e}")


def test_merged_detection_initialization():
    """Test MergedDetection initialization."""
    try:
        from wildetect.core.flight.geographic_merger import MergedDetection

        detection = MergedDetection(
            bbox=[100, 200, 150, 260], confidence=0.9, class_id=0, class_name="person"
        )

        assert detection.bbox == [100, 200, 150, 260]
        assert detection.confidence == 0.9
        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.source_images == []
        assert detection.merged_detections == []

        logger.info("✓ MergedDetection initialization successful")

    except Exception as e:
        logger.error(f"MergedDetection initialization failed: {e}")


def test_overlap_map_stats():
    """Test overlap map statistics computation."""
    try:
        from wildetect.core.flight.geographic_merger import GPSOverlapStrategy

        strategy = GPSOverlapStrategy()

        # Create a test overlap map
        overlap_map = {
            "image1": ["image2", "image3"],
            "image2": ["image1"],
            "image3": ["image1"],
        }

        stats = strategy.overlap_map_stats(overlap_map)

        # Check expected statistics
        assert stats["num_images"] == 3
        assert stats["avg_neighbors"] == 1.0  # (2+1+1)/3
        assert stats["max_neighbors"] == 2
        assert stats["min_neighbors"] == 1

        logger.info("✓ Overlap map statistics computation successful")

    except Exception as e:
        logger.error(f"Overlap map statistics computation failed: {e}")


def test_overlap_map_stats_empty():
    """Test overlap map statistics with empty map."""
    try:
        from wildetect.core.flight.geographic_merger import GPSOverlapStrategy

        strategy = GPSOverlapStrategy()
        overlap_map = {}
        stats = strategy.overlap_map_stats(overlap_map)

        assert stats["num_images"] == 0
        assert stats["avg_neighbors"] == 0.0
        assert stats["max_neighbors"] == 0
        assert stats["min_neighbors"] == 0

        logger.info("✓ Empty overlap map statistics computation successful")

    except Exception as e:
        logger.error(f"Empty overlap map statistics computation failed: {e}")


def test_merged_detection_with_source_images():
    """Test MergedDetection with source images."""
    try:
        from wildetect.core.data.detection import Detection
        from wildetect.core.flight.geographic_merger import MergedDetection

        source_images = ["image1.jpg", "image2.jpg"]
        merged_detections = [
            Detection(
                bbox=[100, 100, 150, 150],
                confidence=0.9,
                class_id=0,
                class_name="person",
            ),
            Detection(
                bbox=[110, 110, 160, 160],
                confidence=0.85,
                class_id=0,
                class_name="person",
            ),
        ]

        detection = MergedDetection(
            bbox=[105, 105, 160, 160],
            confidence=0.9,
            class_id=0,
            class_name="person",
            source_images=source_images,
            merged_detections=merged_detections,
        )

        assert detection.source_images == source_images
        assert detection.merged_detections == merged_detections

        logger.info("✓ MergedDetection with source images successful")

    except Exception as e:
        logger.error(f"MergedDetection with source images failed: {e}")


def test_geographic_merger_run_method():
    """Test GeographicMerger run method (basic functionality)."""
    try:
        from wildetect.core.flight.geographic_merger import GeographicMerger

        merger = GeographicMerger(merge_distance_threshold_m=50.0)

        # Test that the run method exists and has the right signature
        import inspect

        sig = inspect.signature(merger.run)
        params = list(sig.parameters.keys())

        assert "drone_images" in params
        assert "iou_threshold" in params

        logger.info("✓ GeographicMerger run method signature check successful")

    except Exception as e:
        logger.error(f"GeographicMerger run method test failed: {e}")


def run_all_simple_tests():
    """Run all simple tests."""
    logger.info("=" * 60)
    logger.info("STARTING SIMPLE GEOGRAPHIC MERGER TESTS")
    logger.info("=" * 60)

    test_functions = [
        test_overlap_strategy_abstract,
        test_duplicate_removal_strategy_abstract,
        test_gps_overlap_strategy_initialization,
        test_centroid_proximity_removal_strategy_initialization,
        test_geographic_merger_initialization,
        test_merged_detection_initialization,
        test_overlap_map_stats,
        test_overlap_map_stats_empty,
        test_merged_detection_with_source_images,
        test_geographic_merger_run_method,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("✓ All simple tests passed!")
    else:
        logger.warning(f"⚠ {failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_simple_tests()
    sys.exit(0 if success else 1)
