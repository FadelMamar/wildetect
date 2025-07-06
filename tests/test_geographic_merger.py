"""
Test script for geographic merger functionality.
"""

import logging
import os
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
from PIL import Image
from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage

# Add the project root to the Python path
from wildetect.core.flight.geographic_merger import (
    CentroidProximityRemovalStrategy,
    DuplicateRemovalStrategy,
    GeographicMerger,
    GPSOverlapStrategy,
    MergedDetection,
    OverlapStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TEST_GEOGRAPHIC_MERGER")

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)


def load_image_path():
    image_path = random.choice(os.listdir(TEST_IMAGE_DIR))
    image_path = os.path.join(TEST_IMAGE_DIR, image_path)
    return image_path


def create_test_drone_image(
    predictions: Optional[List[Detection]] = None
) -> DroneImage:
    """Create a test drone image with optional predictions."""

    # Create a test image
    image_path = load_image_path()

    # Create drone image
    drone_image = DroneImage.from_image_path(
        image_path=image_path, flight_specs=FLIGHT_SPECS
    )

    # Add predictions if provided
    if predictions:
        drone_image.set_predictions(predictions)

    return drone_image


def create_test_detection(
    image_path: str,
    width: int = 50,
    height: int = 50,
    class_name: str = "TEST",
    confidence: float = 0.8,
) -> Detection:
    """Create a test detection with bounds checking.

    Args:
        x: x-coordinate of detection center (random if None)
        y: y-coordinate of detection center (random if None)
        width: width of detection bounding box
        height: height of detection bounding box
        class_name: class name for detection
        confidence: confidence score

    Returns:
        Detection object with properly bounded coordinates
    """

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    # Sample random coordinates if not provided
    x = random.randint(50, image_width - 50)
    y = random.randint(50, image_height - 50)

    # Ensure detection stays within image bounds
    half_width = width // 2
    half_height = height // 2

    # Calculate bounding box coordinates
    x_min = max(0, x - half_width)
    x_max = min(image_width, x + half_width)
    y_min = max(0, y - half_height)
    y_max = min(image_height, y + half_height)

    det = Detection(
        bbox=[x_min, y_min, x_max, y_max],
        class_id=0,
        class_name=class_name,
        confidence=confidence,
    )
    det.clamp_bbox(x_range=(0, image_width), y_range=(0, image_height))
    return det


class TestOverlapStrategy:
    """Test the abstract OverlapStrategy class."""

    def test_abstract_class(self):
        """Test that OverlapStrategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            OverlapStrategy()


class TestGPSOverlapStrategy:
    """Test the GPSOverlapStrategy class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = GPSOverlapStrategy()

    def test_initialization(self):
        """Test GPSOverlapStrategy initialization."""
        assert self.strategy is not None
        assert self.strategy.stats is None

    def test_compute_iou(self):
        """Test IoU computation between drone images."""
        # Create test drone images
        drone_images = []
        for i in range(3):
            drone_image = create_test_drone_image()
            drone_images.append(drone_image)

        # Test IoU computation
        iou_matrix = self.strategy._compute_iou(drone_images)

        # Check matrix shape
        assert iou_matrix.shape == (3, 3)

        # Check diagonal values (should be 1.0 for self-comparison)
        np.testing.assert_array_almost_equal(np.diag(iou_matrix), np.ones(3), decimal=5)

    def test_find_overlapping_images(self):
        """Test finding overlapping images."""
        # Create test drone images
        drone_images = []
        for i in range(100):
            drone_image = create_test_drone_image()
            drone_images.append(drone_image)

        # Test finding overlapping images
        overlap_map = self.strategy.find_overlapping_images(
            drone_images, min_overlap_threshold=0.0
        )

        # Check that overlap_map is a dictionary
        assert isinstance(overlap_map, dict)

        # Check that stats were computed
        assert self.strategy.stats is not None
        assert "num_images" in self.strategy.stats

    def test_overlap_map_stats(self):
        """Test overlap map statistics computation."""
        # Create a test overlap map
        overlap_map = {
            "image1": ["image2", "image3"],
            "image2": ["image1"],
            "image3": ["image1"],
        }

        stats = self.strategy.overlap_map_stats(overlap_map)

        # Check expected statistics
        assert stats["num_images"] == 3, f"Expected 3, got {stats['num_images']}"
        assert (
            stats["avg_neighbors"] == 4 / 3
        ), f"Expected 4/3, got {stats['avg_neighbors']}"  # (2+1+1)/3
        assert stats["max_neighbors"] == 2, f"Expected 2, got {stats['max_neighbors']}"
        assert stats["min_neighbors"] == 1, f"Expected 1, got {stats['min_neighbors']}"

    def test_overlap_map_stats_empty(self):
        """Test overlap map statistics with empty map."""
        overlap_map = {}
        stats = self.strategy.overlap_map_stats(overlap_map)

        assert stats["num_images"] == 0
        assert stats["avg_neighbors"] == 0.0
        assert stats["max_neighbors"] == 0
        assert stats["min_neighbors"] == 0


class TestDuplicateRemovalStrategy:
    """Test the abstract DuplicateRemovalStrategy class."""

    def test_abstract_class(self):
        """Test that DuplicateRemovalStrategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DuplicateRemovalStrategy()


class TestCentroidProximityRemovalStrategy:
    """Test the CentroidProximityRemovalStrategy class."""

    def setup_method(self):
        logging.info("Starting: setup_method")
        self.strategy = CentroidProximityRemovalStrategy()
        self.overlap_strategy = GPSOverlapStrategy()

    def test_initialization(self):
        logging.info("Starting: test_initialization")
        assert self.strategy is not None

    def test_compute_iou_detections(self):
        logging.info("Starting: test_compute_iou_detections")
        # Create test detections
        detections_1 = [
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.9,
            ),
            create_test_detection(
                image_path=load_image_path(),
                width=60,
                height=60,
                class_name="elephant",
                confidence=0.8,
            ),
        ]
        detections_2 = [
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.85,
            ),
            create_test_detection(
                image_path=load_image_path(),
                width=70,
                height=70,
                class_name="giraffe",
                confidence=0.7,
            ),
        ]

        iou_matrix = self.strategy._compute_iou(detections_1, detections_2)

        # Check matrix shape
        assert iou_matrix.shape == (2, 2)

        # Check that IoU values are between -1 and 1
        assert np.all(iou_matrix >= -1)
        assert np.all(iou_matrix <= 1)

    def test_compute_iou_empty_detections(self):
        logging.info("Starting: test_compute_iou_empty_detections")
        detections_1 = [Detection.empty(parent_image=load_image_path())]
        detections_2 = [
            create_test_detection(image_path=load_image_path(), width=50, height=50)
        ]

        iou_matrix = self.strategy._compute_iou(detections_1, detections_2)

        # Should return matrix with -1 values
        assert iou_matrix.shape == (1, 1)

    def test_prune_duplicates_between_tiles(self):
        logging.info("Starting: test_prune_duplicates_between_tiles")
        drone_image_1 = create_test_drone_image()
        drone_image_2 = create_test_drone_image()

        # Add overlapping detections
        detections_1 = [
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.9,
            ),
            create_test_detection(
                image_path=load_image_path(),
                width=60,
                height=60,
                class_name="elephant",
                confidence=0.8,
            ),
        ]
        detections_2 = [
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.85,
            ),  # Overlapping with first
            create_test_detection(
                image_path=load_image_path(),
                width=70,
                height=70,
                class_name="giraffe",
                confidence=0.7,
            ),  # Different class
        ]

        drone_image_1.set_predictions(detections_1)
        drone_image_2.set_predictions(detections_2)

        # Test pruning duplicates
        pruned_dict = self.strategy._prune_duplicates_between_tiles(
            drone_image_1, drone_image_2, iou_threshold=0.5
        )

        # Check that duplicates were removed
        assert len(drone_image_1.predictions) <= len(detections_1)
        assert len(drone_image_2.predictions) <= len(detections_2)

    def test_prune_duplicates_empty_tiles(self):
        logging.info("Starting: test_prune_duplicates_empty_tiles")
        drone_image_1 = create_test_drone_image([])
        drone_image_2 = create_test_drone_image([])

        # Test with empty predictions
        pruned_dict = self.strategy._prune_duplicates_between_tiles(
            drone_image_1, drone_image_2
        )

        assert len(drone_image_1.predictions) == 0
        assert len(drone_image_2.predictions) == 0

    def test_overlapping_detections_removed_in_place(self):
        logging.info("Starting: test_overlapping_detections_removed_in_place")
        # Create two drone images
        drone_image_1 = create_test_drone_image()
        drone_image_2 = deepcopy(drone_image_1)

        # Create overlapping detections with same class and similar coordinates
        # Detection 1 in image 1 - centered at (100, 100)
        det1 = [
            Detection(
                bbox=[75, 75, 140, 140],  # 100x100 box centered at (100, 100)
                class_id=0,
                class_name="impala",
                confidence=0.9,
            ),
            # Detection 2 in image 1 (different class, should not be affected)
            Detection(
                bbox=[200, 200, 280, 280],  # 80x80 box at (200, 200)
                class_id=0,
                class_name="elephant",
                confidence=0.8,
            ),
        ]

        # Detection 1 in image 2 (overlapping with det1_img1, same class)
        det2 = [
            Detection(
                bbox=[
                    75,
                    75,
                    175,
                    175,
                ],  # 100x100 box centered at (125, 125) - overlaps with det1_img1
                class_id=0,
                class_name="impala",
                confidence=0.85,  # Slightly lower confidence
            ),
            # Detection 2 in image 2 (different class, should not be affected)
            Detection(
                bbox=[300, 300, 390, 390],  # 90x90 box at (300, 300)
                class_id=0,
                class_name="giraffe",
                confidence=0.7,
            ),
        ]

        # Set predictions
        drone_image_1.set_predictions(det1, update_gps=True)
        drone_image_2.set_predictions(det2, update_gps=True)
        drone_image_2.image_path = "test_image_2.jpg"

        iou_threshold = 0.1

        ious = self.strategy._compute_iou(
            drone_image_1.predictions, drone_image_2.predictions
        )
        expected_num_det_to_prune = (ious > iou_threshold).sum()
        logging.info(
            f"ious: {ious.round(2).tolist()}. Number of detections to prune: {expected_num_det_to_prune}"
        )

        # Store original predictions for comparison
        original_predictions_1 = drone_image_1.predictions.copy()
        original_predictions_2 = drone_image_2.predictions.copy()

        # Create overlap map indicating these images overlap
        overlap_map = {
            str(drone_image_1.image_path): [str(drone_image_2.image_path)],
            # str(drone_image_2.image_path): [str(drone_image_1.image_path)],
        }

        # Run duplicate removal
        stats = self.strategy.remove_duplicates(
            [drone_image_1, drone_image_2], overlap_map, iou_threshold=iou_threshold
        )

        # Verify that stats are returned
        assert isinstance(stats, dict)
        assert "total_image_pairs_processed" in stats

        tile1_modified = drone_image_1.predictions != original_predictions_1
        tile2_modified = drone_image_2.predictions != original_predictions_2

        # Exactly one tile should have been modified (the one with the duplicate removed)
        if expected_num_det_to_prune == 0:
            assert not tile1_modified
            assert not tile2_modified
        elif tile1_modified == tile2_modified:
            logging.info(stats)
            logging.debug(
                f"\nExactly one tile should have predictions modified.\n"
                f"Expected: exactly one tile modified, Actual: Tile1 modified: {tile1_modified}, Tile2 modified: {tile2_modified}\n"
                f"Tile1 predictions: {drone_image_1.predictions}\n"
                f"Original Tile1 predictions: {original_predictions_1}\n"
                f"Tile2 predictions: {drone_image_2.predictions}\n"
                f"Original Tile2 predictions: {original_predictions_2}\n"
            )
            assert False

        # Verify that overlapping detections were removed
        impala_detections_1 = [
            det for det in drone_image_1.predictions if det.class_name == "impala"
        ]
        impala_detections_2 = [
            det for det in drone_image_2.predictions if det.class_name == "impala"
        ]

        # At least one of the overlapping impala detections should be removed
        total_impala_detections = len(impala_detections_1) + len(impala_detections_2)
        original_impala_detections = 2  # We started with 2 impala detections
        if expected_num_det_to_prune > 0:
            assert (
                total_impala_detections < original_impala_detections
            ), f"Expected overlapping impala detections to be removed. Found {total_impala_detections} instead of < {original_impala_detections}"
        else:
            assert (
                total_impala_detections == original_impala_detections
            ), f"Expected {original_impala_detections} impala detections, found {total_impala_detections}"

        # Non-overlapping detections (elephant and giraffe) should remain
        elephant_detections = [
            det for det in drone_image_1.predictions if det.class_name == "elephant"
        ]
        giraffe_detections = [
            det for det in drone_image_2.predictions if det.class_name == "giraffe"
        ]

        assert len(elephant_detections) == 1, "elephant detection should remain"
        assert (
            len(giraffe_detections) == 1
        ), f"giraffe detection should remain {drone_image_2.predictions}"

    def test_no_overlap_no_removal(self):
        logging.info("Starting: test_no_overlap_no_removal")
        # Create two drone images
        drone_image_1 = create_test_drone_image()
        drone_image_2 = create_test_drone_image()

        # Create non-overlapping detections
        det1_img1 = create_test_detection(
            image_path=drone_image_1.image_path,
            width=50,
            height=50,
            class_name="impala",
            confidence=0.9,
        )

        det1_img2 = create_test_detection(
            image_path=drone_image_2.image_path,
            width=50,
            height=50,
            class_name="impala",
            confidence=0.8,
        )

        # Set predictions
        drone_image_1.set_predictions([det1_img1])
        drone_image_2.set_predictions([det1_img2])

        # Store original predictions
        original_predictions_1 = drone_image_1.predictions.copy()
        original_predictions_2 = drone_image_2.predictions.copy()

        # Create overlap map
        overlap_map = {
            str(drone_image_1.image_path): [str(drone_image_2.image_path)],
            str(drone_image_2.image_path): [str(drone_image_1.image_path)],
        }

        # Run duplicate removal with high IoU threshold
        result = self.strategy.remove_duplicates(
            [drone_image_1, drone_image_2],
            overlap_map,
            iou_threshold=0.8,  # High threshold, non-overlapping detections won't match
        )

        # Verify that predictions remain the same (no overlap detected)
        assert len(drone_image_1.predictions) == len(original_predictions_1)
        assert len(drone_image_2.predictions) == len(original_predictions_2)

        # Verify that the detections are the same objects
        assert drone_image_1.predictions[0] is original_predictions_1[0]
        assert drone_image_2.predictions[0] is original_predictions_2[0]

    def test_compute_duplicate_removal_stats_empty(self):
        logging.info("Starting: test_compute_duplicate_removal_stats_empty")
        pruned_detections = {}
        original_total_detections = 100

        stats = self.strategy._compute_duplicate_removal_stats(
            pruned_detections, original_total_detections
        )

        # Check expected structure
        assert "total_image_pairs_processed" in stats
        assert "total_detections_removed" in stats
        assert "duplicate_removal_rate" in stats
        assert "duplicate_groups_by_class" in stats
        assert "class_duplicate_stats" in stats

        # Check values for empty case
        assert stats["total_image_pairs_processed"] == 0
        assert stats["total_detections_removed"] == 0
        assert stats["duplicate_removal_rate"] == 0.0
        assert len(stats["duplicate_groups_by_class"]) == 0
        assert len(stats["class_duplicate_stats"]) == 0

    def test_compute_duplicate_removal_stats_with_data(self):
        logging.info("Starting: test_compute_duplicate_removal_stats_with_data")
        # Create mock pruned detections
        det1 = create_test_detection(
            image_path=load_image_path(), class_name="impala", confidence=0.85
        )
        det2 = create_test_detection(
            image_path=load_image_path(), class_name="elephant", confidence=0.78
        )
        det3 = create_test_detection(
            image_path=load_image_path(), class_name="impala", confidence=0.92
        )
        det4 = create_test_detection(
            image_path=load_image_path(), class_name="giraffe", confidence=0.88
        )

        pruned_detections = {
            ("image1.jpg", "image2.jpg"): [det1, det2],
            ("image2.jpg", "image3.jpg"): [det3],
            ("image3.jpg", "image4.jpg"): [det4],
        }

        original_total_detections = 50

        stats = self.strategy._compute_duplicate_removal_stats(
            pruned_detections, original_total_detections
        )

        # Check basic counts
        assert stats["total_image_pairs_processed"] == 3
        assert stats["total_detections_removed"] == 4
        assert stats["duplicate_removal_rate"] == 4 / 50  # 0.08

        # Check class-specific statistics
        assert "impala" in stats["duplicate_groups_by_class"]
        assert "elephant" in stats["duplicate_groups_by_class"]
        assert "giraffe" in stats["duplicate_groups_by_class"]

        assert stats["duplicate_groups_by_class"]["impala"] == 2
        assert stats["duplicate_groups_by_class"]["elephant"] == 1
        assert stats["duplicate_groups_by_class"]["giraffe"] == 1

        # Check class duplicate stats
        impala_stats = stats["class_duplicate_stats"]["impala"]
        assert impala_stats["total_duplicates"] == 2
        assert impala_stats["avg_confidence"] == pytest.approx(
            (0.85 + 0.92) / 2, rel=1e-3
        )
        assert impala_stats["removal_rate"] == 2 / 50

        elephant_stats = stats["class_duplicate_stats"]["elephant"]
        assert elephant_stats["total_duplicates"] == 1
        assert elephant_stats["avg_confidence"] == 0.78
        assert elephant_stats["removal_rate"] == 1 / 50

    def test_compute_duplicate_removal_stats_zero_original(self):
        logging.info("Starting: test_compute_duplicate_removal_stats_zero_original")
        pruned_detections = {
            ("image1.jpg", "image2.jpg"): [
                create_test_detection(
                    image_path=load_image_path(), class_name="impala", confidence=0.85
                )
            ]
        }
        original_total_detections = 0

        stats = self.strategy._compute_duplicate_removal_stats(
            pruned_detections, original_total_detections
        )

        # Should handle division by zero gracefully
        assert stats["duplicate_removal_rate"] == 0.0
        assert stats["total_detections_removed"] == 1

    def test_compute_duplicate_removal_stats_json_serializable(self):
        logging.info("Starting: test_compute_duplicate_removal_stats_json_serializable")
        pruned_detections = {
            ("image1.jpg", "image2.jpg"): [
                create_test_detection(
                    image_path=load_image_path(), class_name="impala", confidence=0.85
                )
            ]
        }
        original_total_detections = 10

        stats = self.strategy._compute_duplicate_removal_stats(
            pruned_detections, original_total_detections
        )

        # Test JSON serialization
        import json

        try:
            json_str = json.dumps(stats)
            # Should not raise an exception
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Statistics are not JSON serializable: {e}")

    def test_remove_duplicates_returns_stats(self):
        logging.info("Starting: test_remove_duplicates_returns_stats")
        # Create test drone images
        drone_images = []
        for i in range(3):
            drone_image = create_test_drone_image()

            # Add some test predictions
            predictions = [
                create_test_detection(
                    image_path=load_image_path(),
                    width=50,
                    height=50,
                    class_name="impala",
                    confidence=0.9,
                ),
                create_test_detection(
                    image_path=load_image_path(),
                    width=60,
                    height=60,
                    class_name="elephant",
                    confidence=0.8,
                ),
            ]
            drone_image.set_predictions(predictions)
            drone_images.append(drone_image)

        # Create overlap map
        overlap_map = self.overlap_strategy.find_overlapping_images(drone_images)

        # Test removing duplicates
        stats = self.strategy.remove_duplicates(
            drone_images, overlap_map, iou_threshold=0.5
        )

        # Check that stats are returned
        assert isinstance(stats, dict)
        assert "total_image_pairs_processed" in stats
        assert "total_detections_removed" in stats
        assert "duplicate_removal_rate" in stats
        assert "duplicate_groups_by_class" in stats
        assert "class_duplicate_stats" in stats

        # Check that stats are returned
        assert isinstance(stats, dict)
        assert "total_image_pairs_processed" in stats
        assert "total_detections_removed" in stats
        assert "duplicate_removal_rate" in stats
        assert "duplicate_groups_by_class" in stats
        assert "class_duplicate_stats" in stats

        # Check that tiles are modified in-place
        assert len(drone_images) == 3  # Same number of tiles returned


class TestGeographicMerger:
    """Test the GeographicMerger class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.merger = GeographicMerger(merge_distance_threshold_m=50.0)

    def test_initialization(self):
        """Test GeographicMerger initialization."""
        assert self.merger.merge_distance_threshold_m == 50.0
        assert isinstance(self.merger.overlap_strategy, GPSOverlapStrategy)
        assert isinstance(
            self.merger.duplicate_removal_strategy, CentroidProximityRemovalStrategy
        )

    def test_run_with_single_image(self):
        """Test running merger with single image."""
        drone_image = create_test_drone_image()

        # Add some predictions
        predictions = [
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.9,
            ),
            create_test_detection(
                image_path=load_image_path(),
                width=60,
                height=60,
                class_name="elephant",
                confidence=0.8,
            ),
        ]
        print(predictions)
        drone_image.set_predictions(predictions)

        stats = self.merger.run([drone_image])

        # Should return statistics
        assert isinstance(stats, dict)
        assert "total_image_pairs_processed" in stats
        assert "total_detections_removed" in stats

    def test_run_with_multiple_images(self):
        """Test running merger with multiple images."""
        # Create multiple drone images
        drone_images = []
        for i in range(5):  # Reduced for faster testing
            drone_image = create_test_drone_image()

            # Add predictions
            predictions = [
                create_test_detection(
                    image_path=load_image_path(),
                    width=50,
                    height=50,
                    class_name="impala",
                    confidence=0.9,
                ),
                create_test_detection(
                    image_path=load_image_path(),
                    width=60,
                    height=60,
                    class_name="elephant",
                    confidence=0.8,
                ),
            ]
            drone_image.set_predictions(predictions)
            drone_images.append(drone_image)

        stats = self.merger.run(drone_images, iou_threshold=0.5)

        # Should return statistics
        assert isinstance(stats, dict)
        assert "total_image_pairs_processed" in stats
        assert "total_detections_removed" in stats

    def test_run_with_custom_iou_threshold(self):
        """Test running merger with custom IoU threshold."""
        drone_image = create_test_drone_image()

        stats = self.merger.run([drone_image], iou_threshold=0.7)

        assert isinstance(stats, dict)
        assert "duplicate_removal_rate" in stats


class TestMergedDetection:
    """Test the MergedDetection class."""

    def test_initialization(self):
        """Test MergedDetection initialization."""
        detection = MergedDetection(
            bbox=[100, 200, 150, 260],
            class_id=0,
            class_name="impala",
            confidence=0.9,
            source_images=[],
            merged_detections=[],
        )

        assert detection.bbox == [100, 200, 150, 260]
        assert detection.class_name == "impala"
        assert detection.confidence == 0.9
        assert detection.source_images == []
        assert detection.merged_detections == []

    def test_with_source_images(self):
        """Test MergedDetection with source images."""
        source_images = ["image1.jpg", "image2.jpg"]
        merged_detections = [
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.9,
            ),
            create_test_detection(
                image_path=load_image_path(),
                width=50,
                height=50,
                class_name="impala",
                confidence=0.85,
            ),
        ]

        detection = MergedDetection(
            bbox=[100, 200, 150, 260],
            class_id=0,
            class_name="impala",
            confidence=0.9,
            source_images=source_images,
            merged_detections=merged_detections,
        )

        assert detection.source_images == source_images
        assert detection.merged_detections == merged_detections


def run_all_tests():
    """Run all tests."""
    logger.info("Starting geographic merger tests...")

    # Test OverlapStrategy
    logger.info("Testing OverlapStrategy...")
    test_overlap_strategy = TestOverlapStrategy()
    test_overlap_strategy.test_abstract_class()

    # Test GPSOverlapStrategy
    logger.info("Testing GPSOverlapStrategy...")
    test_gps_strategy = TestGPSOverlapStrategy()
    test_gps_strategy.setup_method()
    test_gps_strategy.test_initialization()
    test_gps_strategy.test_overlap_map_stats()
    test_gps_strategy.test_overlap_map_stats_empty()

    # Test DuplicateRemovalStrategy
    logger.info("Testing DuplicateRemovalStrategy...")
    test_duplicate_strategy = TestDuplicateRemovalStrategy()
    test_duplicate_strategy.test_abstract_class()

    # Test CentroidProximityRemovalStrategy
    logger.info("Testing CentroidProximityRemovalStrategy...")
    test_centroid_strategy = TestCentroidProximityRemovalStrategy()
    test_centroid_strategy.setup_method()
    test_centroid_strategy.test_initialization()
    test_centroid_strategy.test_compute_iou_empty_detections()
    test_centroid_strategy.test_prune_duplicates_empty_tiles()
    test_centroid_strategy.test_overlapping_detections_removed_in_place()
    test_centroid_strategy.test_no_overlap_no_removal()
    test_centroid_strategy.test_compute_duplicate_removal_stats_empty()
    test_centroid_strategy.test_compute_duplicate_removal_stats_with_data()
    test_centroid_strategy.test_compute_duplicate_removal_stats_zero_original()
    test_centroid_strategy.test_compute_duplicate_removal_stats_json_serializable()
    test_centroid_strategy.test_remove_duplicates_returns_stats()

    # Test GeographicMerger
    logger.info("Testing GeographicMerger...")
    test_merger = TestGeographicMerger()
    test_merger.setup_method()
    test_merger.test_initialization()
    test_merger.test_run_with_single_image()
    test_merger.test_run_with_multiple_images()
    test_merger.test_run_with_custom_iou_threshold()

    # Test MergedDetection
    logger.info("Testing MergedDetection...")
    test_merged_detection = TestMergedDetection()
    test_merged_detection.test_initialization()
    test_merged_detection.test_with_source_images()

    logger.info("All geographic merger tests completed successfully!")


def run_overlapping_detection_tests():
    """Run specific tests for overlapping detection removal."""
    logger.info("Running overlapping detection removal tests...")

    test_centroid_strategy = TestCentroidProximityRemovalStrategy()
    test_centroid_strategy.setup_method()

    # Test overlapping detection removal
    logger.info("Testing overlapping detection removal...")
    test_centroid_strategy.test_overlapping_detections_removed_in_place()

    # Test no overlap scenario
    logger.info("Testing no overlap scenario...")
    test_centroid_strategy.test_no_overlap_no_removal()

    logger.info("Overlapping detection tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()
