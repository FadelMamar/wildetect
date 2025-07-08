"""
Tests for updated FiftyOneManager integration with Detection and DroneImage classes.

This test suite verifies the integration between WildDetect's data structures
and FiftyOne's dataset management capabilities, including native geolocation support.
"""

import logging
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import fiftyone as fo
from wildetect.core.config import FlightSpecs
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.visualization.fiftyone_manager import FiftyOneManager

TEST_IMAGE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"
FLIGHT_SPECS = FlightSpecs(sensor_height=24.0, focal_length=35.0, flight_height=180.0)
NUM_IMAGES = 100
TEST_IMAGE_DIR_NO_GPS = r"D:\workspace\data\general_dataset\original-data\train\images"

# Cache image paths to ensure consistency
_cached_image_paths = None
_cached_no_gps_paths = None


def get_cached_image_paths():
    global _cached_image_paths
    if _cached_image_paths is None:
        _cached_image_paths = [
            os.path.join(TEST_IMAGE_DIR, f)
            for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    return _cached_image_paths


def get_cached_no_gps_paths():
    global _cached_no_gps_paths
    if _cached_no_gps_paths is None:
        _cached_no_gps_paths = [
            os.path.join(TEST_IMAGE_DIR_NO_GPS, f)
            for f in os.listdir(TEST_IMAGE_DIR_NO_GPS)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    return _cached_no_gps_paths


def load_image_path():
    paths = get_cached_image_paths()
    return (
        random.choice(paths)
        if paths
        else r"D:\workspace\data\savmap_dataset_v2\raw\images\sample.jpg"
    )


def load_image_no_gps_path():
    paths = get_cached_no_gps_paths()
    return (
        random.choice(paths)
        if paths
        else r"D:\workspace\data\general_dataset\original-data\train\images\sample.jpg"
    )


def create_test_images() -> list[DroneImage]:
    """Create test DroneImage instances with geographic footprints."""
    drone_image = [
        DroneImage.from_image_path(load_image_path(), flight_specs=FLIGHT_SPECS)
        for _ in range(NUM_IMAGES)
    ]
    return drone_image


class TestFiftyOneManagerUpdated(unittest.TestCase):
    """Test cases for updated FiftyOneManager integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Configure logging for tests
        logging.basicConfig(level=logging.ERROR)

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_name = "test_wildlife_dataset_updated"

        # Create test image files
        self.create_test_images()

        # Sample configuration
        self.config = {
            "fiftyone": {"dataset_name": self.test_dataset_name, "enable_brain": False}
        }

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up FiftyOne datasets
        try:
            fo.delete_dataset(self.test_dataset_name)
        except ValueError:
            pass  # Dataset doesn't exist

        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_images(self) -> list[DroneImage]:
        """Create test DroneImage instances with geographic footprints."""
        self.drone_images = [
            DroneImage.from_image_path(load_image_path(), flight_specs=FLIGHT_SPECS)
            for _ in range(NUM_IMAGES)
        ]
        return self.drone_images

    def create_sample_detections(self) -> List[Detection]:
        """Create sample Detection objects for testing."""
        detections = []

        # Sample detection 1
        detection1 = Detection(
            bbox=[10, 20, 50, 80],  # [x1, y1, x2, y2]
            confidence=0.85,
            class_id=1,
            class_name="giraffe",
        )
        detections.append(detection1)

        # Sample detection 2
        detection2 = Detection(
            bbox=[60, 30, 90, 70],
            confidence=0.72,
            class_id=2,
            class_name="elephant",
        )
        detections.append(detection2)

        return detections

    def create_sample_drone_image(self) -> DroneImage:
        """Create a sample DroneImage for testing."""
        # Create a drone image with sample data
        drone_image = DroneImage.from_image_path(
            image_path=load_image_path(), flight_specs=FLIGHT_SPECS
        )

        # Add sample detections
        detections = self.create_sample_detections()
        drone_image.set_predictions(detections)

        return drone_image

    def test_initialization(self):
        """Test FiftyOneManager initialization."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        self.assertIsNotNone(manager.dataset)
        self.assertEqual(manager.dataset_name, self.test_dataset_name)
        self.assertEqual(manager.config, self.config)

    def test_add_detections(self):
        """Test adding Detection objects to dataset."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        detections = self.create_sample_detections()

        # Use a consistent image path
        image_path = load_image_path()

        # Add detections

        # Verify dataset has the sample
        sample = manager.dataset.first()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.filepath, image_path)

        # Verify detections were added
        self.assertIn("detections", sample)
        self.assertEqual(len(sample.detections), 2)

        # Verify detection properties - use .detections attribute
        fo_detection = sample.detections.detections[0]
        self.assertEqual(fo_detection.label, "giraffe")
        self.assertEqual(fo_detection.confidence, 0.85)

    def test_add_drone_image_with_geolocation(self):
        """Test adding DroneImage with native geolocation to dataset."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        drone_image = self.create_sample_drone_image()

        # Add drone image
        manager.add_drone_image(drone_image)

        # Verify dataset has the sample
        sample = manager.dataset.first()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.filepath, drone_image.image_path)

        # Verify native geolocation was added
        self.assertIn("location", sample)
        self.assertIsInstance(sample["location"], fo.GeoLocation)

        # Verify point coordinates - use actual GPS coordinates from the image
        self.assertIsNotNone(sample["location"].point)
        self.assertEqual(len(sample["location"].point), 2)  # [lon, lat]

        # Verify polygon was added if available
        if sample["location"].polygon:
            self.assertIsInstance(sample["location"].polygon, list)

        # Verify metadata was preserved
        self.assertEqual(sample["total_count"], 2)
        self.assertEqual(sample["num_tiles"], 1)  # Default single tile
        self.assertIn("image_width", sample)
        self.assertIn("image_height", sample)
        self.assertIn("gsd", sample)
        self.assertIn("timestamp", sample)

        # Verify species counts
        species_counts = sample["species_counts"]
        self.assertEqual(species_counts["giraffe"], 1)
        self.assertEqual(species_counts["elephant"], 1)

    def test_add_drone_image_without_geolocation(self):
        """Test adding DroneImage without GPS data."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create drone image without GPS
        drone_image = DroneImage.from_image_path(
            image_path=load_image_no_gps_path(), flight_specs=FLIGHT_SPECS
        )

        # Add detections
        detections = self.create_sample_detections()
        drone_image.set_predictions(detections)

        # Add drone image
        manager.add_drone_image(drone_image)

        # Verify dataset has the sample
        sample = manager.dataset.first()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.filepath, drone_image.image_path)

        # Verify no geolocation was added
        self.assertNotIn("location", sample)

        # Verify other metadata was preserved
        self.assertEqual(sample["total_count"], 2)
        self.assertIn("gsd", sample)  # Don't check specific value as it varies

    def test_add_drone_images_batch(self):
        """Test adding multiple DroneImage objects."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create multiple drone images
        drone_image1 = self.create_sample_drone_image()
        drone_image2 = self.create_sample_drone_image()

        drone_images = [drone_image1, drone_image2]

        # Add drone images in batch
        manager.add_drone_images(drone_images)

        # Verify both samples were added
        self.assertEqual(manager.dataset.count(), 2)

        # Verify samples have correct filepaths
        filepaths = [sample.filepath for sample in manager.dataset]
        self.assertIn(drone_image1.image_path, filepaths)
        self.assertIn(drone_image2.image_path, filepaths)

    def test_get_detections_with_gps(self):
        """Test filtering samples with GPS data."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        drone_image = self.create_sample_drone_image()

        # Add drone image with GPS data
        manager.add_drone_image(drone_image)

        # Get samples with GPS data
        gps_samples = manager.get_detections_with_gps()

        # Verify GPS samples were found
        self.assertEqual(len(gps_samples), 1)
        sample = gps_samples.first()
        self.assertIsNotNone(sample["location"])

    def test_get_detections_without_gps(self):
        """Test filtering samples without GPS data."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create drone image without GPS
        drone_image = DroneImage.from_image_path(
            image_path=load_image_no_gps_path(), flight_specs=FLIGHT_SPECS
        )
        detections = self.create_sample_detections()
        drone_image.set_predictions(detections)

        # Add drone image without GPS
        manager.add_drone_image(drone_image)

        # Get samples with GPS data
        gps_samples = manager.get_detections_with_gps()

        # Verify no GPS samples were found
        self.assertEqual(len(gps_samples), 0)

    def test_native_geolocation_structure(self):
        """Test that native FiftyOne geolocation is properly structured."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        drone_image = self.create_sample_drone_image()

        # Add drone image
        manager.add_drone_image(drone_image)

        # Get sample
        sample = manager.dataset.first()
        location = sample["location"]

        # Verify FiftyOne GeoLocation structure
        self.assertIsInstance(location, fo.GeoLocation)
        self.assertIsInstance(location.point, list)
        self.assertEqual(len(location.point), 2)  # [lon, lat]
        if location.polygon:
            self.assertIsInstance(location.polygon, list)

    def test_coordinate_transformation(self):
        """Test coordinate transformation in detections."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create detection with relative coordinates
        detection = Detection(
            bbox=[10, 20, 50, 80],
            confidence=0.85,
            class_id=1,
            class_name="deer",
            parent_image=load_image_path(),
        )

        # Verify FiftyOne detection format
        sample = manager.dataset.first()
        fo_detection = sample.detections.detections[0]  # Use .detections attribute

        # FiftyOne uses [x, y, width, height] format
        self.assertEqual(fo_detection.bounding_box[0], 10)  # x
        self.assertEqual(fo_detection.bounding_box[1], 20)  # y
        self.assertEqual(fo_detection.bounding_box[2], 40)  # width
        self.assertEqual(fo_detection.bounding_box[3], 60)  # height

    def test_metadata_preservation(self):
        """Test that all metadata is preserved correctly."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create detection with rich metadata
        detection = Detection(
            bbox=[10, 20, 50, 80],
            confidence=0.85,
            class_id=1,
            class_name="deer",
            gps_loc="40.7128,-74.0060",
            image_gps_loc="40.7128,-74.0060",
            parent_image=load_image_path(),
            metadata={"custom_field": "custom_value"},
        )

        # Add detection

        # Verify metadata was preserved
        sample = manager.dataset.first()
        fo_detection = sample.detections.detections[0]  # Use .detections attribute

        # Check metadata fields
        self.assertEqual(fo_detection.metadata["gps_loc"], "40.7128,-74.0060")
        self.assertEqual(fo_detection.metadata["image_gps_loc"], "40.7128,-74.0060")
        self.assertEqual(fo_detection.metadata["parent_image"], detection.parent_image)
        self.assertEqual(fo_detection.metadata["custom_field"], "custom_value")

    def test_empty_detections(self):
        """Test handling of empty detections."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create empty detection
        empty_detection = Detection.empty(load_image_path())
        detections = [empty_detection]

        # Verify sample was added but no detections
        sample = manager.dataset.first()
        self.assertIsNotNone(sample)
        self.assertEqual(sample["total_count"], 0)

    def test_dataset_initialization_failure(self):
        """Test handling of dataset initialization failure."""
        # Test with invalid dataset name
        with self.assertRaises((ValueError, RuntimeError)):
            manager = FiftyOneManager("", self.config)

    def test_save_and_load_dataset(self):
        """Test saving and loading dataset."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        drone_image = self.create_sample_drone_image()

        # Add sample data
        manager.add_drone_image(drone_image)

        # Save dataset
        manager.save_dataset()

        # Create new manager and load dataset
        new_manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Verify data was preserved
        self.assertEqual(new_manager.dataset.count(), 1)
        sample = new_manager.dataset.first()
        self.assertEqual(sample["total_count"], 2)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Test with None dataset (should raise RuntimeError)
        manager.dataset = None

        with self.assertRaises(RuntimeError):
            manager.get_annotation_stats()

        with self.assertRaises(RuntimeError):
            manager._ensure_dataset_initialized()

        with self.assertRaises(RuntimeError):
            manager.get_dataset_info()

    def test_launch_app_static_method(self):
        """Test the static launch_app method."""
        with patch("subprocess.run") as mock_run:
            FiftyOneManager.launch_app()
            mock_run.assert_called_once_with(["uv", "run", "fiftyone", "launch", "app"])

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        drone_image = self.create_sample_drone_image()

        # Add sample data
        manager.add_drone_image(drone_image)

        # Get dataset info
        info = manager.get_dataset_info()

        # Verify info structure
        self.assertIn("name", info)
        self.assertIn("num_samples", info)
        self.assertIn("tags", info)
        self.assertIn("fields", info)

        # Verify values
        self.assertEqual(info["name"], self.test_dataset_name)
        self.assertEqual(info["num_samples"], 1)

    def test_get_annotation_stats(self):
        """Test getting annotation statistics."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)
        drone_image = self.create_sample_drone_image()

        # Add sample data
        manager.add_drone_image(drone_image)

        # Get statistics
        stats = manager.get_annotation_stats()

        # Verify stats structure
        self.assertIn("total_samples", stats)
        self.assertIn("annotated_samples", stats)
        self.assertIn("total_detections", stats)
        self.assertIn("species_counts", stats)

        # Verify values
        self.assertEqual(stats["total_samples"], 1)
        self.assertEqual(stats["annotated_samples"], 1)
        self.assertEqual(stats["total_detections"], 2)
        self.assertEqual(stats["species_counts"]["giraffe"], 1)
        self.assertEqual(stats["species_counts"]["elephant"], 1)


class TestFiftyOneManagerIntegrationUpdated(unittest.TestCase):
    """Integration tests for updated FiftyOneManager with real FiftyOne operations."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.test_dataset_name = "integration_test_dataset_updated"
        self.config = {
            "fiftyone": {"dataset_name": self.test_dataset_name, "enable_brain": False}
        }

    def tearDown(self):
        """Clean up integration test fixtures."""
        try:
            fo.delete_dataset(self.test_dataset_name)
        except ValueError:
            pass

    def test_native_geolocation_creation(self):
        """Test creating native FiftyOne geolocation."""
        # Create a sample with native FiftyOne geolocation
        sample = fo.Sample(filepath=load_image_path())

        # Add native FiftyOne geolocation
        sample["location"] = fo.GeoLocation(
            point=[-74.0060, 40.7128],  # [longitude, latitude]
            polygon=[
                [
                    [-74.0160, 40.7028],  # Bottom-left
                    [-73.9960, 40.7028],  # Bottom-right
                    [-73.9960, 40.7228],  # Top-right
                    [-74.0160, 40.7228],  # Top-left
                    [-74.0160, 40.7028],  # Close polygon
                ]
            ],
        )

        # Verify structure
        self.assertIsInstance(sample["location"], fo.GeoLocation)
        self.assertEqual(sample["location"].point[0], -74.0060)  # longitude
        self.assertEqual(sample["location"].point[1], 40.7128)  # latitude
        self.assertEqual(len(sample["location"].polygon), 1)
        self.assertEqual(len(sample["location"].polygon[0]), 5)

    def test_geolocation_filtering(self):
        """Test filtering by geolocation."""
        manager = FiftyOneManager(self.test_dataset_name, self.config)

        # Create and add drone image with GPS
        drone_image = DroneImage.from_image_path(
            image_path=load_image_path(), flight_specs=FLIGHT_SPECS
        )
        manager.add_drone_image(drone_image)

        # Test filtering by location existence
        gps_samples = manager.get_detections_with_gps()
        self.assertEqual(len(gps_samples), 1)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
