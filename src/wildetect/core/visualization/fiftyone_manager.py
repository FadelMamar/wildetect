"""
FiftyOne integration for WildDetect.

This module handles dataset creation, visualization, and annotation collection
using FiftyOne for wildlife detection datasets.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F

from ..config import get_config
from ..data.detection import Detection
from ..data.drone_image import DroneImage

logger = logging.getLogger(__name__)


class FiftyOneManager:
    """Manages FiftyOne datasets for wildlife detection."""

    def __init__(self, dataset_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize FiftyOne manager.

        Args:
            dataset_name: Name of the dataset to use
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.dataset_name = dataset_name
        self.dataset = None

        # Initialize dataset
        self._init_dataset()

    def _init_dataset(self):
        """Initialize or load the FiftyOne dataset."""
        try:
            # Try to load existing dataset
            self.dataset = fo.load_dataset(self.dataset_name)
            logger.info(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            # Create new dataset
            self.dataset = fo.Dataset(self.dataset_name)
            logger.info(f"Created new dataset: {self.dataset_name}")

    def _ensure_dataset_initialized(self):
        """Ensure dataset is initialized before operations."""
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call _init_dataset() first.")

    def add_images(
        self, image_paths: List[str], detections: Optional[List[Dict[str, Any]]] = None
    ):
        """Add images to the dataset.

        Args:
            image_paths: List of image file paths
            detections: Optional list of detection results
        """
        self._ensure_dataset_initialized()

        # Create a list of None values if detections is None
        detection_list: List[Optional[Dict[str, Any]]] = []
        if detections is None:
            detection_list = [None] * len(image_paths)
        else:
            # Convert to List[Optional[Dict[str, Any]]] for type safety
            detection_list = [det for det in detections]

        samples = []
        for image_path, detection in zip(image_paths, detection_list):
            sample = fo.Sample(filepath=image_path)

            if detection and "detections" in detection:
                # Add detection annotations
                detections_list = []
                for det in detection["detections"]:
                    detection_obj = fo.Detection(
                        bounding_box=det["bbox"],
                        confidence=det["confidence"],
                        label=det["class_name"],
                    )
                    detections_list.append(detection_obj)

                sample["detections"] = fo.Detections(detections=detections_list)
                sample["total_count"] = detection.get("total_count", 0)
                sample["species_counts"] = detection.get("species_counts", {})

            samples.append(sample)

        if self.dataset is not None:
            self.dataset.add_samples(samples)
            logger.info(f"Added {len(samples)} images to dataset")

    def add_detections(self, detections: List[Detection], image_path: str):
        """Add Detection objects to the dataset.

        Args:
            detections: List of Detection objects
            image_path: Path to the source image
        """
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        # Convert Detection objects to FiftyOne format
        fo_detections = []
        for detection in detections:
            if not detection.is_empty:
                fo_detection = detection.to_fiftyone()
                fo_detections.append(fo_detection)

        # Create or update sample
        try:
            # Try to find existing sample by filepath
            existing_samples = self.dataset.match(F("filepath") == image_path)
            if len(existing_samples) > 0:
                sample = existing_samples.first()
                # Update existing sample
                if fo_detections:
                    sample["detections"] = fo.Detections(detections=fo_detections)
                sample["total_count"] = len(fo_detections)
                sample.save()
            else:
                # Create new sample
                sample = fo.Sample(filepath=image_path)
                if fo_detections:
                    sample["detections"] = fo.Detections(detections=fo_detections)
                sample["total_count"] = len(fo_detections)
                self.dataset.add_sample(sample)
        except Exception as e:
            logger.error(f"Error adding detections for {image_path}: {e}")
            # Fallback: create new sample
            # sample = fo.Sample(filepath=image_path)
            # if fo_detections:
            #     sample["detections"] = fo.Detections(detections=fo_detections)
            # sample["total_count"] = len(fo_detections)
            # if self.dataset is not None:
            #     self.dataset.add_sample(sample)

        logger.info(f"Added {len(fo_detections)} detections for image: {image_path}")

    def add_drone_image(self, drone_image: DroneImage):
        """Add a DroneImage to the dataset.

        Args:
            drone_image: DroneImage object to add
        """
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        # Get all detections from the drone image
        all_detections = drone_image.get_all_predictions()

        # Convert to FiftyOne format
        fo_detections = []
        for detection in all_detections:
            if not detection.is_empty:
                fo_detection = detection.to_fiftyone()
                fo_detections.append(fo_detection)

        # Create sample with metadata
        sample = fo.Sample(filepath=drone_image.image_path)

        if fo_detections:
            sample["detections"] = fo.Detections(detections=fo_detections)

        # Add metadata
        sample["total_count"] = len(fo_detections)
        sample["num_tiles"] = len(drone_image.tiles)
        sample["image_width"] = drone_image.width
        sample["image_height"] = drone_image.height

        # Add native FiftyOne geolocation if GPS data is available
        if drone_image.latitude is not None and drone_image.longitude is not None:
            # Create GeoLocation with point coordinates
            sample["location"] = fo.GeoLocation(
                point=[drone_image.longitude, drone_image.latitude]  # [lon, lat] format
            )

            # Add polygon if geographic footprint is available
            if drone_image.geo_polygon_points is not None:
                # Convert geographic bounds to polygon points
                polygon = [[lon, lat] for lon, lat in drone_image.geo_polygon_points]

                # Update GeoLocation with polygon
                sample["location"] = fo.GeoLocation(
                    point=[drone_image.longitude, drone_image.latitude],
                    polygon=[polygon],
                )

        # Add other metadata
        if drone_image.gsd is not None:
            sample["gsd"] = drone_image.gsd

        if drone_image.timestamp is not None:
            sample["timestamp"] = drone_image.timestamp

        # Add species counts
        species_counts = {}
        for detection in all_detections:
            if not detection.is_empty:
                species = detection.class_name
                species_counts[species] = species_counts.get(species, 0) + 1
        sample["species_counts"] = species_counts

        self.dataset.add_sample(sample)
        logger.info(
            f"Added drone image with {len(fo_detections)} detections: {drone_image.image_path}"
        )

    def add_drone_images(self, drone_images: List[DroneImage]):
        """Add multiple DroneImage objects to the dataset.

        Args:
            drone_images: List of DroneImage objects to add
        """
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        samples = []
        total_detections = 0

        for drone_image in drone_images:
            # Get all detections from the drone image
            self.add_drone_image(drone_image)

        self.dataset.add_samples(samples)
        logger.info(
            f"Added {len(samples)} drone images with {total_detections} total detections"
        )

    def get_detections_with_gps(self) -> List[fo.Sample]:
        """Get all samples that have GPS data.

        Returns:
            List of samples with GPS data
        """
        self._ensure_dataset_initialized()

        if self.dataset is None:
            return []

        return self.dataset.match(F("location").exists())

    @staticmethod
    def launch_app():
        """Launch the FiftyOne app."""
        import subprocess

        subprocess.Popen(
            ["uv", "run", "fiftyone", "app", "launch"],
            env=os.environ,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        self._ensure_dataset_initialized()

        if self.dataset is None:
            return {
                "name": "uninitialized",
                "num_samples": 0,
                "tags": [],
                "fields": [],
            }

        # Use count() method instead of len() for FiftyOne datasets
        num_samples = self.dataset.count()

        return {
            "name": self.dataset.name,
            "num_samples": num_samples,
            "tags": [],  # FiftyOne doesn't have get_tags() method
            "fields": list(self.dataset.get_field_schema().keys()),
        }

    def compute_similarity(self):
        """Compute similarity between samples using FiftyOne Brain."""
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        try:
            if self.config.get("fiftyone", {}).get("enable_brain", False):
                fob.compute_similarity(self.dataset, "detections")
                logger.info("Computed similarity embeddings")
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")

    def find_hardest_samples(self, num_samples: int = 100):
        """Find the most challenging samples for annotation."""
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return None

        try:
            if self.config.get("fiftyone", {}).get("enable_brain", False):
                # Use a different approach since compute_hardest doesn't exist
                # Find samples with low confidence detections
                low_confidence = self.dataset.match(
                    F("detections.detections.confidence") < 0.5
                ).limit(num_samples)
                logger.info(f"Found {len(low_confidence)} low confidence samples")
                return low_confidence
        except Exception as e:
            logger.error(f"Error finding hardest samples: {e}")
        return None

    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about annotations in the dataset."""
        self._ensure_dataset_initialized()

        if self.dataset is None:
            return {
                "total_samples": 0,
                "annotated_samples": 0,
                "total_detections": 0,
                "species_counts": {},
            }

        # Use count() method instead of len() for FiftyOne datasets
        total_samples = self.dataset.count()
        annotated_samples = self.dataset.match(F("detections").exists()).count()

        stats = {
            "total_samples": total_samples,
            "annotated_samples": annotated_samples,
            "total_detections": 0,
            "species_counts": {},
        }

        # Count detections and species
        for sample in self.dataset:
            if "detections" in sample:
                detections = sample["detections"]
                if detections:
                    stats["total_detections"] += len(detections.detections)

                    for detection in detections.detections:
                        species = detection.label
                        stats["species_counts"][species] = (
                            stats["species_counts"].get(species, 0) + 1
                        )

        return stats

    def save_dataset(self):
        """Save the dataset to disk."""
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        try:
            self.dataset.save()
            logger.info("Dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

    def close(self):
        """Close the dataset."""
        if self.dataset:
            self.dataset.close()
            logger.info("Dataset closed")
