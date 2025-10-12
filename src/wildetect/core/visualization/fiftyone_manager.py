"""
FiftyOne integration for WildDetect.

This module handles dataset creation, visualization, and annotation collection
using FiftyOne for wildlife detection datasets.
"""

import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Union

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.utils.geojson as fogeojson
from fiftyone import ViewField as F

from ..config import ROOT
from ..data.drone_image import DroneImage

logger = logging.getLogger(__name__)


class FiftyOneManager:
    """Manages FiftyOne datasets for wildlife detection."""

    def __init__(
        self,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
        persistent: bool = True,
    ):
        """Initialize FiftyOne manager.

        Args:
            dataset_name: Name of the dataset to use
            config: Optional configuration override
        """
        self.config = config
        self.dataset_name = dataset_name
        self.dataset = None
        self.persistent = persistent

        self.prediction_field = "detections"

        self._init_dataset()

    def _init_dataset(self):
        """Initialize or load the FiftyOne dataset."""
        try:
            # Try to load existing dataset
            self.dataset = fo.load_dataset(self.dataset_name)
            logger.info(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            # Create new dataset
            self.dataset = fo.Dataset(self.dataset_name, persistent=self.persistent)
            logger.info(f"Created new dataset: {self.dataset_name}")

    def _ensure_dataset_initialized(self):
        """Ensure dataset is initialized before operations."""
        if self.dataset is None:
            self._init_dataset()

    def _create_fo_sample(self, drone_image: DroneImage):
        """Add a DroneImage to the dataset.

        Args:
            drone_image: DroneImage object to add
        """
        self._ensure_dataset_initialized()

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        # Get all detections from the drone image
        all_detections = drone_image.get_non_empty_predictions()

        # Convert to FiftyOne format
        fo_detections = []
        for detection in all_detections:
            fo_detection = detection.to_fiftyone(
                image_width=drone_image.width, image_height=drone_image.height
            )
            fo_detections.append(fo_detection)

        # Create sample with metadata
        sample = fo.Sample(filepath=drone_image.image_path)

        if fo_detections:
            sample[self.prediction_field] = fo.Detections(detections=fo_detections)

        # Add metadata
        sample["total_count"] = len(fo_detections)
        sample["num_tiles"] = len(drone_image.tiles)
        sample["image_width"] = drone_image.width
        sample["image_height"] = drone_image.height

        # Add native FiftyOne geolocation if GPS data is available
        if drone_image.latitude is not None and drone_image.longitude is not None:
            # Create GeoLocation with point coordinates
            gps = [drone_image.longitude, drone_image.latitude]
            geo_json = fogeojson.parse_point(gps)
            sample["gps"] = fo.GeoLocation.from_geo_json(geo_json)
            logger.debug(f"adding geo location to sample: {sample['filepath']}")

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

        logger.debug(
            f"Created sample for drone image with {len(fo_detections)} detections: {drone_image.image_path}"
        )
        return sample

    def add_drone_images(self, drone_images: List[DroneImage]):
        """Add multiple DroneImage objects to the dataset.

        Args:
            drone_images: List of DroneImage objects to add
        """
        self._ensure_dataset_initialized()

        assert isinstance(drone_images, list), "drone_images must be a list"
        for drone_image in drone_images:
            assert isinstance(
                drone_image, DroneImage
            ), "drone_images must be a list of DroneImage objects"

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        samples = []
        total_detections = 0
        for drone_image in drone_images:
            sample = self._create_fo_sample(drone_image)
            samples.append(sample)
            total_detections += sample["total_count"]

        self.dataset.add_samples(samples)
        self.dataset.save()
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

        return self.dataset.match(F("gps").exists())

    @staticmethod
    def launch_app():
        """Launch the FiftyOne app."""
        import subprocess
        import sys

        # Cross-platform subprocess creation
        if sys.platform == "win32":
            subprocess.Popen(
                ["uv", "run", "fiftyone", "app", "launch"],
                env=os.environ.copy(),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            subprocess.Popen(
                ["uv", "run", "fiftyone", "app", "launch"],
                env=os.environ.copy(),
            )

    def send_predictions_to_labelstudio(
        self, annot_key: str, dotenv_path: Optional[str] = None
    ):
        """Launch the FiftyOne annotation app."""
        if dotenv_path is not None:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path, override=True)

        with open(ROOT / "config/class_mapping.json", "r", encoding="utf-8") as f:
            label_map = json.load(f)

        classes = list(label_map.values())

        try:
            dataset = fo.load_dataset(self.dataset_name)
            dataset.annotate(
                annot_key,
                backend="labelstudio",
                label_field=self.prediction_field,
                label_type="detections",
                classes=classes,
                api_key=os.environ["LABEL_STUDIO_API_KEY"],
                url=os.environ["LABEL_STUDIO_URL"],
            )
        except Exception:
            logger.error(f"Error exporting to LabelStudio: {traceback.format_exc()}")
            raise Exception(f"Error exporting to LabelStudio: {traceback.format_exc()}")

    # TODO: debug
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

        num_samples = self.dataset.count()

        return {
            "name": self.dataset.name,
            "num_samples": num_samples,
            "tags": [],  # FiftyOne doesn't have get_tags() method
            "fields": list(self.dataset.get_field_schema().keys()),
        }

    # TODO: debug
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

    # TODO: debug
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
