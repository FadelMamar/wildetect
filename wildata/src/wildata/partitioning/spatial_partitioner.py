"""
Spatial partitioner for handling GPS coordinates and spatial metadata.

This module provides functionality to extract and use spatial information
from image metadata for robust train-val-test partitioning.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..logging_config import get_logger
from .strategies import SpatialGroupShuffleSplit
from .utils import GPSUtils, get_timestamp


# TODO: manual tests
class SpatialPartitioner:
    """
    Spatial partitioner for handling GPS coordinates and spatial metadata.

    This class extracts spatial information from image files (EXIF) and uses it
    to create meaningful groups for train-val-test splitting that respect
    spatial autocorrelation.
    """

    def __init__(
        self,
        spatial_threshold: float = 0.01,  # degrees for GPS coordinates
        clustering_method: str = "dbscan",
        gps_keys: Optional[List[str]] = None,  # Deprecated, kept for compatibility
        metadata_keys: Optional[List[str]] = None,
    ):
        """
        Initialize spatial partitioner.

        Args:
            spatial_threshold: Distance threshold for spatial grouping (degrees)
            clustering_method: Method for spatial clustering ('dbscan', 'grid')
            gps_keys: Keys to look for GPS coordinates in metadata
            metadata_keys: Additional metadata keys to consider
        """
        self.spatial_threshold = spatial_threshold
        self.clustering_method = clustering_method
        self.gps_keys = gps_keys or [
            "gps_lat",
            "gps_lon",
            "latitude",
            "longitude",
            "lat",
            "lon",
        ]
        self.metadata_keys = metadata_keys or []
        self.logger = get_logger(__name__)

    def extract_spatial_coordinates(
        self,
        images: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract spatial coordinates from image files using EXIF GPS data.
        Args:
            images: List of image dictionaries from COCO format
            metadata: (Unused, kept for compatibility)
        Returns:
            Tuple of (coordinates array, valid indices)
        """
        coordinates = []
        valid_indices = []
        for i, image in enumerate(images):
            file_name = image.get("file_name")
            if not file_name:
                self.logger.warning(f"Image {i} missing file_name, skipping.")
                continue
            try:
                coords = GPSUtils.get_gps_coord(file_name)
            except Exception as e:
                self.logger.warning(f"Error extracting GPS from {file_name}: {e}")
                coords = None
            if coords is not None:
                coordinates.append(coords)
                valid_indices.append(i)
            else:
                self.logger.warning(
                    f"No GPS coordinates found in EXIF for image {file_name} (index {i})"
                )
        if not coordinates:
            raise ValueError("No spatial coordinates found in any images (via EXIF)")
        return np.array(coordinates), valid_indices

    def create_spatial_splitter(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> SpatialGroupShuffleSplit:
        """
        Create a spatial-aware splitter.

        Args:
            test_size: Proportion of groups for test set
            val_size: Proportion of groups for validation set
            random_state: Random state for reproducibility

        Returns:
            Configured SpatialGroupShuffleSplit instance
        """
        return SpatialGroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            train_size=1.0 - test_size - val_size,
            random_state=random_state,
            spatial_threshold=self.spatial_threshold,
            clustering_method=self.clustering_method,
        )

    def partition_dataset(
        self,
        images: List[Dict[str, Any]],
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        Partition dataset using spatial information.

        Args:
            images: List of image dictionaries from COCO format
            metadata: Optional list of metadata dictionaries
            test_size: Proportion of groups for test set
            val_size: Proportion of groups for validation set
            random_state: Random state for reproducibility

        Returns:
            Dictionary mapping split names to image indices
        """
        # Extract spatial coordinates
        coordinates, valid_indices = self.extract_spatial_coordinates(images)

        # Create spatial splitter
        splitter = self.create_spatial_splitter(test_size, val_size, random_state)

        # Fit the splitter
        splitter.fit(
            X=np.zeros(len(coordinates)),  # Dummy features
            spatial_coords=coordinates,
        )

        # Generate splits
        splits = list(
            splitter.split(X=np.zeros(len(coordinates)), groups=splitter.groups_)
        )
        train_idx, test_val_idx = splits[0]

        # Further split test_val into test and validation
        if val_size > 0:
            val_splitter = SpatialGroupShuffleSplit(
                n_splits=1,
                test_size=val_size / (test_size + val_size),
                random_state=random_state,
                spatial_threshold=self.spatial_threshold,
                clustering_method=self.clustering_method,
            )

            # Get groups for test_val subset
            test_val_coords = coordinates[test_val_idx]
            test_val_groups = splitter.groups_[test_val_idx]

            val_splitter.fit(
                X=np.zeros(len(test_val_coords)), spatial_coords=test_val_coords
            )

            val_splits = list(
                val_splitter.split(
                    X=np.zeros(len(test_val_coords)), groups=val_splitter.groups_
                )
            )
            val_train_idx, val_test_idx = val_splits[0]

            # Map back to original indices
            test_idx = test_val_idx[val_test_idx]
            val_idx = test_val_idx[val_train_idx]
        else:
            test_idx = test_val_idx
            val_idx = []

        # Map back to original image indices
        result = {
            "train": [valid_indices[i] for i in train_idx],
            "val": [valid_indices[i] for i in val_idx],
            "test": [valid_indices[i] for i in test_idx],
        }

        self.logger.info(
            f"Spatial partitioning complete: {len(result['train'])} train, "
            f"{len(result['val'])} val, {len(result['test'])} test images"
        )

        return result

    def get_spatial_statistics(
        self,
        images: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get spatial statistics for the dataset.

        Args:
            images: List of image dictionaries
            metadata: Optional list of metadata dictionaries

        Returns:
            Dictionary with spatial statistics
        """
        try:
            coordinates, valid_indices = self.extract_spatial_coordinates(images)

            stats = {
                "total_images": len(images),
                "images_with_coordinates": len(coordinates),
                "coverage_percentage": len(coordinates) / len(images) * 100,
                "spatial_bounds": {
                    "min_lat": float(np.min(coordinates[:, 0])),
                    "max_lat": float(np.max(coordinates[:, 0])),
                    "min_lon": float(np.min(coordinates[:, 1])),
                    "max_lon": float(np.max(coordinates[:, 1])),
                },
                "spatial_span": {
                    "lat_span": float(
                        np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
                    ),
                    "lon_span": float(
                        np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
                    ),
                },
            }

            return stats
        except Exception as e:
            self.logger.error(f"Error computing spatial statistics: {e}")
            return {"error": str(e)}

    def extract_timestamps(
        self,
        images: List[Dict[str, Any]],
    ) -> List[Optional[str]]:
        """
        Extract timestamps from image files using EXIF data.
        Returns a list of timestamps (or None if not found) for each image.
        """
        timestamps = []
        for i, image in enumerate(images):
            file_name = image.get("file_name")
            if not file_name:
                timestamps.append(None)
                continue
            try:
                ts = get_timestamp(file_name)
            except Exception as e:
                self.logger.warning(f"Error extracting timestamp from {file_name}: {e}")
                ts = None
            timestamps.append(ts)
        return timestamps
