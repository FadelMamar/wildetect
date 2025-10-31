"""
Camp-based partitioner for wildlife camp areas.

This module provides functionality to group images by camp areas to ensure
that images from the same camp area stay together in train/val splits.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logging_config import get_logger
from .strategies import CampBasedSplit

# TODO: manual tests


class CampPartitioner:
    """
    Camp-based partitioner for wildlife camp areas.

    This class groups images by camp areas to ensure that images from the same
    camp area stay together in train/val splits, preventing spatial data leakage.
    """

    def __init__(
        self,
        camp_metadata_key: str = "camp_id",
        fallback_keys: Optional[List[str]] = None,
        create_individual_groups: bool = True,
    ):
        """
        Initialize camp partitioner.

        Args:
            camp_metadata_key: Key for camp identifier in metadata
            fallback_keys: Additional keys to try if camp_metadata_key not found
            create_individual_groups: Whether to create individual groups for images without camp info
        """
        self.camp_metadata_key = camp_metadata_key
        self.fallback_keys = fallback_keys or ["camp", "area", "region", "location"]
        self.create_individual_groups = create_individual_groups
        self.logger = get_logger(__name__)

    def extract_camp_groups(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Extract camp groups from image metadata.

        Args:
            images: List of image dictionaries from COCO format
            metadata: Optional list of metadata dictionaries

        Returns:
            Tuple of (camp groups, valid indices)
        """
        camp_groups = []
        valid_indices = []

        for i, image in enumerate(images):
            # Try to extract camp info from image metadata
            camp_id = self._extract_camp_id(image)

            if camp_id is None and metadata is not None and i < len(metadata):
                # Try metadata if available
                camp_id = self._extract_camp_id(metadata[i])

            if camp_id is not None:
                camp_groups.append(str(camp_id))
                valid_indices.append(i)
            elif self.create_individual_groups:
                # Create individual group for this image
                camp_groups.append(f"individual_{i}")
                valid_indices.append(i)
            else:
                self.logger.warning(f"No camp information found for image {i}")

        return camp_groups, valid_indices

    def _extract_camp_id(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract camp identifier from a dictionary.

        Args:
            data: Dictionary containing potential camp information

        Returns:
            Camp identifier or None if not found
        """
        # Try primary camp key
        if self.camp_metadata_key in data:
            camp_id = data[self.camp_metadata_key]
            if camp_id is not None:
                return str(camp_id)

        # Try fallback keys
        for key in self.fallback_keys:
            if key in data:
                camp_id = data[key]
                if camp_id is not None:
                    return str(camp_id)

        # Try nested structures
        if "metadata" in data:
            return self._extract_camp_id(data["metadata"])

        if "properties" in data:
            return self._extract_camp_id(data["properties"])

        return None

    def create_camp_splitter(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> CampBasedSplit:
        """
        Create a camp-based splitter.

        Args:
            test_size: Proportion of camps for test set
            val_size: Proportion of camps for validation set
            random_state: Random state for reproducibility

        Returns:
            Configured CampBasedSplit instance
        """
        return CampBasedSplit(
            n_splits=1,
            test_size=test_size,
            train_size=1.0 - test_size - val_size,
            random_state=random_state,
            camp_metadata_key=self.camp_metadata_key,
        )

    def partition_dataset(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        Partition dataset using camp information.

        Args:
            images: List of image dictionaries from COCO format
            metadata: Optional list of metadata dictionaries
            test_size: Proportion of camps for test set
            val_size: Proportion of camps for validation set
            random_state: Random state for reproducibility

        Returns:
            Dictionary mapping split names to image indices
        """
        # Extract camp groups
        camp_groups, valid_indices = self.extract_camp_groups(images, metadata)

        if not camp_groups:
            raise ValueError("No camp information found in any images")

        # Create metadata for camp-based splitting
        camp_metadata = [{"camp_id": camp_id} for camp_id in camp_groups]

        # Create camp splitter
        splitter = self.create_camp_splitter(test_size, val_size, random_state)

        # Fit the splitter
        splitter.fit(
            X=np.zeros(len(camp_groups)),  # Dummy features
            metadata=camp_metadata,
        )

        # Generate splits
        splits = list(
            splitter.split(X=np.zeros(len(camp_groups)), groups=splitter.groups_)
        )
        train_idx, test_val_idx = splits[0]

        # Further split test_val into test and validation
        if val_size > 0:
            val_splitter = CampBasedSplit(
                n_splits=1,
                test_size=val_size / (test_size + val_size),
                random_state=random_state,
                camp_metadata_key=self.camp_metadata_key,
            )

            # Get metadata for test_val subset
            test_val_metadata = [camp_metadata[i] for i in test_val_idx]

            val_splitter.fit(X=np.zeros(len(test_val_idx)), metadata=test_val_metadata)

            val_splits = list(
                val_splitter.split(
                    X=np.zeros(len(test_val_idx)), groups=val_splitter.groups_
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
            f"Camp-based partitioning complete: {len(result['train'])} train, "
            f"{len(result['val'])} val, {len(result['test'])} test images"
        )

        return result

    def get_camp_statistics(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Get camp statistics for the dataset.

        Args:
            images: List of image dictionaries
            metadata: Optional list of metadata dictionaries

        Returns:
            Dictionary with camp statistics
        """
        try:
            camp_groups, valid_indices = self.extract_camp_groups(images, metadata)

            if not camp_groups:
                return {"error": "No camp information found"}

            unique_camps = list(set(camp_groups))
            camp_counts = {}
            for camp in camp_groups:
                camp_counts[camp] = camp_counts.get(camp, 0) + 1

            stats = {
                "total_images": len(images),
                "images_with_camp_info": len(camp_groups),
                "coverage_percentage": len(camp_groups) / len(images) * 100,
                "unique_camps": len(unique_camps),
                "camp_distribution": camp_counts,
                "avg_images_per_camp": len(camp_groups) / len(unique_camps),
                "min_images_per_camp": min(camp_counts.values()) if camp_counts else 0,
                "max_images_per_camp": max(camp_counts.values()) if camp_counts else 0,
            }

            return stats
        except Exception as e:
            self.logger.error(f"Error computing camp statistics: {e}")
            return {"error": str(e)}
