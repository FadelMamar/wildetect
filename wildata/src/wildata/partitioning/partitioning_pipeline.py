"""
Main partitioning pipeline for orchestrating train-val-test splits.

This module provides a high-level interface for partitioning datasets using
various strategies (spatial, camp-based, metadata-based) and integrates
with the existing data pipeline.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..logging_config import get_logger
from .camp_partitioner import CampPartitioner
from .spatial_partitioner import SpatialPartitioner


# TODO: manual tests
class PartitioningStrategy(Enum):
    """Available partitioning strategies."""

    SPATIAL = "spatial"
    CAMP_BASED = "camp_based"
    METADATA_BASED = "metadata_based"
    HYBRID = "hybrid"


class PartitioningPipeline:
    """
    Main partitioning pipeline for orchestrating train-val-test splits.

    This class provides a high-level interface for partitioning datasets using
    various strategies and integrates with the existing data pipeline.
    """

    def __init__(
        self,
        strategy: Union[str, PartitioningStrategy] = PartitioningStrategy.SPATIAL,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize partitioning pipeline.

        Args:
            strategy: Partitioning strategy to use
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for specific partitioners
        """
        self.strategy = (
            PartitioningStrategy(strategy) if isinstance(strategy, str) else strategy
        )
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.logger = get_logger(__name__)

        # Initialize partitioners based on strategy
        self._initialize_partitioners(**kwargs)

    def _initialize_partitioners(self, **kwargs):
        """Initialize partitioners based on the selected strategy."""
        self.spatial_partitioner = None
        self.camp_partitioner = None

        if self.strategy in [PartitioningStrategy.SPATIAL, PartitioningStrategy.HYBRID]:
            self.spatial_partitioner = SpatialPartitioner(
                spatial_threshold=kwargs.get("spatial_threshold", 0.01),
                clustering_method=kwargs.get("clustering_method", "dbscan"),
                gps_keys=kwargs.get("gps_keys"),
                metadata_keys=kwargs.get("spatial_metadata_keys"),
            )

        if self.strategy in [
            PartitioningStrategy.CAMP_BASED,
            PartitioningStrategy.HYBRID,
        ]:
            self.camp_partitioner = CampPartitioner(
                camp_metadata_key=kwargs.get("camp_metadata_key", "camp_id"),
                fallback_keys=kwargs.get("camp_fallback_keys"),
                create_individual_groups=kwargs.get("create_individual_groups", True),
            )

    def partition_dataset(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, List[int]]:
        """
        Partition dataset using the selected strategy.

        Args:
            images: List of image dictionaries from COCO format
            metadata: Optional list of metadata dictionaries
            **kwargs: Additional arguments for specific strategies

        Returns:
            Dictionary mapping split names to image indices
        """
        self.logger.info(f"Starting partitioning with strategy: {self.strategy.value}")

        if self.strategy == PartitioningStrategy.SPATIAL:
            return self._partition_spatial(images, metadata, **kwargs)
        elif self.strategy == PartitioningStrategy.CAMP_BASED:
            return self._partition_camp_based(images, metadata, **kwargs)
        elif self.strategy == PartitioningStrategy.HYBRID:
            return self._partition_hybrid(images, metadata, **kwargs)
        else:
            raise ValueError(f"Unknown partitioning strategy: {self.strategy}")

    def _partition_spatial(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, List[int]]:
        """Partition using spatial strategy."""
        if self.spatial_partitioner is None:
            raise ValueError("Spatial partitioner not initialized")

        return self.spatial_partitioner.partition_dataset(
            images=images,
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
        )

    def _partition_camp_based(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, List[int]]:
        """Partition using camp-based strategy."""
        if self.camp_partitioner is None:
            raise ValueError("Camp partitioner not initialized")

        return self.camp_partitioner.partition_dataset(
            images=images,
            metadata=metadata,
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
        )

    def _partition_hybrid(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, List[int]]:
        """Partition using hybrid strategy (combines multiple strategies)."""
        # Try spatial first, fall back to camp-based, then metadata-based
        strategies_to_try = []

        if self.spatial_partitioner:
            strategies_to_try.append(("spatial", self.spatial_partitioner))
        if self.camp_partitioner:
            strategies_to_try.append(("camp_based", self.camp_partitioner))

        if not strategies_to_try:
            raise ValueError("No partitioners available for hybrid strategy")

        # Try each strategy until one works
        for strategy_name, partitioner in strategies_to_try:
            try:
                self.logger.info(
                    f"Trying {strategy_name} strategy for hybrid partitioning"
                )

                if strategy_name == "spatial":
                    result = self.spatial_partitioner.partition_dataset(
                        images=images,
                        test_size=self.test_size,
                        val_size=self.val_size,
                        random_state=self.random_state,
                    )
                elif strategy_name == "camp_based":
                    result = self.camp_partitioner.partition_dataset(
                        images=images,
                        metadata=metadata,
                        test_size=self.test_size,
                        val_size=self.val_size,
                        random_state=self.random_state,
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")

                self.logger.info(
                    f"Hybrid partitioning successful using {strategy_name} strategy"
                )
                return result

            except Exception as e:
                self.logger.warning(f"{strategy_name} strategy failed: {e}")
                continue

    def get_statistics(
        self,
        images: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the dataset.

        Args:
            images: List of image dictionaries
            metadata: Optional list of metadata dictionaries

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {
            "strategy": self.strategy.value,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "total_images": len(images),
        }

        # Get statistics from each available partitioner
        if self.spatial_partitioner:
            try:
                spatial_stats = self.spatial_partitioner.get_spatial_statistics(images)
                stats["spatial"] = spatial_stats
            except Exception as e:
                stats["spatial"] = {"error": str(e)}

        if self.camp_partitioner:
            try:
                camp_stats = self.camp_partitioner.get_camp_statistics(images, metadata)
                stats["camp_based"] = camp_stats
            except Exception as e:
                stats["camp_based"] = {"error": str(e)}

        return stats

    def apply_partitioning_to_coco_data(
        self,
        coco_data: Dict[str, Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply partitioning to COCO format data and return split data.

        Args:
            coco_data: COCO format data dictionary
            metadata: Optional list of metadata dictionaries
            **kwargs: Additional arguments for partitioning

        Returns:
            Dictionary mapping split names to COCO format data
        """
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        # Get partitioning indices
        split_indices = self.partition_dataset(images, metadata, **kwargs)

        # Create split data
        split_data = {}
        for split_name, indices in split_indices.items():
            # Get images for this split
            split_images = [images[i] for i in indices]

            # Get annotations for this split
            split_image_ids = {img["id"] for img in split_images}
            split_annotations = [
                ann for ann in annotations if ann["image_id"] in split_image_ids
            ]

            # Create COCO format data for this split
            split_data[split_name] = {
                "images": split_images,
                "annotations": split_annotations,
                "categories": categories,
            }

        self.logger.info(
            f"Applied partitioning to COCO data: {len(split_data)} splits created"
        )
        return split_data

    def save_partitioning_config(
        self,
        output_path: Union[str, Path],
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Save partitioning configuration to file.

        Args:
            output_path: Path to save configuration
            additional_info: Additional information to include
        """
        config = {
            "strategy": self.strategy.value,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
        }

        if additional_info:
            config.update(additional_info)

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Partitioning configuration saved to {output_path}")

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "PartitioningPipeline":
        """
        Create partitioning pipeline from configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configured PartitioningPipeline instance
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(**config)
