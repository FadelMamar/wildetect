"""
Partitioning strategies for handling spatial autocorrelation in aerial imagery data.

This module provides specialized splitting strategies that extend scikit-learn's
GroupShuffleSplit to handle spatial autocorrelation, camp-based grouping, and
metadata-based partitioning.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit

from ..logging_config import get_logger


# TODO: manual tests
class SpatialGroupShuffleSplit(GroupShuffleSplit):
    """
    Spatial-aware GroupShuffleSplit that handles spatial autocorrelation.

    This strategy groups spatially close images together to prevent data leakage
    between train and validation sets. It uses GPS coordinates or spatial metadata
    to create meaningful groups.
    """

    def __init__(
        self,
        n_splits: int = 1,
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        spatial_threshold: float = 0.01,  # degrees for GPS coordinates
        clustering_method: str = "dbscan",
    ):
        """
        Initialize spatial group shuffle split.

        Args:
            n_splits: Number of splits to generate
            test_size: Proportion of groups for test set
            train_size: Proportion of groups for train set
            random_state: Random state for reproducibility
            spatial_threshold: Distance threshold for spatial grouping (degrees)
            clustering_method: Method for spatial clustering ('dbscan', 'grid')
        """
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.spatial_threshold = spatial_threshold
        self.clustering_method = clustering_method
        self.logger = get_logger(__name__)

    def fit(self, X, y=None, groups=None, spatial_coords=None):
        """
        Fit the spatial splitter.

        Args:
            X: Features (not used in spatial splitting)
            y: Target values
            groups: Pre-defined groups (optional)
            spatial_coords: Array of [lat, lon] coordinates

        Returns:
            self
        """
        if spatial_coords is not None and groups is None:
            # Create spatial groups based on coordinates
            groups = self._create_spatial_groups(spatial_coords)
        # No need to set self.groups_ here; GroupShuffleSplit does not use it
        return self

    def _create_spatial_groups(self, spatial_coords: np.ndarray) -> np.ndarray:
        """
        Create spatial groups using clustering.

        Args:
            spatial_coords: Array of [lat, lon] coordinates

        Returns:
            Array of group labels (np.ndarray of int)
        """
        if self.clustering_method == "dbscan":
            # Use DBSCAN for spatial clustering
            clustering = DBSCAN(
                eps=self.spatial_threshold, min_samples=1, metric="euclidean"
            )
            groups = clustering.fit_predict(spatial_coords)
        elif self.clustering_method == "grid":
            # Simple grid-based grouping
            groups = self._grid_based_grouping(spatial_coords)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        groups = np.array(groups, dtype=int).flatten()
        n_unique = len(set(groups.tolist()))
        self.logger.info(f"Created {n_unique} spatial groups")
        return groups

    def _grid_based_grouping(self, spatial_coords: np.ndarray) -> np.ndarray:
        """
        Create groups based on grid cells.

        Args:
            spatial_coords: Array of [lat, lon] coordinates

        Returns:
            Array of group labels
        """
        # Create grid cells based on spatial threshold
        lat_coords = spatial_coords[:, 0]
        lon_coords = spatial_coords[:, 1]

        # Calculate grid cell indices
        lat_cells = np.floor(lat_coords / self.spatial_threshold).astype(int)
        lon_cells = np.floor(lon_coords / self.spatial_threshold).astype(int)

        # Create unique group labels
        groups = lat_cells * 10000 + lon_cells  # Simple hash for unique groups
        return groups


# TODO: manual tests
class CampBasedSplit(GroupShuffleSplit):
    """
    Camp-based splitting strategy for wildlife camp areas, with optional label-stratified group assignment.
    Groups images by camp areas to ensure that images from the same camp area stay together in train/val/test splits, preventing spatial data leakage.
    Optionally balances label (category) distribution across splits using a greedy group-level stratified assignment, similar to GroupStratifiedSpatialGroupShuffleSplit.
    """

    def __init__(
        self,
        n_splits: int = 1,
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        camp_metadata_key: str = "camp_id",
        stratify: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize camp-based split.
        Args:
            n_splits: Number of splits to generate
            test_size: Proportion of camps for test set
            train_size: Proportion of camps for train set
            random_state: Random state for reproducibility
            camp_metadata_key: Key for camp identifier in metadata
            stratify: Whether to balance label distribution across splits
            logger: Optional logger
        """
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.camp_metadata_key = camp_metadata_key
        self.stratify = stratify
        self.logger = logger or get_logger(__name__)

    def fit(self, X, y=None, groups=None, metadata=None):
        """
        Fit the camp-based splitter.
        Args:
            X: Features (not used in camp-based splitting)
            y: Target values
            groups: Pre-defined groups (optional)
            metadata: List of metadata dictionaries with camp information
        Returns:
            self
        """
        if metadata is not None and groups is None:
            # Create camp-based groups from metadata
            groups = self._create_camp_groups(metadata)
        # No need to set self.groups_ here; GroupShuffleSplit does not use it
        return self

    def _create_camp_groups(self, metadata: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create groups based on camp identifiers.
        Args:
            metadata: List of metadata dictionaries
        Returns:
            Array of group labels (np.ndarray of int)
        """
        groups = []
        camp_id_to_group = {}
        group_counter = 0
        for meta in metadata:
            camp_id = meta.get(self.camp_metadata_key)
            if camp_id is None:
                # If no camp_id, create individual group
                groups.append(group_counter)
                group_counter += 1
            else:
                # Use existing group for this camp or create new one
                if camp_id not in camp_id_to_group:
                    camp_id_to_group[camp_id] = group_counter
                    group_counter += 1
                groups.append(camp_id_to_group[camp_id])
        groups = np.array(groups, dtype=int).flatten()
        n_unique = len(set(groups.tolist()))
        self.logger.info(f"Created {n_unique} camp-based groups")
        return groups

    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Split groups into train/val/test splits, balancing label distribution if requested.
        Args:
            X: Dummy features (not used)
            y: Not used
            groups: Group assignment for each sample (e.g., camp id)
            annotations: List of all COCO annotations
            labels: List of all category ids
        Yields:
            train_idx, test_idx (indices into X)
        """
        if not self.stratify or annotations is None or labels is None or groups is None:
            # Fallback to parent split
            gss = GroupShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size if self.test_size is not None else 0.2,
                train_size=self.train_size if self.train_size is not None else 0.7,
                random_state=self.random_state,
            )
            yield from gss.split(X, y, groups)
            return
        # 1. Build group -> image indices mapping
        group_to_indices = {}
        for idx, g in enumerate(groups):
            group_to_indices.setdefault(g, []).append(idx)
        group_ids = list(group_to_indices.keys())
        # 2. Build group -> label histogram
        idx_to_imgid = {
            idx: kwargs.get("image_ids", [])[idx] if "image_ids" in kwargs else idx
            for idx in range(len(X))
        }
        group_label_counts = {g: {l: 0 for l in labels} for g in group_ids}
        imgid_to_group = {}
        for g, idxs in group_to_indices.items():
            for idx in idxs:
                imgid = idx_to_imgid[idx]
                imgid_to_group[imgid] = g
        for ann in annotations:
            imgid = ann["image_id"]
            catid = ann["category_id"]
            g = imgid_to_group.get(imgid)
            if g is not None and catid in labels:
                group_label_counts[g][catid] += 1
        # 3. Shuffle groups for randomness
        rng = np.random.RandomState(self.random_state)
        shuffled_groups = group_ids[:]
        rng.shuffle(shuffled_groups)
        # 4. Greedy assignment to splits
        n_total = len(shuffled_groups)
        test_size = self.test_size if self.test_size is not None else 0.2
        train_size = self.train_size if self.train_size is not None else 0.7
        n_test = int(round(test_size * n_total))
        n_train = int(round(train_size * n_total))
        n_val = n_total - n_train - n_test
        split_targets = [n_train, n_val, n_test]
        split_names = ["train", "val", "test"]
        splits = {name: [] for name in split_names}
        split_label_counts = {name: {l: 0 for l in labels} for name in split_names}
        for g in shuffled_groups:
            best_split = None
            best_score = None
            for split_idx, split_name in enumerate(split_names):
                if len(splits[split_name]) >= split_targets[split_idx]:
                    continue
                temp_counts = split_label_counts[split_name].copy()
                for l in labels:
                    temp_counts[l] += group_label_counts[g][l]
                total = sum(temp_counts.values())
                mean = total / len(labels) if labels else 1
                imbalance = sum(abs(temp_counts[l] - mean) for l in labels)
                if best_score is None or imbalance < best_score:
                    best_score = imbalance
                    best_split = split_name
            if best_split is None:
                best_split = min(split_names, key=lambda s: len(splits[s]))
            splits[best_split].append(g)
            for l in labels:
                split_label_counts[best_split][l] += group_label_counts[g][l]
        split_indices = {name: [] for name in split_names}
        for split_name, group_list in splits.items():
            for g in group_list:
                split_indices[split_name].extend(group_to_indices[g])
        yield split_indices["train"], split_indices["test"]


class GroupStratifiedSpatialGroupShuffleSplit(SpatialGroupShuffleSplit):
    """
    Greedy group-level stratified splitter for spatial groups.
    Assigns groups to splits to balance label (category) distribution as much as possible,
    while respecting group boundaries (no group is split across splits).
    """

    def __init__(
        self,
        n_splits: int = 1,
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        spatial_threshold: float = 0.01,
        clustering_method: str = "dbscan",
        stratify: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            spatial_threshold=spatial_threshold,
            clustering_method=clustering_method,
        )
        self.stratify = stratify
        self.logger = logger or get_logger(__name__)

    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Split groups into train/val/test splits, balancing label distribution.
        Args:
            X: Dummy features (not used)
            y: Not used
            groups: Group assignment for each sample (e.g., cluster id)
            annotations: List of all COCO annotations
            labels: List of all category ids
        Yields:
            train_idx, test_idx (indices into X)
        """
        if not self.stratify or annotations is None or labels is None or groups is None:
            # Fallback to parent split
            yield from super().split(X, y, groups)
            return

        # 1. Build group -> image indices mapping
        group_to_indices = {}
        for idx, g in enumerate(groups):
            group_to_indices.setdefault(g, []).append(idx)
        group_ids = list(group_to_indices.keys())

        # 2. Build group -> label histogram
        # Map image index to group
        img_idx_to_group = {
            idx: g for g, idxs in group_to_indices.items() for idx in idxs
        }
        # Map image id to index
        imgid_to_idx = {i: idx for idx, i in enumerate([a for a in range(len(X))])}
        # Map image index to image id (COCO)
        idx_to_imgid = {
            idx: kwargs.get("image_ids", [])[idx] if "image_ids" in kwargs else idx
            for idx in range(len(X))
        }
        # Build group label histograms
        group_label_counts = {g: {l: 0 for l in labels} for g in group_ids}
        # Map image id to group
        imgid_to_group = {}
        for g, idxs in group_to_indices.items():
            for idx in idxs:
                imgid = idx_to_imgid[idx]
                imgid_to_group[imgid] = g
        for ann in annotations:
            imgid = ann["image_id"]
            catid = ann["category_id"]
            g = imgid_to_group.get(imgid)
            if g is not None and catid in labels:
                group_label_counts[g][catid] += 1
        # 3. Shuffle groups for randomness
        rng = np.random.RandomState(self.random_state)
        shuffled_groups = group_ids[:]
        rng.shuffle(shuffled_groups)
        # 4. Greedy assignment to splits
        n_total = len(shuffled_groups)
        test_size = self.test_size if self.test_size is not None else 0.2
        train_size = self.train_size if self.train_size is not None else 0.7
        n_test = int(round(test_size * n_total))
        n_train = int(round(train_size * n_total))
        n_val = n_total - n_train - n_test
        split_targets = [n_train, n_val, n_test]
        split_names = ["train", "val", "test"]
        splits = {name: [] for name in split_names}
        split_label_counts = {name: {l: 0 for l in labels} for name in split_names}
        # Assign groups one by one
        for g in shuffled_groups:
            # Find split with lowest max label imbalance
            best_split = None
            best_score = None
            for split_idx, split_name in enumerate(split_names):
                if len(splits[split_name]) >= split_targets[split_idx]:
                    continue
                # Simulate adding this group
                temp_counts = split_label_counts[split_name].copy()
                for l in labels:
                    temp_counts[l] += group_label_counts[g][l]
                # Compute imbalance (max deviation from mean)
                total = sum(temp_counts.values())
                mean = total / len(labels) if labels else 1
                imbalance = sum(abs(temp_counts[l] - mean) for l in labels)
                if best_score is None or imbalance < best_score:
                    best_score = imbalance
                    best_split = split_name
            if best_split is None:
                # All splits full, assign to smallest
                best_split = min(split_names, key=lambda s: len(splits[s]))
            splits[best_split].append(g)
            for l in labels:
                split_label_counts[best_split][l] += group_label_counts[g][l]
        # 5. Collect indices for each split
        split_indices = {name: [] for name in split_names}
        for split_name, group_list in splits.items():
            for g in group_list:
                split_indices[split_name].extend(group_to_indices[g])
        # Only yield train/test for compatibility
        yield split_indices["train"], split_indices["test"]
