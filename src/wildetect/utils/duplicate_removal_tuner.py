"""Hyperparameter tuning for duplicate detection removal using Optuna.

This module provides a tuner class that optimizes the GeographicMerger thresholds
to maximize duplicate detection F1-score based on ground truth duplicate annotations.

Usage:
    >>> from wildetect.utils.duplicate_removal_tuner import DuplicateRemovalTuner
    >>> tuner = DuplicateRemovalTuner(
    ...     csv_path="animal duplicates.csv",
    ...     ls_manager=ls_manager,
    ...     flight_specs=flight_specs,
    ... )
    >>> result = tuner.run()
    >>> print(f"Best IoU threshold: {result['best_iou_threshold']}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.spatial.distance import cdist

from wildetect.core.config import FlightSpecs
from wildetect.core.visualization.labelstudio_manager import LabelStudioManager

if TYPE_CHECKING:
    from wildetect.core.data import DroneImage
    from wildetect.core.data.detection import Detection

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate detections with centroid distance analysis.

    The detections in a group should represent the same animal appearing in
    multiple overlapping images. Methods are provided to compute statistics
    on how close the detection centroids are in UTM coordinates.
    """

    group_id: int
    bbox_ids: Set[str]
    task_ids: Set[int]
    species: str
    detections: List[Detection] = field(default_factory=list)

    def __post_init__(self):
        self.task_ids = set(map(int, self.task_ids))

    def add_detection(self, detection: Detection) -> None:
        """Add a detection to this group."""
        self.detections.append(detection)

    def get_geo_centroids(self) -> List[Tuple[float, float]]:
        """Get geographic centroids (UTM coordinates) for all detections.

        Returns:
            List of (easting, northing) tuples for each detection with a valid geo_box
        """
        centroids = []
        for det in self.detections:
            if det.geo_box is not None:
                # geo_box is [west, south, east, north] in UTM
                west, south, east, north = det.geo_box
                centroid_easting = (east + west) / 2
                centroid_northing = (north + south) / 2
                centroids.append((centroid_easting, centroid_northing))
        return centroids

    def compute_pairwise_distances(self) -> List[float]:
        """Compute pairwise Euclidean distances between detection centroids.

        Uses scipy.spatial.distance.cdist for efficient vectorized computation.

        Returns:
            List of distances (in meters if using UTM) between all pairs of centroids
        """

        centroids = self.get_geo_centroids()
        if len(centroids) < 2:
            return []

        # Convert to numpy array for cdist
        centroids_array = np.array(centroids)

        # Compute full distance matrix using Euclidean distance
        distance_matrix = cdist(centroids_array, centroids_array, metric="euclidean")

        # Extract upper triangle (excluding diagonal) to get unique pairwise distances
        upper_triangle_indices = np.triu_indices(len(centroids), k=1)
        distances = distance_matrix[upper_triangle_indices].tolist()

        return distances

    def get_distance_stats(self) -> Dict[str, float]:
        """Compute statistics on centroid distances within this duplicate group.

        Returns:
            Dict with:
                - mean_distance: Average pairwise distance (meters)
                - max_distance: Maximum pairwise distance (meters)
                - min_distance: Minimum pairwise distance (meters)
                - std_distance: Standard deviation of distances
                - num_detections: Number of detections with valid geo_box
                - num_pairs: Number of pairwise comparisons
        """
        distances = self.compute_pairwise_distances()
        centroids = self.get_geo_centroids()

        if len(distances) == 0:
            return {
                "mean_distance": float("nan"),
                "max_distance": float("nan"),
                "min_distance": float("nan"),
                "std_distance": float("nan"),
                "num_detections": len(centroids),
                "num_pairs": 0,
            }

        return {
            "mean_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
            "min_distance": float(np.min(distances)),
            "std_distance": float(np.std(distances)),
            "num_detections": len(centroids),
            "num_pairs": len(distances),
        }

    def get_centroid_spread(self) -> float:
        """Get the maximum spread (diameter) of centroids in meters.

        This represents the largest distance between any two detections
        in the group, useful for understanding how spread out the
        "same animal" detections are across images.

        Returns:
            Maximum distance between any two centroids, or NaN if < 2 detections
        """
        distances = self.compute_pairwise_distances()
        if len(distances) == 0:
            return float("nan")
        return float(np.max(distances))


class DuplicateRemovalTuner:
    """Tune GeographicMerger thresholds to maximize duplicate detection F1-score.

    Given a CSV of manually annotated duplicate groups (same animal appearing in
    multiple images), uses Optuna to find optimal thresholds for:
    1. iou_threshold: Detection-level IoU for duplicate matching
    2. min_overlap_threshold: Image-level IoU for considering images as overlapping

    The optimization objective is the F1-score of correctly identifying duplicates.
    """

    def __init__(
        self,
        csv_path: str,
        ls_manager: LabelStudioManager,
        flight_specs: FlightSpecs,
        iou_threshold_range: Tuple[float, float] = (-1.0, 1.0),
        min_overlap_threshold_range: Tuple[float, float] = (0.0, 0.1),
        n_trials: int = 50,
        verbose: bool = False,
    ):
        """Initialize the duplicate removal tuner.

        Args:
            csv_path: Path to CSV file with duplicate annotations.
                Expected columns: image, bounding box, species, duplicate
            ls_manager: LabelStudioManager instance for fetching task data
            flight_specs: Flight specifications for DroneImage creation
            iou_threshold_range: Search range for detection-level IoU threshold
            min_overlap_threshold_range: Search range for image-level overlap threshold
            n_trials: Number of Optuna optimization trials
        """
        self.csv_path = csv_path
        self.ls_manager = ls_manager
        self.flight_specs = flight_specs
        self.iou_threshold_range = iou_threshold_range
        self.min_overlap_threshold_range = min_overlap_threshold_range
        self.n_trials = n_trials
        self.study: Optional[optuna.Study] = None
        self.verbose = verbose
        # Load and parse CSV
        self.df = pd.read_csv(csv_path).rename(
            columns={"image": "task_id", "bounding box": "bbox_id"}
        )
        self.df["task_id"] = self.df["task_id"].apply(int)

        # Parse duplicate groups
        self.duplicate_groups = self._parse_duplicate_groups()
        logger.info(
            f"Loaded {len(self.duplicate_groups)} duplicate groups "
            f"with {len(self.df)} total annotations"
        )

        # Cache for DroneImage objects to avoid re-fetching
        self._task_cache: Dict[int, Any] = {}

    def _parse_duplicate_groups(self) -> List[DuplicateGroup]:
        """Parse CSV into DuplicateGroup objects."""
        groups = []
        for group_id, group_df in self.df.groupby("duplicate"):
            groups.append(
                DuplicateGroup(
                    group_id=int(group_id),
                    bbox_ids=set(group_df["bbox_id"].unique()),
                    task_ids=set(group_df["task_id"].unique()),
                    species=group_df["species"].iloc[0],
                )
            )
        return groups

    def _get_task(self, task_id: int):
        """Fetch task from Label Studio with caching."""
        if task_id not in self._task_cache:
            try:
                task = self.ls_manager.get_task(task_id, as_sdk_task=False)
                self._task_cache[task_id] = task
            except Exception as e:
                logger.warning(f"Failed to fetch task {task_id}: {e}")
                self._task_cache[task_id] = None
        return self._task_cache[task_id]

    def _load_group_images(self, group: DuplicateGroup) -> Tuple[List[DroneImage], int]:
        """Load DroneImage objects for a duplicate group.

        Returns:
            Tuple of (list of DroneImages, original bbox count)
        """
        from wildetect.core.data import DroneImage
        images = []
        original_bbox_count = 0

        for task_id in group.task_ids:
            task = self._get_task(task_id)
            if task is None:
                continue

            # Filter task to only include bboxes from this group
            filtered_task = task.filter(valid_bbox_ids=group.bbox_ids)
            try:
                drone_image = DroneImage.from_ls_task(
                    filtered_task, flight_specs=self.flight_specs
                )
                # Set predictions to annotations for duplicate removal tuning
                drone_image.predictions = drone_image.annotations
                # Skip if no GPS location
                if drone_image.tile_gps_loc is None:
                    continue
                logger.debug(f"Loaded DroneImage for task {task_id}")

                num_bboxes = len(drone_image.get_non_empty_annotations())
                # Skip if no bboxes
                if num_bboxes == 0:
                    continue
                logger.debug(f"Loaded {num_bboxes} bboxes for task {task_id}")

                original_bbox_count += num_bboxes
                images.append(drone_image)
            except Exception as e:
                logger.warning(f"Failed to create DroneImage for task {task_id}: {e}")

        return images, original_bbox_count

    def _evaluate_group(
        self,
        group: DuplicateGroup,
        iou_threshold: float,
        min_overlap_threshold: float,
    ) -> Dict[str, int]:
        """Evaluate merger performance on a single duplicate group.

        For a duplicate group of N bboxes that represent the same animal,
        the ideal outcome is that N-1 duplicates are removed, leaving 1.

        Returns:
            Dict with 'expected_removals', 'actual_removals', 'original_count'
        """
        from wildetect.core.flight import GeographicMerger

        images, original_count = self._load_group_images(group)

        if len(images) < 2 or original_count == 0:
            return {"expected_removals": 0, "actual_removals": 0, "original_count": 0}

        logger.debug(f"Evaluating group {group.group_id} with {len(images)} images")

        # Run the merger
        merger = GeographicMerger(verbose=self.verbose)
        try:
            merged_images = merger.run(
                images,
                iou_threshold=iou_threshold,
                min_overlap_threshold=min_overlap_threshold,
            )
        except Exception as e:
            logger.warning(f"Merger failed for group {group.group_id}: {e}")
            return {"expected_removals": 0, "actual_removals": 0, "original_count": 0}

        # Count remaining detections after merge
        remaining_count = sum(
            len(img.get_non_empty_predictions()) for img in merged_images
        )

        # Expected: all N bboxes should merge to 1
        expected_removals = original_count - 1
        actual_removals = original_count - remaining_count

        return {
            "expected_removals": expected_removals,
            "actual_removals": actual_removals,
            "original_count": original_count,
        }

    def _compute_metrics(
        self,
        iou_threshold: float,
        min_overlap_threshold: float,
    ) -> Dict[str, float]:
        """Compute MAE (Mean Absolute Error) for duplicate removal accuracy.

        For each duplicate group:
        - Expected removals = N - 1 (where N = number of bboxes representing same animal)
        - Actual removals = original_count - remaining_count after merging
        - Error = |expected - actual|

        MAE = average error across all groups (lower is better)
        """
        errors = []
        total_expected = 0
        total_actual = 0
        total_original = 0
        groups_evaluated = 0

        for group in self.duplicate_groups:
            result = self._evaluate_group(group, iou_threshold, min_overlap_threshold)

            if result["original_count"] == 0:
                logger.debug(
                    f"Group {group.group_id} has no original bboxes, skipping"
                )
                continue

            expected = result["expected_removals"]
            actual = result["actual_removals"]
            error = abs(expected - actual)
            errors.append(error)

            total_expected += expected
            total_actual += actual
            total_original += result["original_count"]
            groups_evaluated += 1

        if groups_evaluated == 0:
            return {
                "mae": np.inf,
                "removal_accuracy": 0.0,
                "total_expected_removals": 0,
                "total_actual_removals": 0,
                "total_original_bboxes": 0,
                "total_groups": 0,
            }

        mae = sum(errors) / groups_evaluated

        # Also compute removal accuracy (percentage of expected removals achieved)
        removal_accuracy = total_actual / max(total_expected, 1)

        return {
            "mae": mae,
            "removal_accuracy": removal_accuracy,
            "total_expected_removals": total_expected,
            "total_actual_removals": total_actual,
            "total_original_bboxes": total_original,
            "total_groups": groups_evaluated,
        }

    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            MAE (Mean Absolute Error) - lower is better
        """
        iou_threshold = trial.suggest_float(
            "iou_threshold",
            self.iou_threshold_range[0],
            self.iou_threshold_range[1],
        )
        min_overlap_threshold = trial.suggest_float(
            "min_overlap_threshold",
            self.min_overlap_threshold_range[0],
            self.min_overlap_threshold_range[1],
        )

        metrics = self._compute_metrics(iou_threshold, min_overlap_threshold)

        # Log intermediate results
        trial.set_user_attr("removal_accuracy", metrics["removal_accuracy"])
        trial.set_user_attr("total_groups", metrics["total_groups"])

        return metrics["mae"]

    def run(
        self,
        run_name: str = "duplicate-removal-tuner",
        optuna_storage: str = "sqlite:///duplicate-tuner.db",
        optuna_load_if_exists: bool = True,
    ) -> Dict[str, Any]:
        """Run Optuna optimization and return best thresholds + metrics.

        Returns:
            Dictionary containing:
                - best_iou_threshold: Optimal detection-level IoU threshold
                - best_min_overlap_threshold: Optimal image-level overlap threshold
                - best_mae: MAE at optimal thresholds (lower is better)
                - removal_accuracy: Percentage of expected removals achieved
                - study: Optuna study object for further analysis
        """
        self.study = optuna.create_study(
            direction="minimize",  # MAE should be minimized
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=optuna_load_if_exists,
            storage=optuna_storage,
            study_name=run_name,
        )

        logger.info(
            f"Starting duplicate removal threshold optimization "
            f"with {self.n_trials} trials (minimizing MAE)"
        )
        self.study.optimize(self, n_trials=self.n_trials, show_progress_bar=True)

        best_iou = self.study.best_params["iou_threshold"]
        best_overlap = self.study.best_params["min_overlap_threshold"]
        best_metrics = self._compute_metrics(best_iou, best_overlap)

        logger.info("=" * 60)
        logger.info("Duplicate Removal Tuning Complete!")
        logger.info(f"Best iou_threshold: {best_iou:.3f}")
        logger.info(f"Best min_overlap_threshold: {best_overlap:.3f}")
        logger.info(f"Best MAE: {best_metrics['mae']:.3f}")
        logger.info(f"Removal accuracy: {best_metrics['removal_accuracy']:.1%}")
        logger.info(f"Groups evaluated: {best_metrics['total_groups']}")
        logger.info("=" * 60)

        return {
            "best_iou_threshold": best_iou,
            "best_min_overlap_threshold": best_overlap,
            "best_mae": best_metrics["mae"],
            "removal_accuracy": best_metrics["removal_accuracy"],
            "total_expected_removals": best_metrics["total_expected_removals"],
            "total_actual_removals": best_metrics["total_actual_removals"],
            "total_groups": best_metrics["total_groups"],
            "study": self.study,
        }
