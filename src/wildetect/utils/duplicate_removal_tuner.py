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

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import optuna
import pandas as pd
from tqdm import tqdm

from wildetect.core.config import FlightSpecs
from wildetect.core.data import DroneImage
from wildetect.core.flight import GeographicMerger
from wildetect.core.visualization.labelstudio_manager import LabelStudioManager

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate detections."""

    group_id: int
    bbox_ids: Set[str]
    task_ids: Set[int]
    species: str


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
        iou_threshold_range: Tuple[float, float] = (0.1, 0.9),
        min_overlap_threshold_range: Tuple[float, float] = (0.0, 0.5),
        n_trials: int = 50,
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
                original_bbox_count += len(drone_image.get_non_empty_predictions())
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
        images, original_count = self._load_group_images(group)

        if len(images) < 2 or original_count == 0:
            return {"expected_removals": 0, "actual_removals": 0, "original_count": 0}

        # Run the merger
        merger = GeographicMerger()
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
        """Compute precision, recall, F1 for given thresholds.

        Metrics are computed based on duplicate removal accuracy:
        - TP: Correctly identified and removed duplicates
        - FP: Removed bboxes that shouldn't have been removed (over-merging)
        - FN: Duplicates that should have been removed but weren't
        """
        total_expected = 0
        total_actual = 0
        total_original = 0

        for group in self.duplicate_groups:
            result = self._evaluate_group(group, iou_threshold, min_overlap_threshold)
            total_expected += result["expected_removals"]
            total_actual += result["actual_removals"]
            total_original += result["original_count"]

        if total_expected == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # TP: min of expected and actual (correctly removed duplicates)
        # FP: actual - TP (over-removed, shouldn't have been removed)
        # FN: expected - TP (under-removed, should have been removed)
        tp = min(total_expected, total_actual)
        fp = max(0, total_actual - total_expected)
        fn = max(0, total_expected - total_actual)

        e = 1e-6
        precision = tp / (tp + fp + e)
        recall = tp / (tp + fn + e)
        f1 = 2 * precision * recall / (precision + recall + e)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_expected_removals": total_expected,
            "total_actual_removals": total_actual,
            "total_original_bboxes": total_original,
        }

    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            F1-score for the suggested parameters
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
        trial.set_user_attr("precision", metrics["precision"])
        trial.set_user_attr("recall", metrics["recall"])

        return metrics["f1"]

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
                - best_f1_score: F1-score at optimal thresholds
                - best_precision: Precision at optimal thresholds
                - best_recall: Recall at optimal thresholds
                - study: Optuna study object for further analysis
        """
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=optuna_load_if_exists,
            storage=optuna_storage,
            study_name=run_name,
        )

        logger.info(
            f"Starting duplicate removal threshold optimization "
            f"with {self.n_trials} trials"
        )
        self.study.optimize(self, n_trials=self.n_trials, show_progress_bar=True)

        best_iou = self.study.best_params["iou_threshold"]
        best_overlap = self.study.best_params["min_overlap_threshold"]
        best_metrics = self._compute_metrics(best_iou, best_overlap)

        logger.info("=" * 60)
        logger.info("Duplicate Removal Tuning Complete!")
        logger.info(f"Best iou_threshold: {best_iou:.3f}")
        logger.info(f"Best min_overlap_threshold: {best_overlap:.3f}")
        logger.info(f"Best F1-score: {best_metrics['f1']:.3f}")
        logger.info(f"Precision: {best_metrics['precision']:.3f}")
        logger.info(f"Recall: {best_metrics['recall']:.3f}")
        logger.info("=" * 60)

        return {
            "best_iou_threshold": best_iou,
            "best_min_overlap_threshold": best_overlap,
            "best_f1_score": best_metrics["f1"],
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "total_expected_removals": best_metrics["total_expected_removals"],
            "total_actual_removals": best_metrics["total_actual_removals"],
            "study": self.study,
        }
