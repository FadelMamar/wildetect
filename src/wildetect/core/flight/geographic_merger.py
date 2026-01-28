"""
Geographic merging for combining detections across overlapping drone images.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torchmetrics.functional.detection import complete_intersection_over_union
from tqdm import tqdm

from ..data.detection import Detection
from ..data.drone_image import DroneImage

logger = logging.getLogger("GEOGRAPHIC_MERGER")


@dataclass
class MergedDetection(Detection):
    """Represents a detection merged from multiple overlapping images."""

    source_images: List[str] = field(default_factory=list)
    merged_detections: List[Detection] = field(default_factory=list)


class OverlapStrategy(ABC):
    """Abstract strategy for detecting overlapping images."""

    def __init__(
        self,
    ):
        self.stats = dict()

    @abstractmethod
    def find_overlapping_images(
        self, images: List[DroneImage], min_overlap_threshold: float = 0.0
    ) -> Dict[str, List[str]]:
        """Find overlapping images and return mapping of image_id -> overlapping_image_ids.
        Args:
            images (List[DroneImage]): List of DroneImage objects.
            min_overlap_threshold (float): Minimum IoU threshold to consider images as overlapping.
        Returns:
            Dict[str, List[str]]: Mapping from image ID to list of overlapping image IDs.
        """
        pass


class GPSOverlapStrategy(OverlapStrategy):
    """
    GPS-based overlap detection using geographic footprints.
    Provides methods to find overlapping images and compute statistics on the overlap map.
    """

    def __init__(
        self,verbose: bool = False,
    ):
        super().__init__()
        self.stats = {
            "avg_overlap_ratio": 0.0,
            "max_overlap_ratio": 0.0,
            "min_overlap_ratio": 0.0,
        }
        self.verbose = verbose
    def _compute_iou(self, images: List[DroneImage]) -> np.ndarray | None:
        """Compute Intersection over Union (IoU) between all pairs of image tiles.
        Args:
            images (List[Tile]): List of Tile objects.
        Returns:
            np.ndarray: IoU matrix between all pairs of tiles.
        """

        boxes = [img.geo_box for img in images if img.geo_box is not None]
        if len(boxes) == 0:
            return None
        boxes = torch.tensor(boxes)
        box_ious = complete_intersection_over_union(
            preds=boxes, target=boxes, aggregate=False
        ).numpy()
        return box_ious

    def find_overlapping_images(
        self, images: List[DroneImage], min_overlap_threshold: float = 0.0,
    ) -> Dict[str, List[str]]:
        """Find overlapping images using precomputed IoU matrix.
        Args:
            images (List[DroneImage]): List of DroneImage objects.
            min_overlap_threshold (float): Minimum IoU threshold to consider images as overlapping.
        Returns:
            Dict[str, List[str]]: Map of image IDs to their overlapping neighbor image IDs.
        """
        overlap_map = defaultdict(list)
        overlap_ratios = defaultdict(list)
        ious = self._compute_iou(images=images)
        if ious is None:
            return {}
        progress_bar = tqdm(images, desc="Finding overlapping images", disable=not self.verbose)
        for i, img1 in enumerate(progress_bar):
            for j, img2 in enumerate(images):
                if i <= j:  # only consider above diagonal because it's symmetric
                    continue
                if ious[i, j] > min_overlap_threshold:
                    overlap_map[str(img1.image_path)].append(str(img2.image_path))
                    overlap_ratios[str(img1.image_path)].append(ious[i, j])

        overlap_map = dict(overlap_map)
        overlap_ratios = dict(overlap_ratios)
        # compute stats
        self.stats = self.overlap_map_stats(overlap_map, overlap_ratios)
        return overlap_map

    def overlap_map_stats(
        self, overlap_map: Dict[str, List[str]], overlap_ratios: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compute statistics on the overlap map and overlap ratios.
        Args:
            overlap_map (dict): Map of image IDs to their overlapping neighbor image IDs.
            overlap_ratios (Dict[str, List[float]]): Map of image IDs to their overlap IoU values.
        Returns:
            dict: Statistics including number of images, neighbor counts, and overlap ratios.
        """
        # Compute neighbor statistics
        num_images = len(overlap_map)
        neighbor_counts = [len(neighs) for neighs in overlap_map.values()]
        if neighbor_counts:
            avg_neighbors = sum(neighbor_counts) / len(neighbor_counts)
        else:
            avg_neighbors = 0.0
        max_neighbors = max(neighbor_counts) if neighbor_counts else 0
        min_neighbors = min(neighbor_counts) if neighbor_counts else 0

        # Compute overlap ratio statistics using NumPy vectorization
        vals = list(overlap_ratios.values())
        if vals:
            all_ratios = np.concatenate(vals)
            positive_ratios = all_ratios[all_ratios > 0]
        else:
            positive_ratios = np.array([])

        if len(positive_ratios) > 0:
            avg_overlap_ratio = float(np.mean(positive_ratios))
            max_overlap_ratio = float(np.max(positive_ratios))
            min_overlap_ratio = float(np.min(positive_ratios))
        else:
            avg_overlap_ratio = 0.0
            max_overlap_ratio = 0.0
            min_overlap_ratio = 0.0

        return {
            "num_images": num_images,
            "avg_neighbors": avg_neighbors,
            "max_neighbors": max_neighbors,
            "min_neighbors": min_neighbors,
            "avg_overlap_ratio": avg_overlap_ratio,
            "max_overlap_ratio": max_overlap_ratio,
            "min_overlap_ratio": min_overlap_ratio,
        }


class DuplicateRemovalStrategy(ABC):
    """Abstract strategy for removing duplicate detections."""

    @abstractmethod
    def remove_duplicates(
        self,
        tiles: List[DroneImage],
        overlap_map: Dict[str, List[str]],
        iou_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """Remove duplicate detections from tiles in-place based on overlap map and IoU threshold.
        Args:
            tiles (List[Tile]): List of Tile objects.
            overlap_map (Dict[str, List[str]]): Overlap mapping between tiles.
            iou_threshold (float): IoU threshold for considering duplicates.
        Returns:
            Dict[str, Any]: Statistics about the duplicate removal process.
        """
        pass


class CentroidProximityRemovalStrategy(DuplicateRemovalStrategy):
    """
    Responsible for finding groups of duplicate predictions given a list of Tile objects with predictions.
    Can also return unique predictions using a specified duplicate removal strategy.
    """

    def __init__(
        self,
        verbose: bool = False,
    ):
        self.invalid_iou = -999.0
        self.verbose = verbose
    def _compute_iou(
        self, detections_1: List[Detection], detections_2: List[Detection]
    ) -> np.ndarray:
        """Compute Intersection over Union (IoU) between two lists of detections.
        Args:
            detections_1 (List[Detection]): First list of detections.
            detections_2 (List[Detection]): Second list of detections.
        Returns:
            np.ndarray: IoU matrix between detections.
        """

        # Check if all detections have geographic footprints
        missing_geo_boxes_1 = [
            i for i, det in enumerate(detections_1) if det.geo_box is None
        ]
        missing_geo_boxes_2 = [
            i for i, det in enumerate(detections_2) if det.geo_box is None
        ]

        if missing_geo_boxes_1:
            logger.warning(
                f"Detections {missing_geo_boxes_1} in list 1 missing geo_box"
            )
        if missing_geo_boxes_2:
            logger.warning(
                f"Detections {missing_geo_boxes_2} in list 2 missing geo_box"
            )

        # Initialize IoU matrix with -1 (invalid)
        ious = np.full((len(detections_1), len(detections_2)), self.invalid_iou)

        # Get valid detections with geo_boxes
        valid_detections_1 = [
            (i, det) for i, det in enumerate(detections_1) if det.geo_box is not None
        ]
        valid_detections_2 = [
            (j, det) for j, det in enumerate(detections_2) if det.geo_box is not None
        ]

        if not valid_detections_1 or not valid_detections_2:
            logger.warning("No valid geographic boxes found for IoU computation")
            return ious

        # Extract boxes and their original indices
        boxes_1 = [det.geo_box for _, det in valid_detections_1]
        boxes_2 = [det.geo_box for _, det in valid_detections_2]
        indices_1 = [i for i, _ in valid_detections_1]
        indices_2 = [j for j, _ in valid_detections_2]

        # Compute IoU for valid boxes
        boxes_tensor_1 = torch.tensor(boxes_1)
        boxes_tensor_2 = torch.tensor(boxes_2)

        box_ious = complete_intersection_over_union(
            preds=boxes_tensor_1, target=boxes_tensor_2, aggregate=False
        ).numpy()

        # Map results back to original indices
        for i, orig_i in enumerate(indices_1):
            for j, orig_j in enumerate(indices_2):
                ious[orig_i, orig_j] = box_ious[i, j]

        return ious

    def _prune_duplicates_between_tiles(
        self, tile1: DroneImage, tile2: DroneImage, iou_threshold: float = 0.8
    ) -> Tuple[Dict[Tuple[str, str], List[Detection]], Dict[str, Any]]:
        """Prune duplicate detections between two tiles based on IoU and class name.
        Args:
            tile1 (DroneImage): First tile.
            tile2 (DroneImage): Second tile.
            iou_threshold (float): IoU threshold for considering duplicates.
        Returns:
            Tuple[Dict[Tuple[str, str], List[Detection]], Dict[str, Any]]: Pruned detections and IoU statistics.
        """
        # Initialize IoU statistics
        iou_stats = {
            "iou_matrix_shape": (0, 0),
            "iou_range": {"min": 0.0, "max": 0.0, "mean": 0.0},
            "above_threshold_count": 0,
            "duplicate_pairs": [],
            "total_ious_computed": 0,
        }

        # Cache predictions to avoid repeated calls
        dets1 = tile1.get_non_empty_predictions()
        dets2 = tile2.get_non_empty_predictions()

        # Validate that tiles have GPS data for geographic footprint computation
        if tile1.geographic_footprint is None:
            logger.warning(
                f"Tile1 {tile1.image_path} missing GPS data required for geographic footprint computation"
            )
        if tile2.geographic_footprint is None:
            logger.warning(
                f"Tile2 {tile2.image_path} missing GPS data required for geographic footprint computation"
            )

        # Check how many detections still lack geographic footprints
        missing_geo_1 = [det for det in dets1 if det.geo_box is None]
        missing_geo_2 = [det for det in dets2 if det.geo_box is None]

        if missing_geo_1:
            logger.warning(
                f"Tile1 {tile1.image_path} has {len(missing_geo_1)} detections without geographic footprints"
            )
        if missing_geo_2:
            logger.warning(
                f"Tile2 {tile2.image_path} has {len(missing_geo_2)} detections without geographic footprints"
            )

        if len(dets1) == 0 or len(dets2) == 0:
            return (
                {
                    (str(tile1.image_path), str(tile2.image_path)): dets1,
                    (str(tile2.image_path), str(tile1.image_path)): dets2,
                },
                iou_stats,
            )

        ious = self._compute_iou(dets1, dets2)  # shape [N1, N2]
        keep1 = np.ones(len(dets1), dtype=bool)
        keep2 = np.ones(len(dets2), dtype=bool)

        # Update IoU statistics
        iou_stats["iou_matrix_shape"] = ious.shape
        iou_stats["total_ious_computed"] = ious.size
        if ious.size > 0:
            valid_ious = ious[ious != self.invalid_iou]  # Exclude invalid values
            if len(valid_ious) > 0:
                iou_stats["iou_range"]["min"] = float(valid_ious.min())
                iou_stats["iou_range"]["max"] = float(valid_ious.max())
                iou_stats["iou_range"]["mean"] = float(valid_ious.mean())
                above_threshold = valid_ious[valid_ious > iou_threshold]
                iou_stats["above_threshold_count"] = len(above_threshold)
                # Log IoU matrix summary for debugging
                logger.debug(f"IoU matrix shape: {ious.shape}")
                logger.debug(
                    f"IoU range: {valid_ious.min():.3f} to {valid_ious.max():.3f}, "
                    f"Mean: {valid_ious.mean():.3f}"
                )
                logger.debug(
                    f"Detections above IoU threshold ({iou_threshold}): {len(above_threshold)}"
                )

        dets1 = tile1.get_non_empty_predictions()
        dets2 = tile2.get_non_empty_predictions()

        # Filter out predictions of different classes
        for i, det1 in enumerate(dets1):
            for j, det2 in enumerate(dets2):
                if det1.class_name != det2.class_name:
                    ious[i, j] = self.invalid_iou
                    logger.debug(
                        f"det1.class_name: {det1.class_name}, det2.class_name: {det2.class_name}"
                    )
        logger.debug(f"ious : {ious.round(2).tolist()}")
        idxs1, idxs2 = np.where(ious > iou_threshold)
        for i, j in zip(idxs1, idxs2):
            det1 = dets1[i]
            det2 = dets2[j]
            current_iou = ious[i, j]
            if det1.is_empty or det2.is_empty:
                logger.debug(f"Skipping empty detection: {det1} or {det2}")
                keep1[i] = False
                keep2[j] = False
                continue
            # Only process if both are still marked to keep
            if not (keep1[i] and keep2[j]):
                continue
            logger.debug(
                f"det1.class_name: {det1.class_name}, det2.class_name: {det2.class_name}"
            )
            # Collect IoU data for this duplicate pair
            duplicate_pair_data = {
                "iou": float(current_iou),
                "class_name": det1.class_name,
                "det1_confidence": float(det1.confidence),
                "det2_confidence": float(det2.confidence),
                "det1_centroid_distance": float(det1.distance_to_centroid),
                "det2_centroid_distance": float(det2.distance_to_centroid),
                "det1_kept": False,
                "det2_kept": False,
            }

            # Log the IoU and decision being made
            logger.debug(
                f"Processing duplicate pair - IoU: {current_iou:.3f}, "
                f"Class: {det1.class_name}, "
                f"Det1 confidence: {det1.confidence:.3f}, "
                f"Det2 confidence: {det2.confidence:.3f}, "
                f"Det1 centroid distance: {det1.distance_to_centroid:.3f}, "
                f"Det2 centroid distance: {det2.distance_to_centroid:.3f}"
            )

            if det1.distance_to_centroid == det2.distance_to_centroid:
                logger.debug(
                    f"Keeping det2 (same centroid distance), removing det1 - IoU: {current_iou:.3f}"
                )
                keep1[i] = False
                duplicate_pair_data["det2_kept"] = True
            elif det1.distance_to_centroid < det2.distance_to_centroid:
                logger.debug(
                    f"Keeping det1 (closer to centroid), removing det2 - IoU: {current_iou:.3f}"
                )
                keep2[j] = False
                duplicate_pair_data["det1_kept"] = True
            else:
                logger.debug(
                    f"Keeping det2 (closer to centroid), removing det1 - IoU: {current_iou:.3f}"
                )
                keep1[i] = False
                duplicate_pair_data["det2_kept"] = True

            iou_stats["duplicate_pairs"].append(duplicate_pair_data)

        pruned_detections_stats_1 = [det for i, det in enumerate(dets1) if keep1[i]]
        pruned_detections_stats_2 = [det for j, det in enumerate(dets2) if keep2[j]]

        # Log summary of what was kept
        original_count_1 = len(dets1)
        original_count_2 = len(dets2)
        kept_count_1 = len(pruned_detections_stats_1)
        kept_count_2 = len(pruned_detections_stats_2)

        logger.debug(
            f"Tile pruning summary - "
            f"Tile1: {kept_count_1}/{original_count_1} detections kept, "
            f"Tile2: {kept_count_2}/{original_count_2} detections kept"
        )

        if kept_count_1 < original_count_1 or kept_count_2 < original_count_2:
            logger.debug(
                f"Removed duplicates - "
                f"Tile1 removed: {original_count_1 - kept_count_1}, "
                f"Tile2 removed: {original_count_2 - kept_count_2}"
            )

        tile1.set_predictions(pruned_detections_stats_1, update_gps=False)
        tile2.set_predictions(pruned_detections_stats_2, update_gps=False)

        pruned_detections = {
            (str(tile1.image_path), str(tile2.image_path)): pruned_detections_stats_1,
            (str(tile2.image_path), str(tile1.image_path)): pruned_detections_stats_2,
        }

        return pruned_detections, iou_stats

    def remove_duplicates(
        self,
        tiles: List[DroneImage],
        overlap_map: Dict[str, List[str]],
        iou_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        For every image (Tile), consider its neighbors' predictions (from overlap_map).
        For each prediction in the image, check all predictions in neighbor images.
        If class matches and IoU > threshold, save the pair (detection, neighbor_detection) in a list.
        Returns the list of such pairs and statistics about the removal process.
        """
        image_path_to_tile = {str(tile.image_path): tile for tile in tiles}
        pruned_detections = dict()
        all_iou_stats = []

        # Store original counts for statistics
        original_total_detections = sum(
            len(tile.get_non_empty_predictions()) for tile in tiles
        )

        for image_path in tqdm(
            overlap_map.keys(), desc="Removing duplicates in Overlapping regions", disable=not self.verbose
        ):
            tile = image_path_to_tile[image_path]
            for neighbor in overlap_map[image_path]:
                tile2 = image_path_to_tile[str(neighbor)]
                pruned, iou_stats = self._prune_duplicates_between_tiles(
                    tile, tile2, iou_threshold
                )
                pruned_detections.update(pruned)
                all_iou_stats.append(iou_stats)

        # Compute statistics
        stats = self._compute_duplicate_removal_stats(
            pruned_detections, original_total_detections
        )

        # Add IoU statistics
        stats["iou_statistics"] = all_iou_stats

        # Add geographic footprint statistics
        stats["geographic_footprint_stats"] = self.get_geographic_footprint_stats(tiles)

        return stats

    def get_geographic_footprint_stats(self, tiles: List[DroneImage]) -> Dict[str, Any]:
        """Get statistics about geographic footprint availability across tiles.

        Args:
            tiles (List[DroneImage]): List of tiles to analyze

        Returns:
            Dict[str, Any]: Statistics about geographic footprint availability
        """
        stats = {
            "total_tiles": len(tiles),
            "tiles_with_gps": 0,
            "tiles_with_geographic_footprints": 0,
            "total_detections": 0,
            "detections_with_geographic_footprints": 0,
            "tiles_missing_gps_data": [],
            "detections_missing_geographic_footprints": 0,
        }

        for tile in tiles:
            # Check tile GPS data
            has_gps = tile.tile_gps_loc is not None

            if has_gps:
                stats["tiles_with_gps"] += 1
            else:
                stats["tiles_missing_gps_data"].append(tile.image_path)

            # Check tile geographic footprint
            if tile.geographic_footprint is not None:
                stats["tiles_with_geographic_footprints"] += 1

            # Check detection geographic footprints
            for det in tile.get_non_empty_predictions():
                stats["total_detections"] += 1
                if det.geo_box is not None:
                    stats["detections_with_geographic_footprints"] += 1
                else:
                    stats["detections_missing_geographic_footprints"] += 1

        return stats

    def _compute_duplicate_removal_stats(
        self,
        pruned_detections: Dict[Tuple[str, str], List[Detection]],
        original_total_detections: int,
    ) -> Dict[str, Any]:
        """Compute comprehensive statistics about duplicate removal process."""
        stats = {
            "total_image_pairs_processed": len(pruned_detections),
            "total_detections_removed": 0,
            "duplicate_groups_by_class": defaultdict(int),
            "avg_confidence_improvement": 0.0,
            "duplicate_removal_rate": 0.0,
            "class_duplicate_stats": defaultdict(
                lambda: {
                    "total_duplicates": 0,
                    "avg_confidence": 0.0,
                    "removal_rate": 0.0,
                }
            ),
            "geographic_spread": {
                "avg_duplicate_distance": 0.0,
                "duplicate_hotspots": [],
                "boundary_duplicates": 0,
            },
        }

        total_removed = 0
        class_stats = defaultdict(list)

        for (img1, img2), detections in pruned_detections.items():
            for det in detections:
                total_removed += 1
                stats["duplicate_groups_by_class"][det.class_name] += 1
                class_stats[det.class_name].append(det.confidence)

        # Calculate removal rate
        if original_total_detections > 0:
            stats["duplicate_removal_rate"] = total_removed / original_total_detections

        stats["total_detections_removed"] = total_removed

        # Calculate class-specific statistics
        for class_name, confidences in class_stats.items():
            if confidences:
                stats["class_duplicate_stats"][class_name]["total_duplicates"] = len(
                    confidences
                )
                stats["class_duplicate_stats"][class_name]["avg_confidence"] = sum(
                    confidences
                ) / max(len(confidences), 1)
                # Avoid division by zero
                if original_total_detections > 0:
                    stats["class_duplicate_stats"][class_name]["removal_rate"] = (
                        len(confidences) / original_total_detections
                    )
                else:
                    stats["class_duplicate_stats"][class_name]["removal_rate"] = 0.0

        # Convert defaultdict to regular dict for JSON serialization
        stats["duplicate_groups_by_class"] = dict(stats["duplicate_groups_by_class"])
        stats["class_duplicate_stats"] = dict(stats["class_duplicate_stats"])

        return stats


class GeographicMerger:
    """Merges detections across overlapping geographic regions."""

    def __init__(
        self,verbose: bool = False,
    ):
        """Initialize the geographic merger."""
        self.overlap_strategy = GPSOverlapStrategy(verbose=verbose)
        self.duplicate_removal_strategy = CentroidProximityRemovalStrategy(verbose=verbose)
        self.verbose = verbose

    def get_image_overlap_stats(
        self,
    ):
        return self.overlap_strategy.stats

    def find_overlapping_images(
        self, drone_images: List[DroneImage], min_overlap_threshold: float = 0.0
    ) -> Dict[str, List[str]]:
        """Find overlapping images using the overlap strategy."""
        return self.overlap_strategy.find_overlapping_images(
            drone_images, min_overlap_threshold=min_overlap_threshold
        )
    
    def _log(self, message: str):
        if self.verbose:
            logger.info(message)

    def run(
        self,
        drone_images: List[DroneImage],
        iou_threshold: float = 0.2,
        min_overlap_threshold: float = 0.0,
    ) -> List[DroneImage]:
        """Merge detections across overlapping geographic regions.

        Args:
            drone_images (List[DroneImage]): List of drone images with detections
            iou_threshold (float): IoU threshold for duplicate removal

        Returns:
            List[DroneImage]: List of merged DroneImage objects.
        """
        self._log(f"Merging detections from {len(drone_images)} drone images")

        # Validate geographic footprint availability
        geo_stats = self.duplicate_removal_strategy.get_geographic_footprint_stats(
            drone_images
        )
        self._log(f"Geographic footprint availability:")
        self._log(f"  Tiles with GPS data: {geo_stats['tiles_with_gps']}/{geo_stats['total_tiles']}")
        self._log(f"  Detections with geographic footprints: {geo_stats['detections_with_geographic_footprints']}/{geo_stats['total_detections']}")

        overlap_map = self.overlap_strategy.find_overlapping_images(
            drone_images, min_overlap_threshold=min_overlap_threshold
        )

        # Merge detections based on geographic proximity
        # The duplicate_removal_strategy modifies the drone_images in-place to remove duplicates
        self.duplicate_removal_strategy.remove_duplicates(
            drone_images, overlap_map=overlap_map, iou_threshold=iou_threshold
        )
        return drone_images
