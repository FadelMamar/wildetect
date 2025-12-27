"""IoU threshold tuning using Optuna for object detection."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import supervision as sv
from supervision.detection.utils.iou_and_nms import box_iou_batch

logger = logging.getLogger(__name__)


class IoUTuner:
    """Tune NMS and matching IoU thresholds to maximize F1-score using Optuna.

    Given a DataFrame of annotations containing both predictions and groundtruth,
    uses Optuna to find the optimal IoU thresholds for:
    1. NMS filtering of overlapping predictions
    2. Matching predictions to groundtruth boxes

    Example:
        >>> from wildata.converters import LabelStudioParser
        >>> parser = LabelStudioParser.from_file("annotations.json")
        >>> df = parser.to_dataframe()
        >>> tuner = IoUTuner(df, n_trials=50)
        >>> result = tuner.run()
        >>> print(f"Best NMS IoU: {result['best_nms_iou_threshold']:.3f}")
        >>> print(f"Best Match IoU: {result['best_match_iou_threshold']:.3f}")
    """

    def __init__(
        self,
        df_annotations: pd.DataFrame,
        duplicates_csv_path: Optional[str] = None,
        nms_iou_range: Tuple[float, float] = (0.1, 0.9),
        match_iou_range: Tuple[float, float] = (0.3, 0.8),
        n_trials: int = 50,
    ):
        """Initialize IoU tuner.

        Args:
            df_annotations: DataFrame from LabelStudioParser.to_dataframe()
                Must contain columns: task_id, x_pixel, y_pixel, width_pixel,
                height_pixel, origin, score, label
            duplicates_csv_path: Optional path to CSV identifying duplicate predictions
                across images. CSV must have columns: image (task_id),
                bounding box (result_id), duplicate (group ID)
            nms_iou_range: Range for NMS IoU threshold search (min, max)
            match_iou_range: Range for matching IoU threshold search (min, max)
            n_trials: Number of Optuna trials
        """
        self.df = df_annotations
        self.nms_iou_range = nms_iou_range
        self.match_iou_range = match_iou_range
        self.n_trials = n_trials
        self.study: Optional[optuna.Study] = None

        # Validate required columns
        required_columns = [
            "task_id",
            "x_pixel",
            "y_pixel",
            "width_pixel",
            "height_pixel",
            "origin",
            "score",
            "label",
        ]
        missing = [col for col in required_columns if col not in df_annotations.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Separate predictions and groundtruth based on source column
        self.df_preds = df_annotations[df_annotations["source"] == "prediction"].copy()
        self.df_gt = df_annotations[df_annotations["source"] == "annotation"].copy()
        # drop nans
        self.df_preds = self.df_preds.dropna(
            subset=["x_pixel", "y_pixel", "width_pixel", "height_pixel"]
        )
        self.df_gt = self.df_gt.dropna(
            subset=["x_pixel", "y_pixel", "width_pixel", "height_pixel"]
        )

        # Load duplicates mapping: result_id -> duplicate_group_id
        self.duplicates_map: Dict[str, int] = {}
        self.duplicate_groups: Dict[int, list] = {}  # group_id -> list of result_ids
        if duplicates_csv_path:
            self._load_duplicates(duplicates_csv_path)

        logger.info(
            f"IoUTuner initialized: {len(self.df_preds)} predictions, "
            f"{len(self.df_gt)} groundtruth boxes, "
            f"{df_annotations['task_id'].nunique()} tasks, "
            f"{len(self.duplicate_groups)} duplicate groups"
        )

    def _load_duplicates(self, csv_path: str) -> None:
        """Load duplicates CSV and build mapping.

        Args:
            csv_path: Path to CSV with columns: image, bounding box, duplicate
        """
        df_dup = pd.read_csv(csv_path)

        # Map result_id to duplicate group
        for _, row in df_dup.iterrows():
            result_id = str(row["bounding box"])
            group_id = int(row["duplicate"])
            self.duplicates_map[result_id] = group_id

            if group_id not in self.duplicate_groups:
                self.duplicate_groups[group_id] = []
            self.duplicate_groups[group_id].append(result_id)

        logger.info(
            f"Loaded {len(self.duplicates_map)} duplicate entries in {len(self.duplicate_groups)} groups"
        )

    def _df_to_detections(self, df: pd.DataFrame) -> sv.Detections:
        """Convert DataFrame rows to sv.Detections object.

        Args:
            df: DataFrame with bbox columns

        Returns:
            sv.Detections object with xyxy format bboxes
        """
        if df.empty:
            return sv.Detections.empty()

        # Convert to xyxy format: [x_min, y_min, x_max, y_max]
        xyxy = np.column_stack(
            [
                df["x_pixel"].values,
                df["y_pixel"].values,
                df["x_pixel"].values + df["width_pixel"].values,
                df["y_pixel"].values + df["height_pixel"].values,
            ]
        )

        confidence = df["score"].fillna(1.0).values.astype(np.float32)

        # Map labels to class IDs
        unique_labels = df["label"].unique()
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        class_id = df["label"].map(label_to_id).values.astype(int)

        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=confidence,
            class_id=class_id,
        )

    def _compute_metrics(
        self, nms_iou_threshold: float, match_iou_threshold: float
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for given thresholds.

        Handles duplicate predictions across images by collapsing them into groups.
        If any prediction in a duplicate group matches GT, the entire group is
        considered one true positive.

        Args:
            nms_iou_threshold: IoU threshold for NMS filtering
            match_iou_threshold: IoU threshold for pred-to-GT matching

        Returns:
            Dictionary with precision, recall, f1 metrics
        """
        # Track matched predictions by their result_id
        matched_result_ids: set = set()
        all_pred_result_ids: set = set()
        total_gt = 0
        matched_gt_count = 0

        for task_id in self.df["task_id"].unique():
            # Get predictions and GT for this image
            preds_df = self.df_preds[self.df_preds["task_id"] == task_id]
            gt_df = self.df_gt[self.df_gt["task_id"] == task_id]

            # Skip if no predictions or GT
            if preds_df.empty:
                total_gt += len(gt_df)
                continue

            # Track all prediction result_ids
            result_ids = preds_df["result_id"].tolist()

            # Convert to sv.Detections
            preds = self._df_to_detections(preds_df)
            gt = self._df_to_detections(gt_df)

            # Apply NMS on predictions, track which survive
            if len(preds) > 0:
                # Get indices that survive NMS
                orig_len = len(preds)
                preds = preds.with_nms(threshold=nms_iou_threshold)
                # Note: NMS may reorder, but we track by result_id below

            # For each surviving prediction, track its result_id
            # Since NMS may filter, we use the filtered df
            filtered_mask = np.zeros(len(preds_df), dtype=bool)
            if len(preds) > 0:
                # Match remaining xyxy back to original
                for i in range(len(preds)):
                    for j, (_, row) in enumerate(preds_df.iterrows()):
                        orig_xyxy = [
                            row["x_pixel"],
                            row["y_pixel"],
                            row["x_pixel"] + row["width_pixel"],
                            row["y_pixel"] + row["height_pixel"],
                        ]
                        if np.allclose(preds.xyxy[i], orig_xyxy, atol=1e-3):
                            filtered_mask[j] = True
                            break

            surviving_result_ids = [
                rid for rid, mask in zip(result_ids, filtered_mask) if mask
            ]
            all_pred_result_ids.update(surviving_result_ids)

            # Match predictions to GT
            if len(gt) > 0:
                total_gt += len(gt)
                if len(preds) > 0:
                    iou_matrix = box_iou_batch(preds.xyxy, gt.xyxy)
                    matched_gt = set()
                    for pred_idx in range(len(preds)):
                        best_gt_idx = int(np.argmax(iou_matrix[pred_idx]))
                        if iou_matrix[pred_idx, best_gt_idx] >= match_iou_threshold:
                            if best_gt_idx not in matched_gt:
                                matched_gt.add(best_gt_idx)
                                # Mark this result_id as matched
                                if pred_idx < len(surviving_result_ids):
                                    matched_result_ids.add(
                                        surviving_result_ids[pred_idx]
                                    )
                    matched_gt_count += len(matched_gt)

        # Handle duplicate groups: collapse predictions in same group
        # If any member of a duplicate group matched, count group as one TP
        if self.duplicate_groups:
            matched_groups = set()
            non_dup_matched = set()

            for result_id in matched_result_ids:
                if result_id in self.duplicates_map:
                    matched_groups.add(self.duplicates_map[result_id])
                else:
                    non_dup_matched.add(result_id)

            # Count: matched non-duplicates + matched duplicate groups = TP
            tp = len(non_dup_matched) + len(matched_groups)

            # Count total predictions after collapsing duplicates
            counted_groups = set()
            non_dup_preds = set()
            for result_id in all_pred_result_ids:
                if result_id in self.duplicates_map:
                    counted_groups.add(self.duplicates_map[result_id])
                else:
                    non_dup_preds.add(result_id)
            total_preds = len(non_dup_preds) + len(counted_groups)

            fp = total_preds - tp
        else:
            # No duplicates: simple counting
            tp = len(matched_result_ids)
            fp = len(all_pred_result_ids) - tp

        fn = total_gt - matched_gt_count

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def _match_detections(
        self, preds: sv.Detections, gt: sv.Detections, match_threshold: float
    ) -> Tuple[int, int, int]:
        """Match predictions to GT using greedy IoU matching.

        Args:
            preds: Predicted detections
            gt: Groundtruth detections
            match_threshold: IoU threshold for a valid match

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        if len(gt) == 0:
            return 0, len(preds), 0
        if len(preds) == 0:
            return 0, 0, len(gt)

        iou_matrix = box_iou_batch(preds.xyxy, gt.xyxy)

        # Greedy matching: for each prediction, find best matching GT
        matched_gt = set()
        tp = 0
        for pred_idx in range(len(preds)):
            best_gt_idx = int(np.argmax(iou_matrix[pred_idx]))
            if iou_matrix[pred_idx, best_gt_idx] >= match_threshold:
                if best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)

        fp = len(preds) - tp
        fn = len(gt) - len(matched_gt)
        return tp, fp, fn

    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            F1-score for the suggested parameters
        """
        nms_threshold = trial.suggest_float(
            "nms_iou_threshold", self.nms_iou_range[0], self.nms_iou_range[1]
        )
        match_threshold = trial.suggest_float(
            "match_iou_threshold", self.match_iou_range[0], self.match_iou_range[1]
        )
        metrics = self._compute_metrics(nms_threshold, match_threshold)
        return metrics["f1"]

    def run(self) -> Dict[str, Any]:
        """Run Optuna optimization and return best thresholds + metrics.

        Returns:
            Dictionary containing:
                - best_nms_iou_threshold: Optimal NMS IoU threshold
                - best_match_iou_threshold: Optimal matching IoU threshold
                - best_f1_score: F1-score at optimal thresholds
                - best_precision: Precision at optimal thresholds
                - best_recall: Recall at optimal thresholds
                - study: Optuna study object for further analysis
        """
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        logger.info(f"Starting IoU threshold optimization with {self.n_trials} trials")
        self.study.optimize(self, n_trials=self.n_trials, show_progress_bar=True)

        best_nms = self.study.best_params["nms_iou_threshold"]
        best_match = self.study.best_params["match_iou_threshold"]
        best_metrics = self._compute_metrics(best_nms, best_match)

        logger.info("=" * 50)
        logger.info("IoU Tuning Complete!")
        logger.info(f"Best NMS IoU threshold: {best_nms:.3f}")
        logger.info(f"Best Match IoU threshold: {best_match:.3f}")
        logger.info(f"Best F1-score: {best_metrics['f1']:.3f}")
        logger.info(f"Precision: {best_metrics['precision']:.3f}")
        logger.info(f"Recall: {best_metrics['recall']:.3f}")
        logger.info("=" * 50)

        return {
            "best_nms_iou_threshold": best_nms,
            "best_match_iou_threshold": best_match,
            "best_f1_score": best_metrics["f1"],
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "study": self.study,
        }
