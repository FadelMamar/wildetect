"""IoU threshold tuning using Optuna for object detection."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import supervision as sv
from supervision.detection.utils.iou_and_nms import OverlapMetric, box_iou_batch
from supervision.metrics.detection import ConfusionMatrix

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
        df_predictions: pd.DataFrame,
        duplicates_csv_path: Optional[str] = None,
        nms_iou_range: Tuple[float, float] = (0.1, 0.9),
        match_iou_range: Tuple[float, float] = (0.3, 0.8),
        conf_threshold_range: Tuple[float, float] = (0.0, 0.5),
        n_trials: int = 50,
        class_agnostic: bool = True,
        background_classes: list[str] = ["background"],
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
            conf_threshold_range: Range for confidence threshold search (min, max)
            n_trials: Number of Optuna trials
        """
        self.nms_iou_range = nms_iou_range
        self.match_iou_range = match_iou_range
        self.conf_threshold_range = conf_threshold_range
        self.n_trials = n_trials
        self.study: Optional[optuna.Study] = None
        self.class_agnostic = class_agnostic
        self.background_classes = background_classes

        if not self.class_agnostic:
            raise NotImplementedError("Class-agnostic mode is supported for IoUTuner")

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
        self.df_preds = df_predictions.copy()
        self.df_gt = df_annotations.copy()
        self.task_ids = self.get_task_ids()
        self.labels = self.get_labels()
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

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
            f"{len(self.task_ids)} tasks, "
            f"{len(self.duplicate_groups)} duplicate groups"
        )

    def get_task_ids(self):
        diff = set(self.df_preds["task_id"]).difference(set(self.df_gt["task_id"]))
        if len(diff) > 0:
            raise ValueError(
                f"Task IDs do not match between predictions and groundtruth: {diff}"
            )
        return set(self.df_preds["task_id"]).union(set(self.df_gt["task_id"]))

    def get_labels(self) -> list[str]:
        labels = list(set(self.df_preds["label"]).union(set(self.df_gt["label"])))

        if self.background_classes:
            labels = [label for label in labels if label not in self.background_classes]

        if self.class_agnostic:
            labels = ["wildlife"]
            self.df_preds["label"] = "wildlife"
            self.df_gt["label"] = "wildlife"

        return labels

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
        class_id = df["label"].map(self.label_to_id).values.astype(int)

        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=confidence,
            class_id=class_id,
        )

    def _compute_metrics(
        self,
        nms_iou_threshold: float,
        match_iou_threshold: float,
        conf_threshold: float = 0.0,
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

        preds = []
        gts = []
        for task_id in self.task_ids:
            # Get predictions and GT for this image
            preds_df = self.df_preds[self.df_preds["task_id"] == task_id]
            gt_df = self.df_gt[self.df_gt["task_id"] == task_id]
            # Convert to sv.Detections
            preds.append(self._df_to_detections(preds_df))
            gts.append(self._df_to_detections(gt_df))

        preds = [
            pred.with_nms(
                threshold=nms_iou_threshold,
                class_agnostic=self.class_agnostic,
                overlap_method=OverlapMetric.IOU,
            )
            for pred in preds
        ]
        confusion_matrix = ConfusionMatrix.from_detections(
            predictions=preds,
            targets=gts,
            classes=self.labels,
            iou_threshold=match_iou_threshold,
            conf_threshold=conf_threshold,
        )

        tp = confusion_matrix.matrix[0, 0]
        fp = confusion_matrix.matrix[1, 0]
        fn = confusion_matrix.matrix[0, 1]
        tn = confusion_matrix.matrix[1, 1]

        e = 1e-6
        precision = tp / (tp + fp + e)
        recall = tp / (tp + fn + e)
        f1 = 2 * precision * recall / (precision + recall + e)

        return {"precision": precision, "recall": recall, "f1": f1}

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
        conf_threshold = trial.suggest_float(
            "conf_threshold", self.conf_threshold_range[0], self.conf_threshold_range[1]
        )
        metrics = self._compute_metrics(
            nms_iou_threshold=nms_threshold,
            match_iou_threshold=match_threshold,
            conf_threshold=conf_threshold,
        )
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
        best_conf = self.study.best_params["conf_threshold"]
        best_metrics = self._compute_metrics(
            nms_iou_threshold=best_nms,
            match_iou_threshold=best_match,
            conf_threshold=best_conf,
        )

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
