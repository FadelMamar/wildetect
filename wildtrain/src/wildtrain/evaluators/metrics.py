from supervision.metrics import (
    Precision,PrecisionResult,
    Recall,RecallResult,
    F1Score, F1ScoreResult
)

import numpy as np
from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import (
    box_iou_batch,
    mask_iou_batch,
    oriented_box_iou_batch,
)
from supervision.metrics.core import MetricTarget


class MyPrecisionResult(PrecisionResult):

    @property
    def precision_at_50(self) -> float:
        idx = np.where(np.isclose(self.iou_thresholds, 0.5))[0][0]
        return self.precision_scores[idx]

    @property
    def precision_at_75(self) -> float:
        idx = np.where(np.isclose(self.iou_thresholds, 0.75))[0][0]
        return self.precision_scores[idx]

class MyRecallResult(RecallResult):

    @property
    def recall_at_50(self) -> float:
        idx = np.where(np.isclose(self.iou_thresholds, 0.5))[0][0]
        return self.recall_scores[idx]

    @property
    def recall_at_75(self) -> float:
        idx = np.where(np.isclose(self.iou_thresholds, 0.75))[0][0]
        return self.recall_scores[idx] 

class MyF1ScoreResult(F1ScoreResult):
    
    @property
    def f1_at_50(self) -> float:
        idx = np.where(np.isclose(self.iou_thresholds, 0.5))[0][0]
        return self.f1_scores[idx]
    
    @property
    def f1_at_75(self) -> float:
        idx = np.where(np.isclose(self.iou_thresholds, 0.75))[0][0]
        return self.f1_scores[idx]
    
IOU_THRESHOLDS = np.array([0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])

class MyPrecision(Precision):

    def _compute(
        self, predictions_list: list[Detections], targets_list: list[Detections]
    ) -> PrecisionResult:
        iou_thresholds = IOU_THRESHOLDS
        stats = []

        for predictions, targets in zip(predictions_list, targets_list):
            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)

            if len(targets) > 0:
                if len(predictions) == 0:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=int),
                            targets.class_id,
                        )
                    )

                else:
                    if self._metric_target == MetricTarget.BOXES:
                        iou = box_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.MASKS:
                        iou = mask_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                        iou = oriented_box_iou_batch(
                            target_contents, prediction_contents
                        )
                    else:
                        raise ValueError(
                            "Unsupported metric target for IoU calculation"
                        )

                    matches = self._match_detection_batch(
                        predictions.class_id, targets.class_id, iou, iou_thresholds
                    )
                    stats.append(
                        (
                            matches,
                            predictions.confidence,
                            predictions.class_id,
                            targets.class_id,
                        )
                    )

        if not stats:
            return MyPrecisionResult(
                metric_target=self._metric_target,
                averaging_method=self.averaging_method,
                precision_scores=np.zeros(iou_thresholds.shape[0]),
                precision_per_class=np.zeros((0, iou_thresholds.shape[0])),
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
                small_objects=None,
                medium_objects=None,
                large_objects=None,
            )

        concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
        precision_scores, precision_per_class, unique_classes = (
            self._compute_precision_for_classes(*concatenated_stats)
        )

        return MyPrecisionResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            precision_scores=precision_scores,
            precision_per_class=precision_per_class,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )
    
class MyRecall(Recall):

    def _compute(
        self, predictions_list: list[Detections], targets_list: list[Detections]
    ) -> RecallResult:
        iou_thresholds = IOU_THRESHOLDS
        stats = []

        for predictions, targets in zip(predictions_list, targets_list):
            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)

            if len(targets) > 0:
                if len(predictions) == 0:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=int),
                            targets.class_id,
                        )
                    )

                else:
                    if self._metric_target == MetricTarget.BOXES:
                        iou = box_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.MASKS:
                        iou = mask_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                        iou = oriented_box_iou_batch(
                            target_contents, prediction_contents
                        )
                    else:
                        raise ValueError(
                            "Unsupported metric target for IoU calculation"
                        )

                    matches = self._match_detection_batch(
                        predictions.class_id, targets.class_id, iou, iou_thresholds
                    )
                    stats.append(
                        (
                            matches,
                            predictions.confidence,
                            predictions.class_id,
                            targets.class_id,
                        )
                    )

        if not stats:
            return MyRecallResult(
                metric_target=self._metric_target,
                averaging_method=self.averaging_method,
                recall_scores=np.zeros(iou_thresholds.shape[0]),
                recall_per_class=np.zeros((0, iou_thresholds.shape[0])),
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
                small_objects=None,
                medium_objects=None,
                large_objects=None,
            )

        concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
        recall_scores, recall_per_class, unique_classes = (
            self._compute_recall_for_classes(*concatenated_stats)
        )

        return MyRecallResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            recall_scores=recall_scores,
            recall_per_class=recall_per_class,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )

class MyF1Score(F1Score):

    def _compute(
        self, predictions_list: list[Detections], targets_list: list[Detections]
    ) -> F1ScoreResult:
        iou_thresholds = IOU_THRESHOLDS
        stats = []

        for predictions, targets in zip(predictions_list, targets_list):
            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)

            if len(targets) > 0:
                if len(predictions) == 0:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=int),
                            targets.class_id,
                        )
                    )

                else:
                    if self._metric_target == MetricTarget.BOXES:
                        iou = box_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.MASKS:
                        iou = mask_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                        iou = oriented_box_iou_batch(
                            target_contents, prediction_contents
                        )
                    else:
                        raise ValueError(
                            "Unsupported metric target for IoU calculation"
                        )

                    matches = self._match_detection_batch(
                        predictions.class_id, targets.class_id, iou, iou_thresholds
                    )
                    stats.append(
                        (
                            matches,
                            predictions.confidence,
                            predictions.class_id,
                            targets.class_id,
                        )
                    )

        if not stats:
            return MyF1ScoreResult(
                metric_target=self._metric_target,
                averaging_method=self.averaging_method,
                f1_scores=np.zeros(iou_thresholds.shape[0]),
                f1_per_class=np.zeros((0, iou_thresholds.shape[0])),
                iou_thresholds=iou_thresholds,
                matched_classes=np.array([], dtype=int),
                small_objects=None,
                medium_objects=None,
                large_objects=None,
            )

        concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
        f1_scores, f1_per_class, unique_classes = self._compute_f1_for_classes(
            *concatenated_stats
        )

        return MyF1ScoreResult(
            metric_target=self._metric_target,
            averaging_method=self.averaging_method,
            f1_scores=f1_scores,
            f1_per_class=f1_per_class,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            small_objects=None,
            medium_objects=None,
            large_objects=None,
        )