"""Supervision metrics shared by detection evaluation."""

from supervision.metrics import MeanAveragePrecision, MeanAverageRecall
from supervision.metrics.core import AveragingMethod, MetricTarget


def build_detection_metrics(*, class_agnostic: bool, average: str) -> dict:
    """Box metrics used to score a detector outside the Ultralytics loop.

    ``average`` is ``macro``, ``micro``, or ``weighted``, matching
    ``DetectionMetricsConfig``.
    """
    # Imported lazily. evaluators/__init__ imports the evaluator, which imports
    # this function, so a top-level import here is circular.
    from ...evaluators.metrics import MyF1Score, MyPrecision, MyRecall

    boxes = MetricTarget.BOXES
    averaging = getattr(AveragingMethod, average.upper())
    return {
        "mAP": MeanAveragePrecision(
            boxes,
            class_agnostic=class_agnostic,
            class_mapping=None,
        ),
        "mAR": MeanAverageRecall(boxes),
        "precision": MyPrecision(boxes, averaging_method=averaging),
        "recall": MyRecall(boxes, averaging_method=averaging),
        "f1": MyF1Score(boxes, averaging_method=averaging),
    }
