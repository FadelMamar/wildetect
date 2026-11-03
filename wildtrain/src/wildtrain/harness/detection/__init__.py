"""Detection boundary inside the harness.

Ultralytics owns the inner training loop. This package owns backend
selection, dataset YAML preparation, MLflow setup for that loop, and the
supervision metrics used at evaluation.
"""

from .backends import (
    CustomYoloBackend,
    DetectionBackend,
    UltralyticsBackend,
    select_detection_backend,
)
from .data import resolve_detection_data_cfg
from .metrics import build_detection_metrics
from .tracking import configure_detection_tracking

__all__ = [
    "CustomYoloBackend",
    "DetectionBackend",
    "UltralyticsBackend",
    "build_detection_metrics",
    "configure_detection_tracking",
    "resolve_detection_data_cfg",
    "select_detection_backend",
]
