"""Compatibility re-exports.

API request models historically imported schemas from ``wildtrain.cli.models``.
The schemas live in ``wildtrain.shared.schemas``.
"""

from ..shared.schemas import (
    ClassificationConfig,
    ClassificationEvalConfig,
    ClassificationVisualizationConfig,
    DetectionConfig,
    DetectionEvalConfig,
    DetectionVisualizationConfig,
)

__all__ = [
    "ClassificationConfig",
    "ClassificationEvalConfig",
    "ClassificationVisualizationConfig",
    "DetectionConfig",
    "DetectionEvalConfig",
    "DetectionVisualizationConfig",
]
