"""Compatibility re-exports.

API request models historically imported schemas from ``wildtrain.cli.models``.
The schemas live in ``wildtrain.shared.schemas``.
"""

from ..shared.schemas import (
    ClassificationConfig,
    ClassificationEvalConfig,
    ClassificationPipelineConfig,
    ClassificationVisualizationConfig,
    DetectionConfig,
    DetectionEvalConfig,
    DetectionPipelineConfig,
    DetectionVisualizationConfig,
)

__all__ = [
    "ClassificationConfig",
    "ClassificationEvalConfig",
    "ClassificationPipelineConfig",
    "ClassificationVisualizationConfig",
    "DetectionConfig",
    "DetectionEvalConfig",
    "DetectionPipelineConfig",
    "DetectionVisualizationConfig",
]
