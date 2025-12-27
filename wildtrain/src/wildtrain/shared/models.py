"""Pydantic models for CLI configuration validation.

DEPRECATED: This module is maintained for backward compatibility.
Please use wildtrain.shared.schemas for new code.

All models have been reorganized into the schemas package:
- wildtrain.shared.schemas.base - BaseConfig, enums
- wildtrain.shared.schemas.common - Shared primitives (transforms, curriculum, MLflow)
- wildtrain.shared.schemas.classification - Classification configs
- wildtrain.shared.schemas.yolo - YOLO-specific configs
- wildtrain.shared.schemas.detection - Detection configs
- wildtrain.shared.schemas.sweep - Hyperparameter sweep configs
- wildtrain.shared.schemas.pipeline - Pipeline configs
- wildtrain.shared.schemas.visualization - Visualization configs
- wildtrain.shared.schemas.registration - Model registration configs
"""

# Re-export everything from schemas for backward compatibility
from .schemas import *
