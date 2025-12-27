"""Schemas package - organized Pydantic configuration models.

This package provides modular schema definitions for the wildtrain library.
All models are re-exported here for convenient imports.

Example:
    from wildtrain.shared.schemas import BaseConfig, ClassificationConfig
    # or
    from wildtrain.shared.schemas.classification import ClassificationConfig
"""

# Base configuration and enums
from .base import (
    BaseConfig,
    SweepObjectiveTypes,
    SweepDirectionTypes,
)

# Common/shared configurations
from .common import (
    LoggingConfig,
    DatasetStatsConfig,
    TransformConfig,
    TransformsConfig,
    CurriculumConfig,
    SingleClassConfig,
    MLflowConfig,
)

# Classification configurations
from .classification import (
    ClassificationDatasetConfig,
    ClassifierModelConfig,
    ClassifierTrainingConfig,
    ClassifierCheckpointConfig,
    ClassificationConfig,
    ClassificationEvalDatasetConfig,
    ClassificationEvalConfig,
)

# YOLO configurations
from .yolo import (
    YoloConfig,
    YoloDatasetConfig,
    YoloCurriculumConfig,
    YoloPretrainingConfig,
    YoloModelConfig,
    YoloCustomConfig,
    YoloTrainConfig,
)

# Detection configurations
from .detection import (
    DetectionConfig,
    DetectionWeightsConfig,
    DetectionMetricsConfig,
    DetectionEvalParamsConfig,
    DetectionEvalConfig,
)

# Sweep configurations  
from .sweep import (
    SweepOutputConfig,
    ClassificationSweepModelParametersConfig,
    ClassificationSweepTrainParametersConfig,
    ClassificationSweepParametersConfig,
    ClassificationSweepConfig,
    DetectionSweepModelParametersConfig,
    DetectionSweepTrainParametersConfig,
    DetectionSweepParametersConfig,
    DetectionSweepConfig,
)

# Pipeline configurations
from .pipeline import (
    TrainPipelineConfig,
    EvalPipelineConfig,
    PipelineConfig,
    ClassificationPipelineConfig,
    DetectionPipelineConfig,
)

# Visualization configurations
from .visualization import (
    LabelStudioConfig,
    FiftyOneConfig,
    DetectionVisualizationConfig,
    ClassificationVisualizationConfig,
)

# Registration configurations
from .registration import (
    RegistrationBase,
    LocalizerRegistrationConfig,
    ClassifierRegistrationConfig,
    DetectorRegistrationConfig,
    InferenceConfig,
)


__all__ = [
    # Base
    "BaseConfig",
    "SweepObjectiveTypes",
    "SweepDirectionTypes",
    # Common
    "LoggingConfig",
    "DatasetStatsConfig",
    "TransformConfig",
    "TransformsConfig",
    "CurriculumConfig",
    "SingleClassConfig",
    "MLflowConfig",
    # Classification
    "ClassificationDatasetConfig",
    "ClassifierModelConfig",
    "ClassifierTrainingConfig",
    "ClassifierCheckpointConfig",
    "ClassificationConfig",
    "ClassificationEvalDatasetConfig",
    "ClassificationEvalConfig",
    # YOLO
    "YoloConfig",
    "YoloDatasetConfig",
    "YoloCurriculumConfig",
    "YoloPretrainingConfig",
    "YoloModelConfig",
    "YoloCustomConfig",
    "YoloTrainConfig",
    # Detection
    "DetectionConfig",
    "DetectionWeightsConfig",
    "DetectionMetricsConfig",
    "DetectionEvalParamsConfig",
    "DetectionEvalConfig",
    # Sweep
    "SweepOutputConfig",
    "ClassificationSweepModelParametersConfig",
    "ClassificationSweepTrainParametersConfig",
    "ClassificationSweepParametersConfig",
    "ClassificationSweepConfig",
    "DetectionSweepModelParametersConfig",
    "DetectionSweepTrainParametersConfig",
    "DetectionSweepParametersConfig",
    "DetectionSweepConfig",
    # Pipeline
    "TrainPipelineConfig",
    "EvalPipelineConfig",
    "PipelineConfig",
    "ClassificationPipelineConfig",
    "DetectionPipelineConfig",
    # Visualization
    "LabelStudioConfig",
    "FiftyOneConfig",
    "DetectionVisualizationConfig",
    "ClassificationVisualizationConfig",
    # Registration
    "RegistrationBase",
    "LocalizerRegistrationConfig",
    "ClassifierRegistrationConfig",
    "DetectorRegistrationConfig",
    "InferenceConfig",
]
