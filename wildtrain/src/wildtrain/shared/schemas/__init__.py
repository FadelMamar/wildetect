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
    SweepDirectionTypes,
    SweepObjectiveTypes,
)

# Calibrator configurations
from .calibrator import (
    CalibrationConfig,
    CalibrationParametersConfig,
)

# Classification configurations
from .classification import (
    ClassificationConfig,
    ClassificationDatasetConfig,
    ClassificationEvalConfig,
    ClassificationEvalDatasetConfig,
    ClassifierCheckpointConfig,
    ClassifierModelConfig,
    ClassifierTrainingConfig,
)

# Common/shared configurations
from .common import (
    CurriculumConfig,
    DatasetStatsConfig,
    LoggingConfig,
    MLflowConfig,
    SingleClassConfig,
    TransformConfig,
    TransformsConfig,
)

# Detection configurations
from .detection import (
    DetectionConfig,
    DetectionEvalConfig,
    DetectionEvalParamsConfig,
    DetectionMetricsConfig,
    DetectionWeightsConfig,
)

# Pipeline configurations
from .pipeline import (
    ClassificationPipelineConfig,
    DetectionPipelineConfig,
    EvalPipelineConfig,
    PipelineConfig,
    TrainPipelineConfig,
)

# Registration configurations
from .registration import (
    ClassifierRegistrationConfig,
    DetectorRegistrationConfig,
    InferenceConfig,
    LocalizerRegistrationConfig,
    RegistrationBase,
)

# Sweep configurations
from .sweep import (
    ClassificationSweepConfig,
    ClassificationSweepModelParametersConfig,
    ClassificationSweepParametersConfig,
    ClassificationSweepTrainParametersConfig,
    DetectionSweepConfig,
    DetectionSweepModelParametersConfig,
    DetectionSweepParametersConfig,
    DetectionSweepTrainParametersConfig,
    SweepOutputConfig,
)

# Visualization configurations
from .visualization import (
    ClassificationVisualizationConfig,
    DetectionVisualizationConfig,
    FiftyOneConfig,
    LabelStudioConfig,
)

# YOLO configurations
from .yolo import (
    MergingMethodConfig,
    OverlapMetricConfig,
    YoloCurriculumConfig,
    YoloCustomConfig,
    YoloDatasetConfig,
    YoloInferenceConfig,
    YoloModelConfig,
    YoloPretrainingConfig,
    YoloTrainConfig,
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
    "YoloInferenceConfig",
    "YoloDatasetConfig",
    "YoloCurriculumConfig",
    "YoloPretrainingConfig",
    "YoloModelConfig",
    "YoloCustomConfig",
    "YoloTrainConfig",
    "OverlapMetricConfig",
    "MergingMethodConfig",
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
    # Calibrator
    "CalibrationParametersConfig",
    "CalibrationConfig",
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
