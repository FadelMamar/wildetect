"""Pipeline configuration models for training and evaluation orchestration."""

from pathlib import Path
from typing import List
from pydantic import Field, field_validator

from .base import BaseConfig


class TrainPipelineConfig(BaseConfig):
    """Training pipeline configuration."""
    config: str = Field(description="Training config file path")
    debug: bool = Field(default=False, description="Debug mode")
    
    @field_validator('config')
    @classmethod
    def validate_config_file(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Training config file does not exist: {v}")
        return v


class EvalPipelineConfig(BaseConfig):
    """Evaluation pipeline configuration."""
    config: str = Field(description="Evaluation config file path")
    debug: bool = Field(default=False, description="Debug mode")
    
    @field_validator('config')
    @classmethod
    def validate_config_file(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Evaluation config file does not exist: {v}")
        return v


class PipelineConfig(BaseConfig):
    """Base pipeline configuration."""
    disable_train: bool = Field(default=False, description="Disable training pipeline")
    train: TrainPipelineConfig = Field(description="Training configuration")
    disable_eval: bool = Field(default=False, description="Disable evaluation pipeline")
    eval: EvalPipelineConfig = Field(description="Evaluation configuration")
    results_dir: str = Field(description="Results directory for pipeline outputs")
    
    @field_validator('results_dir')
    @classmethod
    def validate_results_dir(cls, v):
        # Ensure results directory exists or can be created
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('train', 'eval')
    @classmethod
    def validate_pipeline_configs(cls, v, info):
        # Validate that at least one pipeline is enabled
        if hasattr(info.data, 'disable_train') and hasattr(info.data, 'disable_eval'):
            if info.data.disable_train and info.data.disable_eval:
                raise ValueError("At least one pipeline (train or eval) must be enabled")
        return v
    
    def is_train_enabled(self) -> bool:
        """Check if training pipeline is enabled."""
        return not self.disable_train
    
    def is_eval_enabled(self) -> bool:
        """Check if evaluation pipeline is enabled."""
        return not self.disable_eval
    
    def get_enabled_pipelines(self) -> List[str]:
        """Get list of enabled pipeline names."""
        pipelines = []
        if self.is_train_enabled():
            pipelines.append("train")
        if self.is_eval_enabled():
            pipelines.append("eval")
        return pipelines
    
    def validate_pipeline_files_exist(self) -> None:
        """Validate that all referenced config files exist."""
        if self.is_train_enabled() and not Path(self.train.config).exists():
            raise ValueError(f"Training config file does not exist: {self.train.config}")
        if self.is_eval_enabled() and not Path(self.eval.config).exists():
            raise ValueError(f"Evaluation config file does not exist: {self.eval.config}")


class ClassificationPipelineConfig(PipelineConfig):
    """Classification pipeline configuration.
    
    This configuration manages the classification training and evaluation pipeline.
    It supports both training and evaluation phases with separate configurations.
    
    Example:
        config = ClassificationPipelineConfig(
            disable_train=False,
            train=TrainPipelineConfig(
                config=str("configs/classification/classification_train.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=str("configs/classification/classification_eval.yaml"),
                debug=False
            ),
            results_dir=str("results/classification")
        )
    """
    
    @field_validator('results_dir')
    @classmethod
    def validate_classification_results_dir(cls, v):
        # Ensure classification-specific results directory
        v = Path(v) / "classification" if Path(v).name != "classification" else v
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    def get_classification_results_path(self) -> str:
        """Get the classification-specific results path."""
        return Path(self.results_dir) / "classification" if Path(self.results_dir).name != "classification" else self.results_dir
    
    @classmethod
    def create_default(cls, results_dir: str = str("results/classification")) -> "ClassificationPipelineConfig":
        """Create a default classification pipeline configuration."""
        return cls(
            disable_train=False,
            train=TrainPipelineConfig(
                config=str("configs/classification/classification_train.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=str("configs/classification/classification_eval.yaml"),
                debug=False
            ),
            results_dir=results_dir
        )


class DetectionPipelineConfig(PipelineConfig):
    """Detection pipeline configuration.
    
    This configuration manages the detection training and evaluation pipeline.
    It supports both training and evaluation phases with separate configurations.
    
    Example:
        config = DetectionPipelineConfig(
            disable_train=False,
            train=TrainPipelineConfig(
                config=str("configs/detection/yolo_configs/yolo.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=str("configs/detection/yolo_configs/yolo_eval.yaml"),
                debug=False
            ),
            results_dir=str("results/yolo")
        )
    """
    
    @field_validator('results_dir')
    @classmethod
    def validate_detection_results_dir(cls, v):
        # Ensure detection-specific results directory
        v = Path(v) / "detection" if Path(v).name != "detection" else v
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    def get_detection_results_path(self) -> str:
        """Get the detection-specific results path."""
        return self.results_dir / "detection" if self.results_dir.name != "detection" else self.results_dir
    
    @classmethod
    def create_default(cls, results_dir: str = str("results/yolo")) -> "DetectionPipelineConfig":
        """Create a default detection pipeline configuration."""
        return cls(
            disable_train=False,
            train=TrainPipelineConfig(
                config=str("configs/detection/yolo_configs/yolo.yaml"),
                debug=False
            ),
            disable_eval=True,
            eval=EvalPipelineConfig(
                config=str("configs/detection/yolo_configs/yolo_eval.yaml"),
                debug=False
            ),
            results_dir=results_dir
        )
