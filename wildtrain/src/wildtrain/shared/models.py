"""Pydantic models for CLI configuration validation."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
import yaml


class BaseConfig(BaseModel):
    """Base configuration class with common fields."""
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """Create config from YAML file."""
        if not Path(yaml_path).exists():
            raise ValueError(f"YAML file does not exist: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


class LoggingConfig(BaseConfig):
    """Logging configuration."""
    mlflow: bool = True
    log_dir: str = Field(default=str("./logs"), description="Logging directory")
    experiment_name: str = Field(default="wildtrain", description="MLflow experiment name")
    run_name: str = Field(default="default", description="MLflow run name")


class DatasetStatsConfig(BaseConfig):
    """Dataset statistics configuration."""
    mean: List[float] = Field(description="Mean values for normalization")
    std: List[float] = Field(description="Standard deviation values for normalization")
    
    @field_validator ('mean', 'std')
    @classmethod
    def validate_stats_length(cls, v):
        if len(v) != 3:
            raise ValueError("Mean and std must have exactly 3 values (RGB)")
        return v


class TransformConfig(BaseConfig):
    """Individual transform configuration."""
    name: str = Field(description="Transform name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Transform parameters")


class TransformsConfig(BaseConfig):
    """Transforms configuration for train and validation."""
    train: List[TransformConfig] = Field(default_factory=list, description="Training transforms")
    val: List[TransformConfig] = Field(default_factory=list, description="Validation transforms")


class CurriculumConfig(BaseConfig):
    """Curriculum learning configuration."""
    enabled: bool = Field(default=False, description="Enable curriculum learning")
    type: Literal["difficulty"] = Field(default="difficulty", description="Curriculum type")
    difficulty_strategy: Literal["linear"] = Field(default="linear", description="Difficulty strategy")
    start_difficulty: float = Field(default=0.0, ge=0.0, le=1.0, description="Starting difficulty")
    end_difficulty: float = Field(default=1.0, ge=0.0, le=1.0, description="Ending difficulty")
    warmup_epochs: int = Field(default=0, ge=0, description="Warmup epochs")
    log_frequency: int = Field(default=1, ge=1, description="Log frequency")
    
    @field_validator('end_difficulty')
    @classmethod
    def validate_end_difficulty(cls, v, info):
        if hasattr(info.data, 'start_difficulty') and v <= info.data.start_difficulty:
            raise ValueError("end_difficulty must be greater than start_difficulty")
        return v


class SingleClassConfig(BaseConfig):
    """Single class configuration for evaluation."""
    enable: bool = Field(description="Whether to enable single class mode")
    background_class_name: str = Field(description="Name of the background class")
    single_class_name: str = Field(description="Name of the single class")
    keep_classes: Optional[List[str]] = Field(default=None, description="Classes to keep (if None, all classes kept)")
    discard_classes: Optional[List[str]] = Field(default=None, description="Classes to discard")
    
    @field_validator('background_class_name', 'single_class_name')
    @classmethod
    def validate_class_names(cls, v):
        if not v or not v.strip():
            raise ValueError("Class name cannot be empty")
        return v.strip()

class ClassificationDatasetConfig(BaseConfig):
    """Dataset configuration."""
    root_data_directory: str = Field(description="Root data directory path")
    dataset_type: Literal["roi", "crop"] = Field(description="Dataset type")
    input_size: int = Field(gt=0, description="Input image size")
    batch_size: int = Field(gt=0, description="Batch size")
    num_workers: int = Field(default=0, ge=0, description="Number of workers")
    stats: DatasetStatsConfig = Field(description="Dataset statistics")
    transforms: TransformsConfig = Field(description="Data transforms")
    curriculum_config: Optional[CurriculumConfig] = Field(default=None, description="Curriculum configuration")
    single_class: Optional[SingleClassConfig] = Field(default=None, description="Single class configuration")
    rebalance: bool = Field(default=False, description="Enable dataset rebalancing")
    
    # Crop dataset specific fields
    crop_size: Optional[int] = Field(default=None, gt=0, description="Crop size for crop datasets")
    max_tn_crops: Optional[int] = Field(default=None, gt=0, description="Maximum true negative crops")
    p_draw_annotations: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Probability of drawing annotations")
    compute_difficulties: Optional[bool] = Field(default=None, description="Compute crop difficulties")
    preserve_aspect_ratio: Optional[bool] = Field(default=None, description="Preserve aspect ratio")
    
    @field_validator('root_data_directory')
    @classmethod
    def validate_data_directory(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Data directory does not exist: {v}")
        return v
    
    @field_validator('crop_size', 'max_tn_crops', 'p_draw_annotations', 'compute_difficulties', 'preserve_aspect_ratio')
    @classmethod
    def validate_crop_fields(cls, v, info):
        if hasattr(info.data, 'dataset_type') and info.data.dataset_type == 'crop':
            if v is None:
                raise ValueError(f"Field is required for crop dataset type")
        return v


class ClassifierModelConfig(BaseConfig):
    """Model configuration."""
    backbone: str = Field(description="Model backbone")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    backbone_source: str = Field(default="timm", description="Backbone source")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate")
    freeze_backbone: bool = Field(default=False, description="Freeze backbone")
    input_size: int = Field(gt=0, description="Model input size")
    mean: List[float] = Field(description="Normalization mean values")
    std: List[float] = Field(description="Normalization std values")
    weights: Optional[str] = Field(default=None, description="Model weights path")
    hidden_dim: int = Field(default=128, gt=0, description="Hidden dimension")
    num_layers: int = Field(default=2, gt=0, description="Number of layers")
    
    @field_validator('mean', 'std')
    @classmethod
    def validate_normalization(cls, v):
        if len(v) != 3:
            raise ValueError("Normalization values must have exactly 3 values (RGB)")
        return v


class ClassifierTrainingConfig(BaseConfig):
    """Training configuration."""
    batch_size: int = Field(gt=0, description="Training batch size")
    epochs: int = Field(gt=0, description="Number of training epochs")
    lr: float = Field(gt=0.0, description="Learning rate")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0, description="Label smoothing")
    weight_decay: float = Field(default=0.0001, ge=0.0, description="Weight decay")
    lrf: float = Field(default=0.1, gt=0.0, description="Learning rate factor")
    precision: str = Field(default="32", description="Training precision")
    accelerator: str = Field(default="auto", description="Training accelerator")
    num_workers: int = Field(default=0, ge=0, description="Number of workers")
    val_check_interval: int = Field(default=1, ge=1, description="Validation check interval")


class ClassifierCheckpointConfig(BaseConfig):
    """Checkpoint configuration."""
    monitor: str = Field(description="Metric to monitor")
    save_top_k: int = Field(default=1, ge=0, description="Number of best models to save")
    mode: Literal["min", "max"] = Field(description="Monitor mode")
    save_last: bool = Field(default=True, description="Save last checkpoint")
    dirpath: str = Field(description="Checkpoint directory")
    patience: int = Field(default=10, gt=0, description="Early stopping patience")
    save_weights_only: bool = Field(default=True, description="Save only weights")
    filename: str = Field(description="Checkpoint filename pattern")
    min_delta: float = Field(default=0.001, ge=0.0, description="Minimum improvement delta")


class MLflowConfig(BaseConfig):
    """MLflow configuration."""
    experiment_name: str = Field(default=None, description="MLflow experiment name")
    run_name: str = Field(default=None, description="MLflow run name")
    alias: str = Field(default=None, description="MLflow alias")
    name: str = Field(default=None, description="MLflow name")
    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    dwnd_location: Optional[str] = Field(default=None, description="DWND location")
    log_model: bool = Field(default=False, description="Log model")


class ClassificationConfig(BaseConfig):
    """Complete classification configuration."""
    dataset: ClassificationDatasetConfig = Field(description="Dataset configuration")
    model: ClassifierModelConfig = Field(description="Model configuration")
    train: ClassifierTrainingConfig = Field(description="Training configuration")
    checkpoint: ClassifierCheckpointConfig = Field(description="Checkpoint configuration")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    
    @field_validator('model')
    @classmethod
    def validate_model_dataset_compatibility(cls, v, info):
        if hasattr(info.data, 'dataset'):
            dataset = info.data.dataset
            if v.input_size != dataset.input_size:
                raise ValueError(f"Model input_size ({v.input_size}) must match dataset input_size ({dataset.input_size})")
        return v


class YoloConfig(BaseConfig):
    """YOLO model configuration."""
    weights: Optional[str] = Field(default=None, description="Model weights path")
    imgsz: int = Field(gt=0, description="Input image size")
    device: str = Field(default="cpu", description="Device to use")
    conf_thres: float = Field(default=0.2, ge=0.0, le=1.0, description="Confidence threshold")
    iou_thres: float = Field(default=0.3, ge=0.0, le=1.0, description="IoU threshold")
    max_det: int = Field(default=300, gt=0, description="Maximum detections")
    overlap_metric: str = Field(default="IOU", description="Overlap metric")
    task: str = Field(default="detect", description="YOLO task type (detect, classify, segment)")
    

class MMDetConfig(BaseConfig):
    """MMDetection model configuration."""
    config: str = Field(description="MMDetection config file path")
    weights: Optional[str] = Field(default=None, description="Model checkpoint path")
    device: str = Field(default="cpu", description="Device to use")
    
    @field_validator('config')
    @classmethod
    def validate_config_file(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Config file does not exist: {v}")
        return v


class YoloDatasetConfig(BaseConfig):
    """YOLO dataset configuration."""
    data_cfg: Optional[str] = Field(default=None, description="str to data configuration file")
    load_as_single_class: bool = Field(default=True, description="Load dataset as single class")
    root_data_directory: str = Field(default="", description="Root data directory")
    force_merge: bool = Field(default=False, description="Force merge")
    keep_classes: Optional[List[str]] = Field(default=None, description="Keep classes")
    discard_classes: Optional[List[str]] = Field(default=None, description="Discard classes")

class YoloCurriculumConfig(BaseConfig):
    """YOLO curriculum learning configuration."""
    data_cfg: Optional[str] = Field(default=None, description="Curriculum data configuration")
    ratios: List[float] = Field(default_factory=list, description="Curriculum ratios")
    epochs: List[int] = Field(default_factory=list, description="Curriculum epochs")
    freeze: List[int] = Field(default_factory=list, description="Curriculum freeze layers")
    lr0s: List[float] = Field(default_factory=list, description="Curriculum learning rates")
    save_dir: Optional[str] = Field(default=None, description="Curriculum save directory")

class YoloPretrainingConfig(BaseConfig):
    """YOLO pretraining configuration."""
    data_cfg: Optional[str] = Field(default=None, description="Pretraining data configuration")
    epochs: int = Field(default=10, description="Pretraining epochs")
    lr0: float = Field(default=0.0001, description="Pretraining learning rate")
    lrf: float = Field(default=0.1, description="Pretraining learning rate factor")
    freeze: int = Field(default=0, description="Pretraining freeze layers")
    save_dir: Optional[str] = Field(default=None, description="Pretraining save directory")

class YoloModelConfig(BaseConfig):
    """YOLO model configuration."""
    pretrained: bool = Field(default=True, description="Use pretrained model")
    weights: Optional[str] = Field(default=None, description="Model weights path")
    architecture_file: Optional[str] = Field(default=None, description="Model architecture file")

class YoloCustomConfig(BaseConfig):
    """YOLO custom configuration."""
    image_encoder_backbone: str = Field(default="timm/vit_base_patch14_dinov2.lvd142m", description="Image encoder backbone")
    image_encoder_backbone_source: str = Field(default="timm", description="Image encoder backbone source")
    count_regressor_layers: int = Field(default=13, description="Count regressor layers")
    area_regressor_layers: int = Field(default=10, description="Area regressor layers")
    roi_classifier_layers: Dict[str, int] = Field(default_factory=dict, description="ROI classifier layers")
    fp_tp_loss_weight: float = Field(default=0.5, description="FP/TP loss weight")
    count_loss_weight: float = Field(default=0.5, description="Count loss weight")
    area_loss_weight: float = Field(default=0.25, description="Area loss weight")
    box_size: int = Field(default=224, description="Box size")

class YoloTrainConfig(BaseConfig):
    """YOLO training configuration."""
    batch: int = Field(description="Training batch size")
    epochs: int = Field(description="Number of training epochs")
    optimizer: str = Field(default="AdamW", description="Optimizer")
    lr0: float = Field(description="Initial learning rate")
    lrf: float = Field(description="Learning rate factor")
    momentum: float = Field(default=0.937, description="Momentum")
    weight_decay: float = Field(default=0.0005, description="Weight decay")
    warmup_epochs: int = Field(default=1, description="Warmup epochs")
    cos_lr: bool = Field(default=True, description="Cosine learning rate")
    patience: int = Field(default=10, description="Patience")
    iou: float = Field(default=0.65, description="IoU threshold")
    imgsz: int = Field(description="Image size")
    
    # Loss weights
    box: float = Field(default=3.5, description="Box loss weight")
    cls: float = Field(default=1.0, description="Class loss weight")
    dfl: float = Field(default=1.5, description="DFL loss weight")
    
    device: str = Field(default="cpu", description="Device")
    workers: int = Field(default=0, description="Number of workers")
    
    # Augmentations
    degrees: float = Field(default=45.0, description="Rotation degrees")
    mixup: float = Field(default=0.0, description="Mixup probability")
    cutmix: float = Field(default=0.5, description="Cutmix probability")
    shear: float = Field(default=10.0, description="Shear")
    copy_paste: float = Field(default=0.0, description="Copy paste probability")
    erasing: float = Field(default=0.0, description="Erasing probability")
    scale: float = Field(default=0.2, description="Scale")
    fliplr: float = Field(default=0.5, description="Horizontal flip probability")
    flipud: float = Field(default=0.5, description="Vertical flip probability")
    hsv_h: float = Field(default=0.0, description="HSV hue")
    hsv_s: float = Field(default=0.1, description="HSV saturation")
    hsv_v: float = Field(default=0.1, description="HSV value")
    translate: float = Field(default=0.2, description="Translation")
    mosaic: float = Field(default=0.0, description="Mosaic probability")
    multi_scale: bool = Field(default=False, description="Multi-scale")
    perspective: float = Field(default=0.0, description="Perspective")
    
    deterministic: bool = Field(default=False, description="Deterministic")
    seed: int = Field(default=41, description="Random seed")
    freeze: int = Field(default=9, description="Freeze layers")
    cache: bool = Field(default=False, description="Cache")


class DetectionConfig(BaseConfig):
    """Complete detection configuration matching the YAML structure."""
    dataset: YoloDatasetConfig = Field(description="Dataset configuration")
    curriculum: YoloCurriculumConfig = Field(description="Curriculum learning configuration")
    pretraining: YoloPretrainingConfig = Field(description="Pretraining configuration")
    model: YoloModelConfig = Field(description="Model configuration")
    name: str = Field(description="Run name")
    project: str = Field(description="Project name")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    use_custom_yolo: bool = Field(default=False, description="Use custom YOLO")
    custom_yolo_kwargs: YoloCustomConfig = Field(description="Custom YOLO configuration")
    train: YoloTrainConfig = Field(description="Training configuration")

class LabelStudioConfig(BaseConfig):
    """Label Studio configuration."""
    url: str = Field(default="http://localhost:8080", description="Label Studio URL")
    api_key: str = Field(default=None, description="Label Studio API key")
    project_id: int = Field(default=1, description="Label Studio project ID")
    model_tag: str = Field(default="version-demo", description="Model tag")

class FiftyOneConfig(BaseConfig):
    """FiftyOne configuration."""
    dataset_name: Optional[str] = Field(default=None, description="FiftyOne dataset name")
    prediction_field: Optional[str] = Field(default=None, description="Prediction field name")

class DetectionVisualizationConfig(BaseConfig):
    """Visualization configuration."""
    fiftyone: FiftyOneConfig = Field(description="FiftyOne configuration")
    localizer: YoloConfig = Field(description="Localizer configuration")
    classifier_weights: Optional[str] = Field(default=None, description="Classifier weights path")
    batch_size: int = Field(gt=0, description="Processing batch size")
    debug: bool = Field(default=False, description="Debug mode")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    label_studio: LabelStudioConfig = Field(description="Label Studio configuration")

class ClassificationVisualizationConfig(BaseConfig):
    """Classification visualization configuration.
    """
    dataset_name: str = Field(description="Name of the FiftyOne dataset to use or create")
    weights: str = Field(description="str to the classifier checkpoint (.ckpt) file")
    prediction_field: str = Field(default="classification_predictions", description="Field name to store predictions in FiftyOne samples")
    batch_size: int = Field(default=32, description="Batch size for prediction inference")
    device: str = Field(default="cpu", description="Device to run inference on (e.g., 'cpu' or 'cuda')")
    debug: bool = Field(default=False, description="If set, only process a small number of samples for debugging")
    mlflow: MLflowConfig = Field(description="MLflow configuration")
    label_to_class_map: Optional[dict] = Field(default=None, description="Label to class map")
        
    @field_validator('weights')
    @classmethod
    def validate_checkpoint_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Classifier checkpoint does not exist: {v}")
        return v
      
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v
    
    @field_validator('dataset_name')
    @classmethod
    def validate_dataset_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()
    
    @field_validator('prediction_field')
    @classmethod
    def validate_prediction_field(cls, v):
        if not v or not v.strip():
            raise ValueError("Prediction field name cannot be empty")
        return v.strip()

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


class ClassificationEvalConfig(BaseConfig):
    """Classification evaluation configuration.
    
    This configuration is specifically for evaluation workflows and has a simpler
    structure compared to training configurations.
    
    Example:
        config = ClassificationEvalConfig(
            classifier="path/to/checkpoint.ckpt",
            split="val",
            device="cpu",
            batch_size=4,
            dataset=ClassificationEvalDatasetConfig(
                root_data_directory="path/to/data",
                single_class=SingleClassConfig(
                    enable=True,
                    background_class_name="background",
                    single_class_name="wildlife",
                    keep_classes=None,
                    discard_classes=["vegetation", "termite mound", "rocks", "other", "label"]
                )
            )
        )
    """
    classifier: str = Field(description="str to the classifier checkpoint (.ckpt) file")
    split: Literal["train", "val", "test"] = Field(default="val", description="Dataset split to evaluate on")
    device: str = Field(default="cpu", description="Device to run evaluation on (cpu, cuda)")
    batch_size: int = Field(default=4, description="Batch size for evaluation")
    dataset: "ClassificationEvalDatasetConfig" = Field(description="Dataset configuration for evaluation")

    
    @field_validator('classifier')
    @classmethod
    def validate_checkpoint_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Classifier checkpoint does not exist: {v}")
        return v
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        valid_devices = ["cpu", "cuda", "cuda:0", "cuda:1"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}, got: {v}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v


class ClassificationEvalDatasetConfig(BaseConfig):
    """Dataset configuration for classification evaluation."""
    root_data_directory: str = Field(description="Root directory containing the dataset")
    single_class: Optional["SingleClassConfig"] = Field(default=None, description="Single class configuration")
    
    @field_validator('root_data_directory')
    @classmethod
    def validate_data_directory_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Dataset directory does not exist: {v}")
        return v


class DetectionEvalConfig(BaseConfig):
    """Detection evaluation configuration.
    
    This configuration is specifically for evaluation workflows and has a different
    structure compared to training configurations.
    
    """
    weights: "DetectionWeightsConfig" = Field(description="Model weights configuration")
    dataset: "YoloDatasetConfig" = Field(description="Dataset configuration")
    device: str = Field(default="cpu", description="Device to run evaluation on")
    metrics: "DetectionMetricsConfig" = Field(description="Evaluation metrics configuration")
    eval: "DetectionEvalParamsConfig" = Field(description="Evaluation parameters")   
    results_dir: str = Field(description="Results directory for pipeline outputs")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        assert (v == "cpu") or ("cuda" in v), f"Device must be one of ['cpu', 'cuda'], got: {v}"
        return v


class DetectionWeightsConfig(BaseConfig):
    """Weights configuration for detection evaluation."""
    localizer: str = Field(description="str to the localizer weights file")
    classifier: Optional[str] = Field(default=None, description="str to the classifier weights file (optional)")
    
    @field_validator('localizer')
    @classmethod
    def validate_localizer_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Localizer weights file does not exist: {v}")
        return v
    
    @field_validator('classifier')
    @classmethod
    def validate_classifier_exists(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"Classifier weights file does not exist: {v}")
        return v


class DetectionMetricsConfig(BaseConfig):
    """Metrics configuration for detection evaluation."""
    average: Literal["macro", "micro", "weighted"] = Field(default="macro", description="Averaging method for metrics")
    class_agnostic: bool = Field(default=False, description="Whether to use class-agnostic evaluation")


class DetectionEvalParamsConfig(BaseConfig):
    """Evaluation parameters for detection."""
    imgsz: int = Field(default=640, description="Image size for evaluation")
    split: Literal["train", "val", "test"] = Field(default="val", description="Split to evaluate on")
    iou: float = Field(default=0.6, description="IoU threshold for evaluation")
    single_cls: bool = Field(default=True, description="Treat dataset as single-class")
    half: bool = Field(default=False, description="Use half precision")
    batch_size: int = Field(default=8, description="Batch size for DataLoader")
    num_workers: int = Field(default=0, description="Number of DataLoader workers")
    rect: bool = Field(default=False, description="Use rectangular batches")
    stride: int = Field(default=32, description="Model stride")
    task: Literal["detect", "classify", "segment"] = Field(default="detect", description="Task type")
    classes: Optional[List[int]] = Field(default=None, description="Optionally restrict to specific class indices")
    cache: bool = Field(default=False, description="Use cache for images/labels")
    multi_modal: bool = Field(default=False, description="Not using multi-modal data")
    conf: float = Field(default=0.1, description="Confidence threshold for evaluation")
    max_det: int = Field(default=300, description="Maximum detections per image")
    verbose: bool = Field(default=False, description="Verbosity level")
    augment: bool = Field(default=False, description="Use Test Time Augmentation")
    
    @field_validator('imgsz')
    @classmethod
    def validate_imgsz(cls, v):
        if v <= 0:
            raise ValueError(f"Image size must be positive, got: {v}")
        return v
    
    @field_validator('iou')
    @classmethod
    def validate_iou(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"IoU must be between 0.0 and 1.0, got: {v}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v
    
    @field_validator('num_workers')
    @classmethod
    def validate_num_workers(cls, v):
        if v < 0:
            raise ValueError(f"Number of workers must be non-negative, got: {v}")
        return v
    
    @field_validator('stride')
    @classmethod
    def validate_stride(cls, v):
        if v <= 0:
            raise ValueError(f"Stride must be positive, got: {v}")
        return v
    
    @field_validator('conf')
    @classmethod
    def validate_conf(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {v}")
        return v
    
    @field_validator('max_det')
    @classmethod
    def validate_max_det(cls, v):
        if v <= 0:
            raise ValueError(f"Max detections must be positive, got: {v}")
        return v




class RegistrationBase(BaseConfig):
    name: Optional[str] = Field(default=None, description="Model name for registration")
    batch_size: int = Field(default=8, gt=0, description="Batch size for inference")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    export_format: str = Field(default="pt", description="export format")
    dynamic:bool = Field(default=False, description="handle different sizes")

    @field_validator('export_format')
    @classmethod
    def validate_export_format(cls, v):
        valid_formats = ["torchscript", "openvino", "onnx", "pt"]
        if v not in valid_formats:
            raise ValueError(f"Export format must be one of {valid_formats}, got: {v}")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError(f"Batch size must be positive, got: {v}")
        return v



class LocalizerRegistrationConfig(BaseConfig):
    """Configuration for registering a detection model to MLflow Model Registry.
    
    This configuration is specifically for registering localizer models.
    """
    yolo: Optional[YoloConfig] = Field(None,description="yolo config")
    mmdet: Optional[MMDetConfig] = Field(None, description="mmdet config")
    processing: RegistrationBase = Field(description="processing information")


class ClassifierRegistrationConfig(BaseConfig):
    """Configuration for registering a classification model to MLflow Model Registry."""
    weights: Optional[str] = Field(default=None,description="Path to the model checkpoint file")
    processing: RegistrationBase = Field(description="processing information")
        
    

class DetectorRegistrationConfig(BaseConfig):
    """Base configuration for model registration.
    
    This is a union configuration that can handle both detector and classifier registration.
    """
    localizer: LocalizerRegistrationConfig = Field(description="Detector registration configuration")
    classifier: ClassifierRegistrationConfig = Field(description="Classifier registration configuration")
    processing: RegistrationBase = Field(description="processing information")
        
class InferenceConfig(BaseConfig):
    port: int = Field(default=4141, description="Port to run the server on")
    workers_per_device: int = Field(default=1, description="Number of workers per device")
    mlflow_registry_name: str = Field(default="detector", description="MLflow registry name")
    mlflow_alias: str = Field(default="demo", description="MLflow alias")
    mlflow_local_dir: str = Field(default="models-registry", description="MLflow local directory")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")


# Update forward references
ClassificationEvalConfig.model_rebuild()
ClassificationEvalDatasetConfig.model_rebuild()
SingleClassConfig.model_rebuild()
DetectionEvalConfig.model_rebuild()
DetectionWeightsConfig.model_rebuild()
DetectionMetricsConfig.model_rebuild()
DetectionEvalParamsConfig.model_rebuild()
ClassificationVisualizationConfig.model_rebuild()
LocalizerRegistrationConfig.model_rebuild()
ClassifierRegistrationConfig.model_rebuild()
DetectorRegistrationConfig.model_rebuild()
InferenceConfig.model_rebuild()
