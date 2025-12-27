"""YOLO-specific configuration models."""

from typing import List, Dict, Optional
from pydantic import Field

from .base import BaseConfig


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
