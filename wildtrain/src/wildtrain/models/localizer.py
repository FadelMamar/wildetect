from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import supervision as sv
import torch
from ultralytics import YOLO

from ..shared.schemas.yolo import (
    MergingMethodConfig,
    OverlapMetricConfig,
    YoloInferenceConfig,
)


class ObjectLocalizer(ABC):
    """
    Abstract base class for object localizers.
    Should implement a forward method that returns bounding boxes for detected objects in a batch of images.
    """

    def __init__(self):
        self.metadata: Optional[Dict[str, Any]] = None

    @abstractmethod
    def predict(self, images: torch.Tensor) -> list[sv.Detections]:
        """Predict without Sahi algorithm"""
        pass

    @abstractmethod
    def predict_sahi(self, image: torch.Tensor, overlap_ratio_wh:tuple[float, float]=(0.2, 0.2), thread_workers:int=3)->list[sv.Detections]:
        """Predict using Sahi algorithm"""
        pass

    def forward(self, images: torch.Tensor) -> list[sv.Detections]:
        """Forward pass for the model"""
        return self.predict(images)

    @property
    def class_mapping(self):
        raise NotImplementedError("Subclasses must implement class_mapping")


class UltralyticsLocalizer(ObjectLocalizer):
    """
    Object localizer using Ultralytics YOLO models.
    Args:
        weights (str): Path to YOLO weights file or model name.
        device (str): Device to run inference on ('cpu' or 'cuda').
        conf_thres (float): Confidence threshold for detections.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        imgsz: int = 800,
        device: str = "cpu",
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        overlap_metric: str = "iou",
        merging_method: str = "nms",
        disable_detection_filtering: bool = False,
        task="detect",
        max_det=300,
        config: Optional[YoloInferenceConfig] = None,
    ):
        super().__init__()

        if config is not None:
            self.model = YOLO(config.weights, task=config.task)
            self.device = config.device
            self.conf_thres = config.conf_thres
            self.iou_thres = config.iou_thres
            self.max_det = config.max_det
            self.overlap_metric = OverlapMetricConfig(config.overlap_metric)
            self.imgsz = imgsz
            self.merging_method = (
                MergingMethodConfig(config.merging_method)
                if hasattr(config, "merging_method")
                else MergingMethodConfig.NMS
            )
            self.disable_detection_filtering = (
                config.disable_detection_filtering
                if hasattr(config, "disable_detection_filtering")
                else False
            )
        else:
            self.model = YOLO(weights, task=task)
            self.device = device
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.max_det = max_det
            self.overlap_metric = OverlapMetricConfig(overlap_metric)
            self.imgsz = imgsz
            self.merging_method = MergingMethodConfig(merging_method)
            self.disable_detection_filtering = disable_detection_filtering

        self.class_agnostic = True  # single class detection is localization
        assert self.overlap_metric in [OverlapMetricConfig.IOU, OverlapMetricConfig.IOS]
        self.overlap_metrics = {
            OverlapMetricConfig.IOU: sv.detection.utils.iou_and_nms.OverlapMetric.IOU,
            OverlapMetricConfig.IOS: sv.detection.utils.iou_and_nms.OverlapMetric.IOS,
        }

    @property
    def class_mapping(self):
        return self.model.names

    @classmethod
    def from_config(cls, config: YoloInferenceConfig):
        return cls(config=config)
    
    def _update_class_mapping(self, detections: sv.Detections)->sv.Detections:
        """Update metadata for detections."""
        detections.metadata["class_mapping"] = self.model.names
        if len(self.model.names) == 1:
            detections.class_id = detections.class_id + 1
        return detections

    def _apply_filtering(
        self,
        detections: sv.Detections,
    ) -> sv.Detections:
        """Apply filtering to remove duplicate predictions."""
        detections = self._update_class_mapping(detections)
        if self.merging_method == "nms":
            return detections.with_nms(
                threshold=self.iou_thres,
                class_agnostic=self.class_agnostic,
                overlap_metric=self.overlap_metrics[self.overlap_metric],
            )
        elif self.merging_method == "nmm":
            return detections.with_nmm(
                threshold=self.iou_thres,
                class_agnostic=self.class_agnostic,
                overlap_metric=self.overlap_metrics[self.overlap_metric],
            )
        else:
            raise ValueError(f"Invalid merging method: {self.merging_method}")

    def predict_sahi(self, image: torch.Tensor, 
        overlap_ratio_wh:tuple[float, float]=(0.2, 0.2),
        thread_workers:int=3,
        )->list[sv.Detections]:
        """
        Args:
            image (torch.Tensor): Image, shape (C, H, W)
            overlap_ratio_wh (tuple[float, float]): Overlap ratio in width and height
            thread_workers (int): Number of worker threads
        Returns:
            list[sv.Detections]: Detections for the image
        """        
        assert image.dim() == 3, "Image must be 3D tensor"
        
        def callback(image_slice: np.ndarray) -> sv.Detections:
            image_slice = torch.from_numpy(image_slice)
            return self.predict(image_slice)[0]

        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(self.imgsz, self.imgsz),
            overlap_ratio_wh=overlap_ratio_wh,
            thread_workers=thread_workers,
            iou_threshold=1.0,  # disable nms
            overlap_filter=None,
        )
        detections = slicer.slice(image.permute(1,2,0).cpu().numpy())
        return detections

    def predict(self, images: torch.Tensor) -> list[sv.Detections]:
        """
        Args:
            images (torch.Tensor): Batch of images, shape (B, C, H, W)
        Returns:
            List (length B) of detections per image. Each detection is a tuple:
                (x1, y1, x2, y2, score, class)
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        assert images.shape[1] == 3, f"Image must be in (B, C, H, W) format. Got {images.shape}"
        assert images.min() >= 0.0 and images.max() <= 1.0, (
            f"Image must be normalized to [0,1]. Got {images.min()} {images.max()}"
        )

        predictions = self.model.predict(
            images,
            imgsz=self.imgsz,
            verbose=False,
            conf=0.05,  # disable confidence filtering
            iou=1.0,  # disable nms
            device=self.device,
            max_det=self.max_det,
        )
        detections = [sv.Detections.from_ultralytics(pred) for pred in predictions]
        if not self.disable_detection_filtering:
            detections = [self._apply_filtering(det) for det in detections]
        return detections
