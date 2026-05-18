from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
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
        self.overlap_metrics = {
            OverlapMetricConfig.IOU: sv.detection.utils.iou_and_nms.OverlapMetric.IOU,
            OverlapMetricConfig.IOS: sv.detection.utils.iou_and_nms.OverlapMetric.IOS,
        }
        self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=self.model,
                confidence_threshold=self.conf_thres,
                image_size=self.imgsz,
                device=self.device,
            )

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
        if self.merging_method == MergingMethodConfig.NMS:
            return detections.with_nms(
                threshold=self.iou_thres,
                class_agnostic=self.class_agnostic,
                overlap_metric=self.overlap_metrics[self.overlap_metric],
            )
        elif self.merging_method == MergingMethodConfig.NMM:
            return detections.with_nmm(
                threshold=self.iou_thres,
                class_agnostic=self.class_agnostic,
                overlap_metric=self.overlap_metrics[self.overlap_metric],
            )
        elif self.merging_method == MergingMethodConfig.NONE:
            return detections
        else:
            raise ValueError(f"Invalid merging method: {self.merging_method}")

    def _sahi_to_supervision(self, sahi_result) -> sv.Detections:
        """Convert SAHI result to supervision Detections."""
        if len(sahi_result.object_prediction_list) == 0:
            return sv.Detections.empty()

        xyxy = np.array(
            [
                [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
                for p in sahi_result.object_prediction_list
            ]
        )
        confidence = np.array(
            [p.score.value for p in sahi_result.object_prediction_list]
        )
        class_ids = np.array(
            [p.category.id for p in sahi_result.object_prediction_list], dtype=int
        )

        # Create Detections object
        detections = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=confidence.astype(np.float32),
            class_id=class_ids,
        )

        # Add class mapping to metadata if available
        if sahi_result.object_prediction_list:
            category_mapping = {
                p.category.id: p.category.name
                for p in sahi_result.object_prediction_list
            }
            detections.metadata = {"class_mapping": category_mapping}
        return self._update_class_mapping(detections)

    def predict_sahi(
        self,
        image: torch.Tensor,
        overlap_ratio_wh: tuple[float, float] = (0.2, 0.2),
        batch_size: int = 8,
    ) -> list[sv.Detections]:
        """
        Args:
            image (torch.Tensor): Image, shape (C, H, W)
            overlap_ratio_wh (tuple[float, float]): Overlap ratio in width and height
        Returns:
            list[sv.Detections]: Detections for the image
        """
        assert image.dim() == 3, f"Image must be 3D tensor, got {image.shape}"
        assert image.min() >= 0.0 and image.max() <= 1.0, (
            f"Image must be normalized to [0,1]. Got {image.min()} {image.max()}"
        )

        # Convert torch tensor to numpy [0, 255] uint8
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Run sliced prediction
        result = get_sliced_prediction(
            image_np,
            self.sahi_model,
            slice_height=self.imgsz,
            slice_width=self.imgsz,
            overlap_height_ratio=overlap_ratio_wh[0],
            overlap_width_ratio=overlap_ratio_wh[1],
            postprocess_type=str(self.merging_method).upper(),
            postprocess_match_threshold=self.iou_thres,
            postprocess_match_metric=str(self.overlap_metric).upper(),
            verbose=0,
            batch_size=batch_size,
        )

        detections = self._sahi_to_supervision(result)
        return [detections]

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
            conf=self.conf_thres,
            iou=self.iou_thres
            device=self.device,
            max_det=self.max_det,
        )
        detections = [sv.Detections.from_ultralytics(pred) for pred in predictions]
        if not self.disable_detection_filtering:
            detections = [self._apply_filtering(det) for det in detections]
        return detections
