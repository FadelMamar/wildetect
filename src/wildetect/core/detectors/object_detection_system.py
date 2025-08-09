"""
Object Detection System for orchestrating detection pipeline.
"""

# Add async imports
import asyncio
import base64
import logging
import os
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional

import aiohttp
import requests
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToPILImage

from wildetect.utils.utils import load_registered_model

from ..config import ROOT, PredictionConfig
from ..data.detection import Detection
from ..factory import build_detector
from ..processor.processor import Classifier, RoIPostProcessor
from ..registry import Detector

logger = logging.getLogger("DETECTION_SYSTEM")


class ObjectDetectionSystem:
    """Main detection system that orchestrates model, processor, and batch handling."""

    def __init__(
        self,
        config: PredictionConfig,
    ):
        """Initialize the detection system.

        Args:
            config: Prediction configuration
            detection_label_map: Mapping from class IDs to class names
            buffer_size: Buffer size for batch processing
            timeout: Timeout for processing operations
        """
        self.config = config
        self.model: Optional[Detector] = None
        self.roi_processor: Optional[RoIPostProcessor] = None
        self.metadata: dict[str, str] = dict()

    @classmethod
    def from_config(cls, config: PredictionConfig) -> "ObjectDetectionSystem":
        try:
            # Build detector
            detector = build_detector(config=config)

            # Create object detection system
            detection_system = cls(config=config)
            detection_system.set_model(detector)

            if config.roi_weights:
                assert config.cls_label_map is not None, "cls_label_map is required"
                assert config.keep_classes is not None, "keep_classes is required"
                roi_processor = RoIPostProcessor(
                    model_path=config.roi_weights,
                    label_map=config.cls_label_map,
                    feature_extractor_path=config.feature_extractor_path,
                    roi_size=config.cls_imgsz,
                    transform=config.transform,
                    device=config.device,
                    classifier=None,
                    keep_classes=list(config.keep_classes),
                )
                detection_system.set_processor(roi_processor)
            logger.info("Detection system setup completed")

        except Exception:
            raise ValueError(
                f"Failed to setup inference engine: {traceback.format_exc()}"
            )

        return detection_system

    @classmethod
    def from_mlflow(cls, config: PredictionConfig) -> "ObjectDetectionSystem":
        """Set up the inference engine with model and processors."""

        load_dotenv(ROOT / ".env", override=False)

        mlflow_model_name = os.environ.get("MLFLOW_DETECTOR_NAME", None)
        mlflow_model_alias = os.environ.get("MLFLOW_DETECTOR_ALIAS", None)
        mlflow_roi_name = os.environ.get("MLFLOW_ROI_NAME", None)
        mlflow_roi_alias = os.environ.get("MLFLOW_ROI_ALIAS", None)

        if mlflow_model_name is None or mlflow_model_alias is None:
            raise ValueError(
                "MLFLOW_DETECTOR_NAME and MLFLOW_DETECTOR_ALIAS are not set"
            )

        if mlflow_roi_name is None or mlflow_roi_alias is None:
            logger.warning("MLFLOW_ROI_NAME and MLFLOW_ROI_ALIAS are not set")

        detector_model, metadata = load_registered_model(
            name=mlflow_model_name, alias=mlflow_model_alias, load_unwrapped=True
        )
        roi_model, roi_metadata = load_registered_model(
            name=mlflow_roi_name, alias=mlflow_roi_alias, load_unwrapped=True
        )

        classifier = None
        if roi_model is not None:
            classifier = Classifier(
                model=roi_model,
                model_path=None,
                label_map=config.cls_label_map,
                device=config.device,
                feature_extractor_path=config.feature_extractor_path,
            )
            box_size = roi_metadata.get("box_size", config.cls_imgsz)
            cls_imgsz_value = roi_metadata.get("cls_imgsz", config.cls_imgsz)
            config.cls_imgsz = int(cls_imgsz_value)
            logger.info(f"ROI box size: {box_size} -> {config.cls_imgsz}")

        try:
            # Build detector
            if detector_model is not None:
                config.model_path = detector_model.ckpt_path

            detector = build_detector(config=config)

            # Create object detection system
            detection_system = cls(config=config)
            detection_system.set_model(detector)
            detection_system.metadata = metadata

            if config.roi_weights or classifier:
                assert config.cls_label_map is not None, "cls_label_map is required"
                assert config.keep_classes is not None, "keep_classes is required"
                roi_processor = RoIPostProcessor(
                    model_path=config.roi_weights,
                    label_map=config.cls_label_map,
                    feature_extractor_path=config.feature_extractor_path,
                    roi_size=config.cls_imgsz,
                    transform=config.transform,
                    device=config.device,
                    classifier=classifier,
                    keep_classes=list(config.keep_classes),
                )
                detection_system.set_processor(roi_processor)

            logger.info("Detection system setup completed")

        except Exception:
            raise ValueError(
                f"Failed to setup inference engine: {traceback.format_exc()}"
            )

        return detection_system

    def set_model(self, model: Detector) -> None:
        """Set the detection model.

        Args:
            model: Detector instance
        """
        self.model = model
        logger.info(f"Set detection model: {model.__class__.__name__}")

    def set_config(self, config: PredictionConfig) -> None:
        """Set the prediction configuration.

        Args:
            config: Prediction configuration
        """
        self.config = config
        if hasattr(self.model, "config"):
            setattr(self.model, "config", config)

    def set_processor(self, roi_processor: Optional[RoIPostProcessor] = None) -> None:
        """Set the ROI post-processor.

        Args:
            roi_processor: ROI post-processor instance
        """
        assert isinstance(
            roi_processor, RoIPostProcessor
        ), "roi_processor must be a RoIPostProcessor instance"
        if roi_processor:
            self.roi_processor = roi_processor
            logger.info("Setting ROI post-processor")
        else:
            logger.info("No ROI post-processor set")

    def _postprocess(
        self, detections: List[List[Detection]], batch: torch.Tensor
    ) -> List[List[Detection]]:
        if self.roi_processor:
            processed_detections = []
            for i in range(batch.shape[0]):
                image = ToPILImage()(batch[i].cpu())
                d = self.roi_processor.run(
                    detections=detections[i],
                    image=image,
                    verbose=self.config.verbose,
                )
                processed_detections.append(d)
            return processed_detections
        else:
            return detections

    @staticmethod
    def preprocess_batch(batch: torch.Tensor) -> torch.Tensor:
        assert batch.ndim == 4, "Image must be a 4D tensor"
        B, C, H, W = batch.shape
        assert C == 3, "Image must have 3 channels"
        assert B >= 1, "Batch must have at least 1 image"
        if batch.max() > 1.0 and batch.min() >= 0.0:
            logger.debug(
                "Batch is not normalized. Normalize it. Expects values in [0, 1]"
            )
            batch = batch / 255.0
        return batch

    def predict(self, batch: torch.Tensor, local: bool = True) -> List[List[Detection]]:
        """Run prediction on a list of tiles."""

        if isinstance(self.config.inference_service_url, str) and not local:
            return self.predict_inference_service(batch, self.config)

        return self.predict_local_batch(batch)

    async def predict_async(
        self, batch: torch.Tensor, local: bool = False
    ) -> List[List[Detection]]:
        """Async version of predict method."""
        if isinstance(self.config.inference_service_url, str) and not local:
            return await self.predict_inference_service_async(batch, self.config)

        # For local prediction, run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_local_batch, batch)

    def predict_local(self, batch: torch.Tensor) -> List[List[Detection]]:
        if not self.model:
            raise RuntimeError("No model set. Call set_model() first.")

        batch = self.preprocess_batch(batch)
        B, C, H, W = batch.shape

        if B > self.config.batch_size:
            raise ValueError(
                f"Batch size {B} is greater than the model's batch size {self.config.batch_size}."
                "Use predict_local_batch instead."
            )

        # Run batch prediction
        batch_padded = self._pad_if_needed(
            batch,
            (self.config.batch_size, C, self.config.tilesize, self.config.tilesize),
        )
        detections = self.model.predict(batch_padded)[:B]
        detections = self._postprocess(detections=detections, batch=batch)
        if len(detections) != B:
            logger.error(
                f"Number of detections and images must match. {len(detections)} != {B}"
            )
            raise ValueError(
                f"Number of detections and images must match. {len(detections)} != {B}"
            )

        return detections

    def predict_local_batch(self, batch: torch.Tensor) -> List[List[Detection]]:
        if batch.shape[0] <= self.config.batch_size:
            return self.predict_local(batch)

        data = TensorDataset(batch)
        dataloader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False)
        detections = []
        for (batch,) in dataloader:
            detections.extend(self.predict_local(batch))
        return detections

    def _pad_if_needed(self, batch: torch.Tensor, out_shape: tuple) -> torch.Tensor:
        assert len(out_shape) == len(batch.shape)
        assert len(out_shape) == 4

        condition = any([a < b for a, b in zip(batch.shape, out_shape)])

        if condition:
            b, c, h, w = batch.shape
            padded = torch.zeros(out_shape)
            padded[:b, :c, :h, :w] = batch.clone()
            batch = padded

        return batch

    @staticmethod
    def predict_inference_service(
        batch: torch.Tensor, config: PredictionConfig
    ) -> List[List[Detection]]:
        batch = ObjectDetectionSystem.preprocess_batch(batch)

        as_bytes = batch.cpu().numpy().tobytes()
        payload = {
            "tensor": base64.b64encode(as_bytes).decode("utf-8"),
            "shape": list(batch.shape),
            "iou_nms": config.nms_iou,
            "conf": config.confidence_threshold,
            "config": deepcopy(config).to_dict(),
        }

        res = requests.post(
            url=config.inference_service_url, json=payload, timeout=config.timeout
        ).json()

        res = res.get("detections", "FAILED")
        if res == "FAILED":
            raise ValueError(f"Inference service failed: {res}")

        detections = []
        for detection_list in res:
            detections.append(
                [Detection.from_dict(detection) for detection in detection_list]
            )

        return detections

    @staticmethod
    async def predict_inference_service_async(
        batch: torch.Tensor, config: PredictionConfig
    ) -> List[List[Detection]]:
        """Async version of predict_inference_service."""
        if not config.inference_service_url:
            raise ValueError("Inference service URL is not configured")

        batch = ObjectDetectionSystem.preprocess_batch(batch)

        as_bytes = batch.cpu().numpy().tobytes()
        payload = {
            "tensor": base64.b64encode(as_bytes).decode("utf-8"),
            "shape": list(batch.shape),
            "iou_nms": config.nms_iou,
            "conf": config.confidence_threshold,
            "config": deepcopy(config).to_dict(),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=str(config.inference_service_url),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as response:
                res = await response.json()

        res = res.get("detections", "FAILED")
        if res == "FAILED":
            raise ValueError(f"Inference service failed: {res}")

        detections = []
        for detection_list in res:
            detections.append(
                [Detection.from_dict(detection) for detection in detection_list]
            )

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the detection system.

        Returns:
            Dictionary with system information
        """
        info = {
            "model_type": self.model.__class__.__name__ if self.model else None,
            "roi_processor": self.roi_processor.__class__.__name__
            if self.roi_processor
            else None,
            "config": {
                "tilesize": self.config.tilesize,
                "confidence_threshold": self.config.confidence_threshold,
                "device": self.config.device,
            },
        }

        if self.model:
            info.update(self.model.get_model_info())

        return info
