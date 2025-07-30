"""
Object Detection System for orchestrating detection pipeline.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from ..config import PredictionConfig
from ..data.detection import Detection
from ..data.tile import Tile
from ..processor.processor import RoIPostProcessor
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

    def set_model(self, model: Detector) -> None:
        """Set the detection model.

        Args:
            model: Detector instance
        """
        self.model = model
        logger.info(f"Set detection model: {model.__class__.__name__}")

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

    def predict(self, batch: torch.Tensor) -> List[List[Detection]]:
        """Run prediction on a list of tiles."""
        if not self.model:
            raise RuntimeError("No model set. Call set_model() first.")

        assert batch.ndim == 4, "Image must be a 4D tensor"
        B, C, H, W = batch.shape
        assert C == 3, "Image must have 3 channels"
        assert B >= 1, "Batch must have at least 1 image"
        if batch.max() > 1.0 and batch.min() >= 0.0:
            logger.debug(
                "Batch is not normalized. Normalize it. Expects values in [0, 1]"
            )
            batch = batch / 255.0

        # Run batch prediction
        detections = self.model.predict(batch)
        detections = self._postprocess(detections=detections, batch=batch)
        if len(detections) != B:
            logger.error(
                f"Number of detections and images must match. {len(detections)} != {B}"
            )
            raise ValueError(
                f"Number of detections and images must match. {len(detections)} != {B}"
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
