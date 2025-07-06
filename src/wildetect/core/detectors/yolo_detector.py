"""
YOLO-based detector implementation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ..config import PredictionConfig
from ..data import Detection
from ..registry import Detector

logger = logging.getLogger(__name__)


class YOLODetector(Detector):
    """YOLO-based detector implementation."""

    def __init__(self, config: PredictionConfig):
        super().__init__(config)
        self.model_path = getattr(config, "model_path", None)
        self.device = config.device

        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the YOLO model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded YOLO model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Get class names from model
            if hasattr(self.model, "names"):
                self.class_names = self.model.names
            else:
                logger.warning(
                    "No class names found in the model. Using default class names."
                )
                self.class_names = {}

            self._is_loaded = True
            logger.info(f"YOLO detector initialized on device: {self.device}")

        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def predict_batch(self, batch: torch.Tensor) -> List[List[Detection]]:
        """Run prediction on an image.

        Args:
            image: Input image
            **kwargs: Additional prediction parameters

        Returns:
            List of detections
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        assert isinstance(batch, torch.Tensor), "Batch must be a tensor"

        try:
            # Run inference
            results = self.model.predict(
                batch,
                imgsz=self.config.tilesize,
                conf=self.config.confidence_threshold,
                verbose=self.config.verbose,
                device=self.device,
            )

            # Process results
            detections = [
                self._process_results(result, image.size)
                for result, image in zip(results, batch)
            ]

            return detections

        except Exception as e:
            logger.error(f"Error during YOLO prediction: {e}")
            raise

    def _process_results(self, result, image_size: tuple) -> List[Detection]:
        """Process YOLO results into Detection objects."""
        detections = []
        try:
            boxes = result.boxes
        except Exception:
            boxes = result.obb

        labels = boxes.cls
        confidence = boxes.conf.cpu().numpy()
        bbox = boxes.xyxy.cpu().numpy()

        detections = []
        for i, (box, conf, label) in enumerate(zip(bbox, confidence, labels)):
            detection = Detection(
                bbox=box.tolist(),
                confidence=conf,
                class_id=int(label),
                class_name=self.class_names.get(int(label)) or f"class_{int(label)}",
                metadata={
                    "model_type": self.model.__class__.__name__,
                    "image_size": image_size,
                },
            )
            detections.append(detection)
        return detections

    def predict(
        self,
        image: torch.Tensor,
    ) -> List[Detection]:
        """Run prediction on a batch of images.

        Args:
            images: List of input images
            **kwargs: Additional prediction parameters

        Returns:
            List of detection lists
        """
        assert isinstance(image, torch.Tensor), "Image must be a tensor"
        shape = image.shape
        assert len(shape) == 3, "Image must be a 3D tensor"
        C, H, W = shape
        assert C == 3, "Image must have 3 channels"

        return self.predict_batch(image.unsqueeze(0))[0]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.model.__class__.__name__,
            "model_path": self.model_path,
            "device": self.device,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
        }
