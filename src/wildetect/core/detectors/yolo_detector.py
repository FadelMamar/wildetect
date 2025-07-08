"""
YOLO-based detector implementation.
"""

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ...utils.utils import load_registered_model
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
        self.metadata = dict()

        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the YOLO model."""
        model, metadata = self.load_from_mlflow()
        if model and metadata:
            self.model, self.metadata = model, metadata
        elif self.model_path and os.path.exists(str(self.model_path)):
            self.model = YOLO(self.model_path, task="detect")
            logger.info(f"Loaded YOLO model from {self.model_path}")
        else:
            raise FileNotFoundError(
                f"Model file not found: {self.model_path} and not found in MLflow. Set env variables MLFLOW_MODEL_NAME and MLFLOW_MODEL_ALIAS"
            )
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

    def _predict_batch(self, batch: torch.Tensor) -> List[List[Detection]]:
        """Run prediction on an image.

        Args:
            image: Input image
            **kwargs: Additional prediction parameters

        Returns:
            List of detections
        """

        assert batch.ndim == 4, "Batch must be a 4D tensor"
        B, C, H, W = batch.shape
        assert C == 3, "Batch must have 3 channels"
        assert B >= 1, "Batch must have at least 1 image"

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
            detections = [self._process_results(result, W, H) for result in results]

            return detections

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during YOLO prediction: {e}")
            raise

    def _process_results(
        self, result, image_width: int, image_height: int
    ) -> List[Detection]:
        """Process YOLO results into Detection objects."""
        detections = []
        try:
            boxes = result.boxes
            if boxes is None:
                raise ValueError("No boxes found in result")
        except ValueError:
            boxes = result.obb
        except Exception:
            raise Exception(f"Error processing results: {result}")

        labels = boxes.cls.int().cpu().tolist()
        confidence = boxes.conf.cpu().tolist()
        bbox = boxes.xyxy.round().int().cpu().tolist()

        detections = []
        for i, (box, conf, label) in enumerate(zip(bbox, confidence, labels)):
            logger.debug(f"Processing detection {i}: {box}, {conf}, {label}")
            detection = Detection(
                bbox=box,
                confidence=conf,
                class_id=label,
                class_name=self.class_names.get(label) or f"class_{label}",
                metadata={
                    "model_type": self.model.__class__.__name__,
                    "image_width": image_width,
                    "image_height": image_height,
                },
            )
            detection.clamp_bbox(x_range=(0, image_width), y_range=(0, image_height))

            detections.append(detection)
        return detections

    def predict(
        self,
        image: torch.Tensor,
    ) -> List[List[Detection]]:
        """Run prediction on a batch of images.

        Args:
            images: List of input images
            **kwargs: Additional prediction parameters

        Returns:
            List of detection lists
        """
        assert isinstance(image, torch.Tensor), "Image must be a tensor"
        shape = image.shape
        if len(shape) == 4:
            return self._predict_batch(image)
        elif len(shape) == 3:
            return self._predict_batch(image.unsqueeze(0))
        else:
            raise ValueError(f"Invalid image shape: {shape}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.model.__class__.__name__,
            "model_path": self.model_path,
            "device": self.device,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "metadata": self.metadata,
        }
