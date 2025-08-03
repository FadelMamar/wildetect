import base64
import logging
import os
import sys
import traceback

# from PIL import Image
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List  # , Optional, Sequence

import litserve as ls

# import numpy as np
import torch

# from .utils import Detector
# import json
from fastapi import HTTPException
from PIL import Image

from wildetect.utils.utils import load_registered_model

from ..config import PredictionConfig
from ..data.detection import Detection
from ..factory import build_detector
from ..processor.processor import Classifier, RoIPostProcessor
from .object_detection_system import ObjectDetectionSystem

logger = logging.getLogger("Inference_service")


def setup_detector(config: PredictionConfig) -> ObjectDetectionSystem:
    """Set up the inference engine with model and processors."""

    mlflow_model_name = os.environ.get("MLFLOW_DETECTOR_NAME", None)
    mlflow_model_alias = os.environ.get("MLFLOW_DETECTOR_ALIAS", None)
    mlflow_roi_name = os.environ.get("MLFLOW_ROI_NAME", None)
    mlflow_roi_alias = os.environ.get("MLFLOW_ROI_ALIAS", None)

    if mlflow_model_name is None or mlflow_model_alias is None:
        logger.warning("MLFLOW_MODEL_NAME and MLFLOW_MODEL_ALIAS are not set")
        raise ValueError("MLFLOW_MODEL_NAME and MLFLOW_MODEL_ALIAS are not set")

    detector_model, metadata = load_registered_model(
        mlflow_model_name, mlflow_model_alias
    )
    roi_model, roi_metadata = load_registered_model(mlflow_roi_name, mlflow_roi_alias)

    if "batch" in metadata:
        config.batch_size = int(metadata.get("batch", config.batch_size))

    if "tilesize" in metadata:
        config.tilesize = int(metadata.get("tilesize", config.tilesize))

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

        detector = build_detector(
            config=config,
        )

        # Create object detection system
        detection_system = ObjectDetectionSystem(
            config=config,
        )
        detection_system.set_model(detector)

        if config.roi_weights or classifier:
            roi_processor = RoIPostProcessor(
                model_path=config.roi_weights,
                label_map=config.cls_label_map,
                feature_extractor_path=config.feature_extractor_path,
                roi_size=config.cls_imgsz,
                transform=config.transform,
                device=config.device,
                classifier=classifier,
                keep_classes=config.keep_classes,
            )
            detection_system.set_processor(roi_processor)

        logger.info("Detection pipeline setup completed")

        return detection_system

    except Exception:
        raise ValueError(f"Failed to setup inference engine: {traceback.format_exc()}")


class MyModelAPI(ls.LitAPI):
    def setup(
        self,
        device,
    ):
        """
        One-time initialization: load your model here.
        `device` is e.g. 'cuda:0' or 'cpu'.
        """
        logger.info(f"Device: {device}")
        config = PredictionConfig(
            device=device,
        )
        self.detection_system = setup_detector(config)

    def decode_request(self, request: dict) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """

        output = dict()

        try:
            img_tensor = request.get("tensor", None)
            shape = request.get("shape", None)
            if shape is None:
                raise ValueError("Shape not found in request")
            if img_tensor is None:
                raise ValueError("Tensor not found in request")

            tensor_bytes = base64.b64decode(img_tensor)
            tensor_bytes = bytearray(tensor_bytes)
            img_tensor = torch.frombuffer(tensor_bytes, dtype=torch.float32).reshape(
                shape
            )

            if len(shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            elif len(shape) < 3:
                raise ValueError("Invalid shape, expected a 3D or 4D tensor")

            output["images"] = img_tensor

        except Exception:
            raise HTTPException(status_code=400, detail=traceback.format_exc())

        return output

    def predict(self, x: dict) -> List[List[Detection]]:
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """

        try:
            batch = x["images"]
            results = self.detection_system.predict(batch)
            return results
        except Exception:
            raise ValueError(f"Prediction failed: {traceback.format_exc()}")

    def encode_response(self, output: List[List[Detection]]) -> dict:
        """
        Wrap the model output in a JSON-serializable dict.
        """
        logger.debug("sending response...")
        encoded_output = []
        for detection_list in output:
            o = [detection.to_dict() for detection in detection_list]
            encoded_output.append(o)

        return encoded_output
