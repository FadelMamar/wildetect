import base64
import datetime
import logging
import os
import sys
import time
import traceback

# from PIL import Image
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List  # , Optional, Sequence

import litserve as ls
import torch
from fastapi import HTTPException

from ..config import ROOT, PredictionConfig
from ..data.detection import Detection
from .object_detection_system import ObjectDetectionSystem


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    log_file = (
        ROOT
        / "logs"
        / "inference_service"
        / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# setup_logging()
logger = logging.getLogger("Inference_service")


class PredictionTimeLogger(ls.Callback):
    def on_before_predict(self, lit_api):
        t0 = time.perf_counter()
        self._start_time = t0

    def on_after_predict(self, lit_api):
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time
        logger.info(f"Prediction took {elapsed:.3f} seconds")


class InferenceService(ls.LitAPI):
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
        self.detection_system = ObjectDetectionSystem.from_mlflow(config)
        self.device = device

    def decode_request(self, request: dict) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """

        output = dict()

        try:
            # Set prediction config
            config = PredictionConfig.from_dict(request.get("config", {}))
            config.device = self.device
            self.detection_system.set_config(config)

            # Set image tensor
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
            return output

        except Exception:
            raise HTTPException(status_code=400, detail=traceback.format_exc())

    def predict(self, x: dict) -> List[List[Detection]]:
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """

        try:
            batch = x["images"]
            results = self.detection_system.predict(batch, local=True)
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

        return dict(detections=encoded_output)


def run_inference_server(port=4141, workers_per_device=1):
    api = InferenceService(max_batch_size=1, enable_async=False)

    server = ls.LitServer(
        api,
        workers_per_device=workers_per_device,
        accelerator="auto",
        fast_queue=True,
        callbacks=[PredictionTimeLogger()],
    )
    server.run(port=port, generate_client_file=False)
