import base64
import datetime
import logging
import os
import sys
import time
import traceback

from typing import List

import litserve as ls
import torch
from fastapi import HTTPException
from wildtrain.models.detector import Detector
from pydantic import BaseModel


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    log_file = (
        "logs"
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

class Request(BaseModel):
    tensor: str
    shape: List[int]

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

        mlflow_registry_name = os.environ.get("MLFLOW_REGISTRY_NAME")
        mlflow_alias = os.environ.get("MLFLOW_ALIAS")
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        for a in [mlflow_alias,mlflow_registry_name,mlflow_tracking_uri]:
            if a is None:
                raise ValueError("Some mlflow environment variables are not set.")

        logger.info(f"Loading model from mlflow: {mlflow_registry_name}@{mlflow_alias} from {mlflow_tracking_uri}")

        dwnd_location = os.environ.get("MLFLOW_LOCAL_DIR")
        self.detection_system = Detector.from_mlflow(name=mlflow_registry_name,
                                                    alias=mlflow_alias,
                                                    mlflow_tracking_uri=mlflow_tracking_uri,
                                                    dwnd_location=dwnd_location
                                                    )
        self.detection_system.set_device(device)

    def decode_request(self, request: Request) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """
        output = dict()
        try:
            # Set image tensor
            img_tensor = request.tensor
            shape = request.shape
            
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

    def predict(self, x: dict) -> dict:
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """
        try:
            batch = x["images"]
            results = self.detection_system.predict(batch, return_as_dict=True)
            return dict(detections=results)
        except Exception:
            raise HTTPException(status_code=400, detail=traceback.format_exc())

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
