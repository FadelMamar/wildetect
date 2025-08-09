import base64
import datetime
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

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
    def setup(self, device):
        """
        One-time initialization: load your model here.
        `device` is e.g. 'cuda:0' or 'cpu'.
        """
        logger.info(f"Device: {device}")
        config = PredictionConfig(device=device)
        self.detection_system = ObjectDetectionSystem.from_mlflow(config)
        self.device = device

    def decode_request(self, request: dict) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """
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

            return {"images": img_tensor}

        except Exception:
            raise HTTPException(status_code=400, detail=traceback.format_exc())

    def batch(self, inputs):
        """
        Process a batch of inputs efficiently using ThreadPoolExecutor.
        This method is called by LitServe when batching is enabled.
        """

        def process_single_input(input_data):
            """Process a single input from the batch."""
            try:
                # Extract tensor data from input
                img_tensor = input_data["images"]
                return img_tensor
            except Exception as e:
                logger.error(f"Error processing input in batch: {e}")
                raise

        # Use ThreadPoolExecutor for parallel processing
        max_workers = max(len(inputs), os.cpu_count() // 2 or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            processed_inputs = list(pool.map(process_single_input, inputs))

        # Stack all tensors into a single batch
        batched_tensor = torch.stack(processed_inputs)
        lengths = [input_data.shape[0] for input_data in processed_inputs]
        return {"images": batched_tensor, "lengths": lengths}

    def predict(self, x: dict):
        """
        Run the model forward pass.
        Input `x` is the output of decode_request or batch.
        """

        try:
            batch = x["images"]
            lengths = x.get("lengths", None)
            with torch.inference_mode():
                results = self.detection_system.predict(batch, local=True)
                results = [
                    [detection.to_dict() for detection in detection_list]
                    for detection_list in results
                ]
            if lengths is None:
                return results
            else:
                return results, lengths
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    def unbatch(self, output: tuple):
        """
        Convert batch output back to individual outputs.
        This method is called by LitServe when batching is enabled.
        """
        assert isinstance(output, tuple), "Output must be a tuple"
        results, lengths = output
        detections = []
        i = 0
        for length in lengths:
            detections.append(results[i : i + length])
            i += length
        return detections

    def encode_response(self, output):
        """
        Wrap the model output in a JSON-serializable dict.
        """
        print(output)
        logger.debug("sending response...")
        return dict(detections=output)


def run_inference_server(port=4141, workers_per_device=1, max_batch_size=1):
    api = InferenceService(max_batch_size=max_batch_size, batch_timeout=0.01)

    server = ls.LitServer(
        api,
        workers_per_device=workers_per_device,
        accelerator="auto",
        fast_queue=True,
        callbacks=[PredictionTimeLogger()],
    )
    server.run(port=port, generate_client_file=False)
