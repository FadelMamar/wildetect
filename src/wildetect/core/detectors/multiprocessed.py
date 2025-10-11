import logging
import os
import queue
import time
import traceback
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from wildtrain.models.detector import Detector
from wildtrain.utils.mlflow import load_registered_model

from ..config import LoaderConfig, PredictionConfig
from ..data import Detection, DroneImage
from ..data.loader import DataLoader
from .base import DetectionPipeline

logger = logging.getLogger(__name__)


def _data_loading_worker(
    data_loader: DataLoader,
    data_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Standalone data loading worker function that can be pickled.

    Args:
        image_paths: List of image paths to process
        loader_config: Data loader configuration
        data_queue: Queue to put batches into
        stop_event: Event to signal stopping
        total_batches: Total number of batches to process
    """
    try:
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Loading data"):
            if stop_event.is_set():
                logger.info("Data loading process stopped by stop event")
                break

            # Add batch ID for tracking
            if isinstance(batch, dict):
                batch["batch_id"] = batch_idx

            # Put in queue with timeout
            while not stop_event.is_set():
                try:
                    data_queue.put(batch, timeout=1.0)
                    break
                except queue.Full:
                    time.sleep(0.1)
                except Exception:
                    stop_event.set()
                    raise Exception(
                        f"Error in data loading process: {traceback.format_exc()}"
                    )

            if stop_event.is_set():
                break

    except Exception as e:
        logger.error(f"{e}")

    finally:
        logger.info("Data loading process finished")
        data_queue.put(None)


def _detection_worker(
    worker_id: int,
    data_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    config: PredictionConfig,
    gpu_id: Optional[int] = None,
) -> None:
    """Standalone detection worker function that can be pickled.

    Args:
        worker_id: Unique identifier for this worker
        total_batches: Total number of batches to process
        data_queue: Queue to get batches from
        result_queue: Queue to put results into
        stop_event: Event to signal stopping
        config: Prediction configuration
        gpu_id: Optional GPU ID to assign to this worker
    """
    logger.info(f"Starting detection worker {worker_id}")

    # Initialize worker with its own model
    detection_system = _init_worker_model(config, gpu_id)

    processed_batches = []
    error_count = 0

    try:
        while not stop_event.is_set():
            try:
                batch = data_queue.get(timeout=0.1)
            except queue.Empty:
                if data_queue.empty() and stop_event.is_set():
                    break
                continue
            except Exception as e:
                stop_event.set()
                raise Exception(f"{traceback.format_exc()}")

            if batch is None:
                logger.info("Detection worker received end of data signal")
                break

            # Process batch
            try:
                batch_tensor = batch.pop("images")
                if config.inference_service_url is None:
                    detections = detection_system.predict(
                        batch_tensor, return_as_dict=False
                    )
                else:
                    detections = detection_system(batch_tensor)

                # Prepare result
                result = {
                    "batch_id": batch.get("batch_id"),
                    "tiles": batch.get("tiles"),
                    "detections": [
                        Detection.from_supervision(det) for det in detections
                    ],
                }

                processed_batches.append(result)
                result_queue.put(result, timeout=1.0)

            except Exception as e:
                logger.error(f"Worker {worker_id} failed to process batch: {e}")
                error_count += 1

                if error_count > 5:
                    logger.error(f"Too many errors. Stopping worker {worker_id}.")
                    stop_event.set()
                    raise Exception(
                        f"Error in detection worker {worker_id}: {traceback.format_exc()}"
                    )

    except Exception as e:
        stop_event.set()
        raise Exception(str(e))

    finally:
        logger.info(f"Detection worker {worker_id} finished")


def _init_worker_model(config: PredictionConfig, gpu_id: Optional[int] = None):
    """Initialize model for worker process.

    Args:
        config: Prediction configuration
        gpu_id: Optional GPU ID to assign to this worker

    Returns:
        Initialized detection system
    """
    # Set specific GPU for this process
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = config.device

    # Load model instance for this process
    if config.inference_service_url is None:
        detection_system = load_registered_model(
            alias=config.mlflow_model_alias,
            name=config.mlflow_model_name,
            load_unwrapped=True,
        )
        detection_system.set_device(device)
        return detection_system
    else:
        return partial(
            Detector.predict_inference_service,
            url=config.inference_service_url,
            timeout=config.timeout,
        )


class MultiProcessingDetectionPipeline(DetectionPipeline):
    """Multi-processing detection pipeline with separate data loading and detection processes."""

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        """Initialize the multi-processing detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
        """
        super().__init__(config=config, loader_config=loader_config)

        # Configure multiprocessing for the platform
        self.num_workers = getattr(config, "num_workers", 1)

        # Process-safe queues using torch.multiprocessing
        self.data_queue = mp.Queue(maxsize=config.queue_size)
        self.result_queue = mp.Queue()

        # Process control
        self.stop_event = mp.Event()
        self.data_process: Optional[mp.Process] = None
        self.worker_processes: List[mp.Process] = []
        self.detection_results: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized MultiProcessingDetectionPipeline with {self.num_workers} inference workers"
        )

    def run_detection(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> List[DroneImage]:
        """Run detection on images using multi-processing pipeline.

        Args:
            image_paths: List of image paths
            image_dir: Directory containing images
            save_path: Optional path to save results

        Returns:
            List of processed drone images with detections
        """
        logger.info("Starting multi-processing detection pipeline")
        assert self.num_workers > 0, f"Received {self.num_workers}"

        # Update config from metadata if available
        self.override_loading_config()

        # Create data loader to get total batch count
        data_loader = self.get_data_loader(
            image_paths=image_paths,
            image_dir=image_dir,
            use_tile_dataset=True,
        )

        total_batches = len(data_loader)
        logger.info(f"Total batches to process: {total_batches}")

        if total_batches == 0:
            logger.warning("No batches to process")
            return []

        # Reset stop event and results
        self.stop_event.clear()
        self.detection_results = []

        # Create progress bars
        detection_progress = tqdm(
            total=total_batches, desc="Processing batches", unit="batch", position=1
        )

        try:
            # Process workers
            self._init_workers()

            # load data
            _data_loading_worker(
                data_loader=data_loader,
                data_queue=self.data_queue,
                stop_event=self.stop_event,
            )

            # collect results
            all_batches = self._collect_results(
                detection_progress, total_batches=total_batches
            )

            # send stop signal
            self.stop_event.set()

            for worker in self.worker_processes:
                worker.join(timeout=1.0)

        except KeyboardInterrupt:
            self.stop_event.set()
        except Exception as e:
            logger.error(f"Error in multi-processing pipeline: {e}")
            self.stop_event.set()
            raise Exception(traceback.format_exc())
        finally:
            detection_progress.close()

        logger.info(
            f"Completed processing {len(all_batches)} batches with {self.error_count} errors"
        )

        # Post-processing
        all_drone_images = self._postprocess(batches=all_batches)
        if len(all_drone_images) == 0:
            logger.warning("No batches were processed")
            return []

        # Save results if path provided
        if save_path:
            self._save_results(all_drone_images, save_path)

        return all_drone_images

    def _collect_results(self, detection_progress: tqdm, total_batches: int):
        # Collect results from worker processes while they're running
        all_batches = []
        while len(all_batches) < total_batches:
            try:
                result = self.result_queue.get(timeout=1.0)
                all_batches.append(result)
                detection_progress.update(1)
            except queue.Empty:
                alive_workers = [w for w in self.worker_processes if w.is_alive()]
                if len(alive_workers) == 0:
                    break
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"{e}")
                break

        return all_batches

    def _init_workers(
        self,
    ):
        # Start worker processes using standalone functions
        for i in range(self.num_workers):
            gpu_id = (
                i % torch.cuda.device_count() if torch.cuda.is_available() else None
            )
            worker = mp.Process(
                target=_detection_worker,
                args=(
                    i,
                    self.data_queue,
                    self.result_queue,
                    self.stop_event,
                    self.config,
                    gpu_id,
                ),
                daemon=True,
            )
            worker.start()
            self.worker_processes.append(worker)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.

        Returns:
            Dictionary with pipeline information
        """
        info = super().get_pipeline_info()
        info.update(
            {
                "pipeline_type": "multiprocessing",
                "num_workers": self.num_workers,
                "queue_stats": {
                    "data_queue_size": self.data_queue.qsize(),
                    "result_queue_size": self.result_queue.qsize(),
                },
            }
        )

        return info
