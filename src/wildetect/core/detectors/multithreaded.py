import logging
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ..config import LoaderConfig, PredictionConfig
from ..data import DroneImage
from ..data.loader import DataLoader
from .base import DetectionPipeline
from .utils import BatchQueue

logger = logging.getLogger(__name__)


class MultiThreadedDetectionPipeline(DetectionPipeline):
    """Multi-threaded detection pipeline with separate data loading and detection threads."""

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        """Initialize the multi-threaded detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
            queue_size: Maximum number of batches in the queue
        """

        super().__init__(config, loader_config)

        # Thread-safe queues
        self.data_queue = BatchQueue(maxsize=config.queue_size)
        self.result_queue = BatchQueue(maxsize=config.queue_size * 2)

        # Thread control
        self.stop_event = threading.Event()
        self.data_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_result: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized MultiThreadedDetectionPipeline with model_type={config.model_type}"
        )

    def _data_loading_thread(self, data_loader: DataLoader, progress_bar: tqdm) -> None:
        """Data loading thread that prepares batches and puts them in the queue.

        Args:
            data_loader: Data loader instance
            progress_bar: Progress bar for tracking
        """
        logger.info("Starting data loading thread")

        try:
            for batch in data_loader:
                if self.stop_event.is_set():
                    logger.info("Data loading thread stopped by stop event")
                    break

                # Prepare batch for GPU processing
                prepared_batch = self._prepare_batch(batch)

                # Put batch in queue with timeout
                while not self.stop_event.is_set():
                    if self.data_queue.put_batch(prepared_batch, timeout=1):
                        progress_bar.update(1)
                        break
                    else:
                        # Queue is full, wait a bit
                        time.sleep(0.5)

                if self.stop_event.is_set():
                    break

        except Exception as e:
            logger.error(f"Error in data loading thread: {e}")
            logger.debug(traceback.format_exc())
            self.stop_event.set()
        finally:
            logger.info("Data loading thread finished")

    def _detection_thread(
        self, total_batches: int, progress_bar: tqdm
    ) -> List[Dict[str, Any]]:
        """Detection thread that processes batches from the queue.

        Args:
            total_batches: Total number of batches to process
            progress_bar: Progress bar for tracking

        Returns:
            List of processed batches with detections
        """
        logger.info("Starting detection thread")
        processed_batches = []

        try:
            while (
                len(processed_batches) < total_batches and not self.stop_event.is_set()
            ):
                if self.data_queue.is_empty() and not self.stop_event.is_set():
                    time.sleep(0.5)  # Wait for data to be available

                # Get batch from queue
                batch = self.data_queue.get_batch(timeout=1.0)

                if batch is None:
                    # No batch available, check if we should continue
                    if self.data_queue.is_empty() and self.stop_event.is_set():
                        break
                    continue

                # Process batch
                try:
                    detections = self._process_batch(batch)
                    batch["detections"] = detections
                    processed_batches.append(batch)
                    progress_bar.update(1)

                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    self.error_count += 1

                    if self.error_count > 5:
                        logger.error("Too many errors. Stopping detection thread.")
                        self.stop_event.set()
                        break

        except Exception as e:
            logger.error(f"Error in detection thread: {e}")
            logger.info(traceback.format_exc())
            self.stop_event.set()
        finally:
            logger.info("Detection thread finished")

        return processed_batches

    def _run_detection_thread(self, total_batches: int, progress_bar: tqdm) -> None:
        """Wrapper to run detection thread and capture result."""
        self.detection_result = self._detection_thread(total_batches, progress_bar)

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a batch for GPU processing.

        Args:
            batch: Raw batch from data loader

        Returns:
            Prepared batch ready for GPU processing
        """
        # Ensure images are tensors and on the correct device
        if "images" in batch and isinstance(batch["images"], torch.Tensor):
            batch["images"] = batch["images"].to(self.config.device, non_blocking=True)

        return batch

    def run_detection(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> List[DroneImage]:
        """Run detection on images using multi-threaded pipeline.

        Args:
            image_paths: List of image paths
            image_dir: Directory containing images
            save_path: Optional path to save results

        Returns:
            List of processed drone images with detections
        """
        logger.info("Starting multi-threaded detection pipeline")

        # Update config from metadata if available
        self.override_loading_config()

        logger.info("Creating dataloader")
        data_loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
            use_tile_dataset=True,
        )

        total_batches = len(data_loader)
        logger.info(f"Total batches to process: {total_batches}")

        if total_batches == 0:
            logger.warning("No batches to process")
            return []

        # Reset stop event
        self.stop_event.clear()

        # Create progress bars
        data_progress = tqdm(
            total=total_batches, desc="Loading batches", unit="batch", position=0
        )
        detection_progress = tqdm(
            total=total_batches, desc="Processing batches", unit="batch", position=1
        )

        try:
            # Start data loading thread
            self.data_thread = threading.Thread(
                target=self._data_loading_thread,
                args=(data_loader, data_progress),
                daemon=True,
            )
            self.data_thread.start()

            # Start detection thread
            time.sleep(1.0)
            self.detection_thread = threading.Thread(
                target=self._run_detection_thread,
                args=(total_batches, detection_progress),
                daemon=True,
            )
            self.detection_thread.start()

            # Wait for both threads to complete
            self.data_thread.join()
            self.detection_thread.join()

            # Get results from detection thread
            all_batches = self.detection_result

        except Exception as e:
            logger.error(f"Error in multi-threaded pipeline: {e}")
            self.stop_event.set()
            raise
        finally:
            # Clean up progress bars
            data_progress.close()
            detection_progress.close()

            # Ensure threads are stopped
            self.stop_event.set()

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

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.

        Returns:
            Dictionary with pipeline information
        """
        info = super().get_pipeline_info()
        info.update(
            {
                "queue_stats": {
                    "data_queue": self.data_queue.get_stats(),
                    "result_queue": self.result_queue.get_stats(),
                },
            }
        )

        return info

    def stop(self) -> None:
        """Stop the multi-threaded pipeline."""
        logger.info("Stopping multi-threaded pipeline")
        self.stop_event.set()

        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5.0)

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
