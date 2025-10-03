import logging
import queue
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import torch
from sympy import E
from tqdm import tqdm

from ..config import LoaderConfig, PredictionConfig
from ..data import Detection, DroneImage
from ..data.loader import DataLoader
from .base import DetectionPipeline

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
        self.data_queue = queue.Queue(maxsize=config.queue_size)
        self.result_queue = queue.Queue(maxsize=0)  # unlimited size

        # Thread control
        self.stop_event = threading.Event()
        self.data_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_result: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized MultiThreadedDetectionPipeline with model_type={config.model_type}"
        )

    def _data_loading_worker(
        self,
        data_loader: DataLoader,
        progress_bar: tqdm,
    ) -> None:
        """Standalone data loading worker function that can be pickled.

        Args:
            data_loader: Data loader instance
            progress_bar: tqdm progress bar
        """
        logger.info("Starting data loading process")

        try:
            for batch in data_loader:
                if self.stop_event.is_set():
                    logger.info("Data loading process stopped by stop event")
                    break

                while not self.stop_event.is_set():
                    try:
                        self.data_queue.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        # Queue is full, wait a bit
                        time.sleep(0.5)

                progress_bar.update(1)

                if self.stop_event.is_set():
                    break

        except Exception as e:
            logger.error(f"Error in data loading process: {e}")
            logger.debug(traceback.format_exc())
            self.stop_event.set()
        finally:
            logger.info("Data loading process finished")

    def _detection_worker(
        self,
        progress_bar: tqdm,
    ) -> None:
        """Standalone detection worker function that can be pickled.

        Args:
            detection_system: Detection system instance
            data_queue: Queue to get batches from
            result_queue: Queue to put results into
            stop_event: Event to signal stopping
            config: Prediction configuration
        """

        # Initialize worker with its own model
        error_count = 0

        # Wait for data loading thread to start
        time.sleep(1.0)

        try:
            while not self.stop_event.is_set():
                try:
                    batch = self.data_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    # No batch available, wait a bit
                    time.sleep(1.0)
                    continue
                except Exception as e:
                    raise e

                # Process batch
                try:
                    batch_tensor = (
                        batch if isinstance(batch, torch.Tensor) else batch["images"]
                    )
                    batch_tensor = batch_tensor.to(
                        self.config.device, non_blocking=True
                    )
                    detections = self._process_batch(batch, progress_bar=progress_bar)
                    self.result_queue.put(detections)
                except Exception as e:
                    error_count += 1
                    if error_count > 5:
                        raise Exception("Too many errors. Stopping detection worker.")

        except Exception as e:
            logger.error(f"Error in detection worker: {e}")
            logger.debug(traceback.format_exc())
            logger.info("Stopping detection worker")
            self.stop_event.set()

        finally:
            logger.info("Detection worker finished")
            self.stop_event.set()

    def run_detection(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        override_loading_config: bool = True,
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
        if override_loading_config:
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
                target=self._data_loading_worker,
                args=(data_loader, data_progress),
                daemon=True,
            )
            self.data_thread.start()

            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self._detection_worker,
                args=(detection_progress),
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
