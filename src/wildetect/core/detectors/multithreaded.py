import logging
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ..config import LoaderConfig, PredictionConfig
from ..data import Detection, DroneImage
from ..data.loader import DataLoader
from .base import DetectionPipeline, SimpleDetectionPipeline

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
        """

        super().__init__(config, loader_config)

        # Thread-safe queues
        self.data_queue = queue.Queue(maxsize=config.queue_size)

        # Thread control
        self.stop_event = threading.Event()
        self.data_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_results: List = []

        logger.info(f"Initialized MultiThreadedDetectionPipeline")

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
                        self.data_queue.put(batch, timeout=0.1)
                        progress_bar.update(1)
                        break
                    except KeyboardInterrupt:
                        logger.info(
                            "Data loading process stopped by keyboard interrupt"
                        )
                        self.stop_event.set()
                        break
                    except queue.Full:
                        # Queue is full, wait a bit
                        time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Data loading process stopped by keyboard interrupt")
            self.stop_event.set()

        except Exception as e:
            logger.error(f"Error in data loading process: {e}")
            logger.debug(traceback.format_exc())
            self.stop_event.set()
        finally:
            # Signal end of data by putting None in the queue
            self.data_queue.put("None")
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

        # Wait for data loading thread to start
        time.sleep(1.0)

        try:
            while not self.stop_event.is_set():
                try:
                    batch: Dict = self.data_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    # No batch available, wait a bit
                    time.sleep(1.0)
                    continue
                except KeyboardInterrupt:
                    logger.info("Detection process stopped by keyboard interrupt")
                    self.stop_event.set()
                    break
                except Exception as e:
                    raise e

                # Check for sentinel value (end of data signal)
                if batch == "None":
                    logger.info("Received end of data signal")
                    break

                # Process batch
                try:
                    batch_tensor = batch.pop("images")
                    batch_tensor = batch_tensor.to(
                        self.config.device, non_blocking=True
                    )
                    detections = self._process_batch(
                        batch_tensor, progress_bar=progress_bar
                    )
                    batch["detections"] = detections
                    self.detection_results.append(batch)
                except Exception as e:
                    self.error_count += 1
                    if self.error_count > 5:
                        raise Exception(
                            f"Too many errors. Stopping detection worker. {traceback.format_exc()}"
                        )

        except KeyboardInterrupt:
            logger.info("Detection process stopped by keyboard interrupt")
            self.stop_event.set()

        except Exception as e:
            logger.error(f"{e}")
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

        if self.image_csv_data is not None:
            image_paths = self._get_image_paths(from_csv=True)        
        else:
            assert (image_paths is not None) ^ (image_dir is not None), "image_paths or image_dir must be provided"

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

        # Reset stop event and results
        self.stop_event.clear()
        self.detection_results.clear()

        # Create progress bars
        data_progress = tqdm(total=total_batches, desc="Loading data", position=0)
        detection_progress = tqdm(
            total=total_batches, desc="Computing detections", position=1
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
                args=(detection_progress,),
                daemon=True,
            )
            self.detection_thread.start()

            # Wait for both threads to complete
            self.data_thread.join()
            self.detection_thread.join()

        except Exception as e:
            logger.error(f"Error in multi-threaded pipeline: {e}")

        finally:
            # Clean up progress bars
            data_progress.close()
            detection_progress.close()
            self.stop_event.set()

        logger.info(
            f"Completed processing {len(self.detection_results)} batches with {self.error_count} errors"
        )

        # Post-processing
        all_drone_images = self._postprocess(batches=self.detection_results)
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
        return info


class SimpleMultiThreadedDetectionPipeline(SimpleDetectionPipeline):
    """Multi-threaded detection pipeline that processes images one at a time.

    This pipeline inherits from SimpleDetectionPipeline and uses separate threads
    for data loading and detection to improve throughput.
    """

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        """Initialize the simple multi-threaded detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
        """
        super().__init__(config, loader_config)

        # Thread-safe queues
        self.image_queue = queue.Queue(maxsize=config.queue_size)
        self.result_queue = queue.Queue()

        # Thread control
        self.stop_event = threading.Event()
        self.data_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None

        logger.info("Initialized SimpleMultiThreadedDetectionPipeline")

    def _data_loading_worker(
        self,
        data_loader: DataLoader,
        progress_bar: tqdm,
    ) -> None:
        """Data loading worker that loads one image at a time.

        Args:
            data_loader: Data loader instance
            progress_bar: tqdm progress bar
        """
        logger.info("Starting data loading worker")

        try:
            for img, idx in data_loader:
                if self.stop_event.is_set():
                    logger.info("Data loading worker stopped by stop event")
                    break

                # Get offset information for this image
                offset_info = data_loader.get_offset_info(idx=idx)

                # Put image and metadata in queue
                while not self.stop_event.is_set():
                    try:
                        self.image_queue.put((img, offset_info), timeout=1.0)
                        progress_bar.update(1)
                        break
                    except KeyboardInterrupt:
                        logger.info("Data loading worker stopped by keyboard interrupt")
                        self.stop_event.set()
                        break
                    except queue.Full:
                        # Queue is full, wait a bit
                        time.sleep(0.5)

                if self.stop_event.is_set():
                    break

        except KeyboardInterrupt:
            logger.info("Data loading worker stopped by keyboard interrupt")
            self.stop_event.set()

        except Exception as e:
            logger.error(f"Error in data loading worker: {e}")
            logger.debug(traceback.format_exc())
            self.stop_event.set()
        finally:
            # Signal end of data by putting sentinel value
            self.image_queue.put(None)
            logger.info("Data loading worker finished")

    def _detection_worker(
        self,
        progress_bar: tqdm,
    ) -> None:
        """Detection worker that processes one image at a time.

        Args:
            progress_bar: tqdm progress bar
        """
        logger.info("Starting detection worker")

        # Wait for data loading thread to start
        time.sleep(0.1)

        try:
            while not self.stop_event.is_set():
                try:
                    item = self.image_queue.get(timeout=0.5)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    # No image available, wait a bit
                    time.sleep(1.0)
                    continue
                except KeyboardInterrupt:
                    logger.info("Detection worker stopped by keyboard interrupt")
                    self.stop_event.set()
                    break
                except Exception as e:
                    raise e

                # Check for sentinel value (end of data signal)
                if item is None:
                    logger.info("Received end of data signal")
                    break

                image_as_patches, offset_info = item

                # Process the image patches in batches
                try:
                    detections = self._process_one_image(image_as_patches)

                    # Post-process and create DroneImage
                    drone_image = self._postprocess_one_image(detections, offset_info)

                    # Put result in result queue
                    self.result_queue.put(drone_image)

                    progress_bar.update(1)

                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error processing image: {e}")
                    logger.debug(traceback.format_exc())
                    if self.error_count > 5:
                        raise Exception(
                            f"Too many errors. Stopping detection worker. {traceback.format_exc()}"
                        )

        except KeyboardInterrupt:
            logger.info("Detection worker stopped by keyboard interrupt")

        except Exception as e:
            logger.error(f"{e}")
            logger.debug(traceback.format_exc())
            logger.info("Stopping detection worker")

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
            override_loading_config: Whether to override loader config from model metadata

        Returns:
            List of processed drone images with detections
        """
        logger.info("Starting simple multi-threaded detection pipeline")

        # Update config from metadata if available
        if override_loading_config:
            self.override_loading_config()

        self.save_path = save_path
        if self.save_path:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Creating dataloader")
        data_loader = self.get_data_loader(
            image_paths=image_paths,
            image_dir=image_dir,
            use_tile_dataset=False,  # Important: SimpleDetectionPipeline uses False
        )

        total_images = len(data_loader)
        logger.info(f"Total images to process: {total_images}")

        if total_images == 0:
            logger.warning("No images to process")
            return []

        # Reset state
        self.stop_event.clear()
        all_drone_images = []

        # Create progress bars
        data_progress = tqdm(total=total_images, desc="Loading images", position=0)
        detection_progress = tqdm(
            total=total_images, desc="Computing detections", position=1
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
                args=(detection_progress,),
                daemon=True,
            )
            self.detection_thread.start()

            # Collect results as they become available
            processed_images = 0
            while processed_images < total_images and not self.stop_event.is_set():
                try:
                    drone_image = self.result_queue.get(timeout=1.0)
                    all_drone_images.append(drone_image)
                    processed_images += 1

                    # Save results incrementally if path provided
                    if self.save_path:
                        self._save_results(drone_image, mode="a")

                except queue.Empty:
                    # Check if threads are still alive
                    if (
                        not self.detection_thread.is_alive()
                        and not self.data_thread.is_alive()
                    ):
                        break
                    continue

            # Wait for both threads to complete
            self.stop()

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
            f"Completed processing {len(all_drone_images)} images with "
            f"{self.total_batches} batches and {self.total_tiles} tiles"
        )

        return all_drone_images

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.

        Returns:
            Dictionary with pipeline information
        """
        info = super().get_pipeline_info()
        info["pipeline_type"] = "simple_multithreaded"
        info["queue_size"] = self.config.queue_size
        return info

    def stop(self) -> None:
        """Stop the multi-threaded pipeline."""
        logger.info("Stopping simple multi-threaded pipeline")
        self.stop_event.set()

        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5.0)

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
