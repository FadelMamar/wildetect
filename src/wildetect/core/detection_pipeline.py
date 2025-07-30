"""
Detection Pipeline for end-to-end wildlife detection processing.
"""

import logging
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from ..utils.utils import load_registered_model
from .config import LoaderConfig, PredictionConfig
from .data.detection import Detection
from .data.drone_image import DroneImage
from .data.loader import DataLoader
from .detectors.object_detection_system import ObjectDetectionSystem
from .factory import build_detector
from .processor.processor import Classifier, RoIPostProcessor

logger = logging.getLogger(__name__)


class BatchQueue:
    """Thread-safe queue for batch data transfer between threads."""

    def __init__(self, maxsize: int = 24):
        """Initialize the batch queue.

        Args:
            maxsize: Maximum number of batches in the queue
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.stats = {"put_count": 0, "get_count": 0, "errors": 0}
        self._lock = threading.Lock()

    def put_batch(self, batch: Dict[str, Any], timeout: float = 1.0) -> bool:
        """Put a prepared batch into the queue.

        Args:
            batch: Batch data to put in queue
            timeout: Timeout for put operation

        Returns:
            True if batch was put successfully, False otherwise
        """
        try:
            self.queue.put(batch, timeout=timeout)
            with self._lock:
                self.stats["put_count"] += 1
            return True
        except queue.Full:
            with self._lock:
                self.stats["errors"] += 1
            logger.debug("Queue full, batch not added")
            return False
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            logger.error(f"Error putting batch in queue: {e}")
            return False

    def get_batch(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a batch from the queue.

        Args:
            timeout: Timeout for get operation

        Returns:
            Batch data or None if timeout/empty
        """
        try:
            batch = self.queue.get(timeout=timeout)
            with self._lock:
                self.stats["get_count"] += 1
            return batch
        except queue.Empty:
            return None
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            logger.error(f"Error getting batch from queue: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            stats = self.stats.copy()
            stats["queue_size"] = self.queue.qsize()
            stats["queue_maxsize"] = self.queue.maxsize
        return stats

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty
        """
        return self.queue.empty()

    def is_full(self) -> bool:
        """Check if queue is full.

        Returns:
            True if queue is full
        """
        return self.queue.full()


class DetectionPipeline:
    """End-to-end detection pipeline combining dataloader with inference engine."""

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        """Initialize the detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
            device: Device to run inference on
        """
        self.config = config
        self.loader_config = loader_config
        self.device = config.device
        self.metadata = dict()

        # assert config.model_path is not None, "Model path must be provided"
        # assert config.model_type is not None, "Model type must be provided"

        # Initialize components
        self.detection_system: Optional[ObjectDetectionSystem] = None
        self.data_loader: Optional[DataLoader] = None

        self.setup()

        logger.info(
            f"Initialized DetectionPipeline with model_type={config.model_type}"
        )

        self.error_count = 0

    def load_registered_model(self, mlflow_model_name, mlflow_model_alias):
        """Load the models from MLflow."""

        if mlflow_model_name is None or mlflow_model_alias is None:
            logger.warning("MLFLOW_MODEL_NAME and MLFLOW_MODEL_ALIAS are not set")
            return None, dict()

        try:
            detector_model, metadata = load_registered_model(
                name=mlflow_model_name,
                alias=mlflow_model_alias,
                load_unwrapped=True,
            )
            logger.info(
                f"Loaded model from MLflow: {mlflow_model_name}/{mlflow_model_alias}"
            )
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            logger.debug(traceback.format_exc())
            return None, dict()

        return detector_model, metadata

    def setup(self) -> None:
        """Set up the inference engine with model and processors."""

        mlflow_model_name = os.environ.get("MLFLOW_DETECTOR_NAME", None)
        mlflow_model_alias = os.environ.get("MLFLOW_DETECTOR_ALIAS", None)
        mlflow_roi_name = os.environ.get("MLFLOW_ROI_NAME", None)
        mlflow_roi_alias = os.environ.get("MLFLOW_ROI_ALIAS", None)

        detector_model, self.metadata = self.load_registered_model(
            mlflow_model_name, mlflow_model_alias
        )
        roi_model, roi_metadata = self.load_registered_model(
            mlflow_roi_name, mlflow_roi_alias
        )

        classifier = None
        if roi_model is not None:
            classifier = Classifier(
                model=roi_model,
                model_path=None,
                label_map=self.config.cls_label_map,
                device=self.config.device,
                feature_extractor_path=self.config.feature_extractor_path,
            )
            box_size = roi_metadata.get("box_size", self.config.cls_imgsz)
            cls_imgsz_value = roi_metadata.get("cls_imgsz", self.config.cls_imgsz)
            self.config.cls_imgsz = int(cls_imgsz_value)
            logger.info(f"ROI box size: {box_size} -> {self.config.cls_imgsz}")

        try:
            # Build detector
            if detector_model is not None:
                self.config.model_path = detector_model.ckpt_path

            detector = build_detector(
                config=self.config,
            )

            # Create object detection system
            self.detection_system = ObjectDetectionSystem(
                config=self.config,
            )
            self.detection_system.set_model(detector)

            if self.config.roi_weights or classifier:
                roi_processor = RoIPostProcessor(
                    model_path=self.config.roi_weights,
                    label_map=self.config.cls_label_map,
                    feature_extractor_path=self.config.feature_extractor_path,
                    roi_size=self.config.cls_imgsz,
                    transform=self.config.transform,
                    device=self.config.device,
                    classifier=classifier,
                    keep_classes=self.config.keep_classes,
                )
                self.detection_system.set_processor(roi_processor)

            logger.info("Detection pipeline setup completed")

        except Exception as e:
            logger.error(f"Failed to setup inference engine: {traceback.format_exc()}")
            raise

    def _process_batch(
        self,
        batch: Dict[str, Any],
        progress_bar: Optional[tqdm] = None,
    ) -> List[List[Detection]]:
        """Process a single batch of tiles."""
        # Run inference on tiles
        if self.detection_system is None:
            raise ValueError("Detection system not initialized")
        detections = self.detection_system.predict(batch["images"])
        if progress_bar:
            progress_bar.update(len(batch["tiles"]))
        return detections

    def _postprocess(
        self, batches: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[DroneImage]:
        """Post-process batch results and convert to DroneImage objects.

        Args:
            batches: Single batch or list of batches containing tiles and detections

        Returns:
            List of DroneImage objects with detections
        """
        # Handle both single batch and list of batches
        if isinstance(batches, dict):
            batches = [batches]

        if len(batches) == 0:
            return []

        # Group tiles by parent image
        drone_images: Dict[str, DroneImage] = {}

        # Process each tile and its detections
        for batch in tqdm(batches, desc="Postprocessing batches"):
            # Handle missing detections key
            detections = batch.get("detections", [])
            for tile, tile_detections in zip(batch["tiles"], detections):
                parent_image = tile.parent_image or tile.image_path

                # Create or get drone image for this parent
                if parent_image not in drone_images:
                    drone_image = DroneImage.from_image_path(
                        image_path=parent_image,
                        flight_specs=self.loader_config.flight_specs,
                    )
                    drone_images[parent_image] = drone_image

                # Set detections on the tile
                if tile_detections:
                    tile.set_predictions(tile_detections, update_gps=False)
                else:
                    # Set empty detection if no detections found
                    tile.set_predictions([], update_gps=False)

                # Add tile to drone image with its offset
                offset = (tile.x_offset or 0, tile.y_offset or 0)
                drone_images[parent_image].add_tile(tile, offset[0], offset[1])

        # Update detections
        for drone_image in drone_images.values():
            drone_image.update_detection_gps("predictions")

        return list(drone_images.values())

    def run_detection(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> List[DroneImage]:
        """Run detection on images.

        Args:
            image_paths: List of image paths
            image_dir: Directory containing images
            save_path: Optional path to save results

        Returns:
            List of processed drone images with detections
        """
        logger.info("Starting detection pipeline")

        if "batch" in self.metadata:
            b = self.loader_config.batch_size
            logger.info(f"Batch size: {b} -> {self.metadata.get('batch', b)}")
            b = self.metadata.get("batch", b)
            self.loader_config.batch_size = int(b)

        if "tilesize" in self.metadata:
            tile_size = self.loader_config.tile_size
            logger.info(
                f"Tile size: {tile_size} -> {self.metadata.get('tilesize', tile_size)}"
            )
            tile_size = self.metadata.get("tilesize", tile_size)
            self.loader_config.tile_size = int(tile_size)

        logger.info(f"Creating dataloader")

        data_loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
        )

        # Process batches
        all_batches = []
        total_batches = len(data_loader)
        logger.info(f"Total batches to process: {total_batches}")

        if total_batches == 0:
            logger.warning("No batches to process")
            return []

        def process_one_batch(batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                detections = self._process_batch(batch)
                batch["detections"] = detections
                return batch
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                return None

        # Simple and reliable approach with tqdm

        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
            # with ThreadPoolExecutor(max_workers=3) as executor:
            # Filter out None results and count errors
            # futures = [executor.submit(process_one_batch, batch) for batch in data_loader]

            for result in map(process_one_batch, data_loader):
                if result is not None:
                    all_batches.append(result)
                else:
                    self.error_count += 1

                if self.error_count > 5:
                    raise RuntimeError("Too many errors. Stopping.")

                pbar.update(1)

        logger.info(
            f"Completed processing {len(all_batches)} batches with {self.error_count} errors"
        )

        # postprocessing
        all_drone_images = self._postprocess(batches=all_batches)
        if len(all_drone_images) == 0:
            logger.warning("No batches were processed")
            return []

        # Save results if path provided
        if save_path:
            self._save_results(all_drone_images, save_path)

        return all_drone_images

    def _save_results(
        self,
        drone_images: List[DroneImage],
        save_path: str,
    ) -> None:
        """Save detection results.

        Args:
            drone_images: List of drone images with detections
            save_path: Path to save results
        """
        import json

        results = []
        for drone_image in drone_images:
            stats = drone_image.get_statistics()
            results.append(stats)

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path_obj, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {save_path}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.

        Returns:
            Dictionary with pipeline information
        """
        info = {
            "model_type": self.config.model_type,
            "model_path": self.config.model_path,
            "device": self.device,
            "has_detection_system": self.detection_system is not None,
            "has_data_loader": self.data_loader is not None,
        }

        if self.detection_system:
            info["detection_system"] = self.detection_system.get_model_info()

        if self.data_loader:
            info["data_loader"] = self.data_loader.get_statistics()

        return info


class MultiThreadedDetectionPipeline:
    """Multi-threaded detection pipeline with separate data loading and detection threads."""

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
        queue_size: int = 3,
    ):
        """Initialize the multi-threaded detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
            queue_size: Maximum number of batches in the queue
        """
        self.config = config
        self.loader_config = loader_config
        self.device = config.device
        self.metadata = dict()
        self.error_count = 0

        # Thread-safe queues
        self.data_queue = BatchQueue(maxsize=queue_size)
        self.result_queue = BatchQueue(maxsize=queue_size * 2)

        # Thread control
        self.stop_event = threading.Event()
        self.data_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_result: List[Dict[str, Any]] = []

        # Initialize components
        self.detection_system: Optional[ObjectDetectionSystem] = None
        self.data_loader: Optional[DataLoader] = None

        self.setup()

        logger.info(
            f"Initialized MultiThreadedDetectionPipeline with model_type={config.model_type}"
        )

    def load_registered_model(self, mlflow_model_name, mlflow_model_alias):
        """Load the models from MLflow."""
        if mlflow_model_name is None or mlflow_model_alias is None:
            logger.warning("MLFLOW_MODEL_NAME and MLFLOW_MODEL_ALIAS are not set")
            return None, dict()

        try:
            detector_model, metadata = load_registered_model(
                name=mlflow_model_name,
                alias=mlflow_model_alias,
                load_unwrapped=True,
            )
            logger.info(
                f"Loaded model from MLflow: {mlflow_model_name}/{mlflow_model_alias}"
            )
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            logger.debug(traceback.format_exc())
            return None, dict()

        return detector_model, metadata

    def setup(self) -> None:
        """Set up the inference engine with model and processors."""
        mlflow_model_name = os.environ.get("MLFLOW_DETECTOR_NAME", None)
        mlflow_model_alias = os.environ.get("MLFLOW_DETECTOR_ALIAS", None)
        mlflow_roi_name = os.environ.get("MLFLOW_ROI_NAME", None)
        mlflow_roi_alias = os.environ.get("MLFLOW_ROI_ALIAS", None)

        detector_model, self.metadata = self.load_registered_model(
            mlflow_model_name, mlflow_model_alias
        )
        roi_model, roi_metadata = self.load_registered_model(
            mlflow_roi_name, mlflow_roi_alias
        )

        classifier = None
        if roi_model is not None:
            classifier = Classifier(
                model=roi_model,
                model_path=None,
                label_map=self.config.cls_label_map,
                device=self.config.device,
                feature_extractor_path=self.config.feature_extractor_path,
            )
            box_size = roi_metadata.get("box_size", self.config.cls_imgsz)
            cls_imgsz_value = roi_metadata.get("cls_imgsz", self.config.cls_imgsz)
            self.config.cls_imgsz = int(cls_imgsz_value)
            logger.info(f"ROI box size: {box_size} -> {self.config.cls_imgsz}")

        try:
            # Build detector
            if detector_model is not None:
                self.config.model_path = detector_model.ckpt_path

            detector = build_detector(config=self.config)

            # Create object detection system
            self.detection_system = ObjectDetectionSystem(config=self.config)
            self.detection_system.set_model(detector)

            if self.config.roi_weights or classifier:
                roi_processor = RoIPostProcessor(
                    model_path=self.config.roi_weights,
                    label_map=self.config.cls_label_map,
                    feature_extractor_path=self.config.feature_extractor_path,
                    roi_size=self.config.cls_imgsz,
                    transform=self.config.transform,
                    device=self.config.device,
                    classifier=classifier,
                    keep_classes=self.config.keep_classes,
                )
                self.detection_system.set_processor(roi_processor)

            logger.info("Multi-threaded detection pipeline setup completed")

        except Exception as e:
            logger.error(f"Failed to setup inference engine: {traceback.format_exc()}")
            raise

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
                    if self.data_queue.put_batch(prepared_batch, timeout=0.5):
                        progress_bar.update(1)
                        break
                    else:
                        # Queue is full, wait a bit
                        time.sleep(0.1)

                if self.stop_event.is_set():
                    break

        except Exception as e:
            logger.error(f"Error in data loading thread: {e}")
            logger.debug(traceback.format_exc())
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
                        break

        except Exception as e:
            logger.error(f"Error in detection thread: {e}")
            logger.debug(traceback.format_exc())
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
            batch["images"] = batch["images"].to(self.device)

        return batch

    def _process_batch(self, batch: Dict[str, Any]) -> List[List[Detection]]:
        """Process a single batch of tiles.

        Args:
            batch: Batch containing images and tiles

        Returns:
            List of detection lists for each image in the batch
        """
        if self.detection_system is None:
            raise ValueError("Detection system not initialized")

        detections = self.detection_system.predict(batch["images"])
        return detections

    def _postprocess(self, batches: List[Dict[str, Any]]) -> List[DroneImage]:
        """Post-process batch results and convert to DroneImage objects.

        Args:
            batches: List of batches containing tiles and detections

        Returns:
            List of DroneImage objects with detections
        """
        if len(batches) == 0:
            return []

        # Group tiles by parent image
        drone_images: Dict[str, DroneImage] = {}

        # Process each tile and its detections
        for batch in tqdm(batches, desc="Postprocessing batches"):
            detections = batch.get("detections", [])
            for tile, tile_detections in zip(batch["tiles"], detections):
                parent_image = tile.get("parent_image", tile.get("image_path", None))
                if parent_image is None:
                    logger.warning(f"Parent image is not set for tile: {tile}")
                    continue

                # Create or get drone image for this parent
                if parent_image not in drone_images:
                    drone_image = DroneImage.from_image_path(
                        image_path=parent_image,
                        flight_specs=self.loader_config.flight_specs,
                    )
                    drone_images[parent_image] = drone_image

                # Set detections on the tile
                if tile_detections:
                    tile.set_predictions(tile_detections, update_gps=False)
                else:
                    # Set empty detection if no detections found
                    tile.set_predictions([], update_gps=False)

                # Add tile to drone image with its offset
                offset = (tile.x_offset or 0, tile.y_offset or 0)
                drone_images[parent_image].add_tile(tile, offset[0], offset[1])

        # Update detections
        for drone_image in drone_images.values():
            drone_image.update_detection_gps("predictions")

        return list(drone_images.values())

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
        if "batch" in self.metadata:
            b = self.loader_config.batch_size
            logger.info(f"Batch size: {b} -> {self.metadata.get('batch', b)}")
            b = self.metadata.get("batch", b)
            self.loader_config.batch_size = int(b)

        if "tilesize" in self.metadata:
            tile_size = self.loader_config.tile_size
            logger.info(
                f"Tile size: {tile_size} -> {self.metadata.get('tilesize', tile_size)}"
            )
            tile_size = self.metadata.get("tilesize", tile_size)
            self.loader_config.tile_size = int(tile_size)

        logger.info("Creating dataloader")
        data_loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
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

    def _save_results(
        self,
        drone_images: List[DroneImage],
        save_path: str,
    ) -> None:
        """Save detection results.

        Args:
            drone_images: List of drone images with detections
            save_path: Path to save results
        """
        import json

        results = []
        for drone_image in drone_images:
            stats = drone_image.get_statistics()
            results.append(stats)

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path_obj, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {save_path}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.

        Returns:
            Dictionary with pipeline information
        """
        info = {
            "model_type": self.config.model_type,
            "model_path": self.config.model_path,
            "device": self.device,
            "has_detection_system": self.detection_system is not None,
            "has_data_loader": self.data_loader is not None,
            "queue_stats": {
                "data_queue": self.data_queue.get_stats(),
                "result_queue": self.result_queue.get_stats(),
            },
        }

        if self.detection_system:
            info["detection_system"] = self.detection_system.get_model_info()

        return info

    def stop(self) -> None:
        """Stop the multi-threaded pipeline."""
        logger.info("Stopping multi-threaded pipeline")
        self.stop_event.set()

        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5.0)

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
