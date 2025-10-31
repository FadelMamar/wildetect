import logging
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopy
import rasterio as rio
import torch
from rasterio.warp import transform
from tqdm import tqdm

from ..config import LoaderConfig, PredictionConfig
from ..data import DataLoader, Detection, DroneImage, Tile
from ..data.utils import get_images_paths
from .base import BaseDetectionPipeline

logger = logging.getLogger(__name__)


class RasterDetectionPipeline(BaseDetectionPipeline):
    """Detection pipeline for processing large raster files.

    This pipeline works with DataLoader when raster_path is provided.
    It processes the raster in overlapping windows and returns detections
    with their spatial coordinates in the raster coordinate system.
    """

    def __init__(self, config: PredictionConfig, loader_config: LoaderConfig):
        super().__init__(config, loader_config)
        self.drone_image: Optional[DroneImage] = None
        self.src: Optional[rio.DatasetReader] = None
        self.raster_path: Optional[str] = None

    def set_drone_image(self, image_paths: List[str]):
        assert (
            len(image_paths) == 1
        ), f"Only one image path is supported for raster detection. Received {len(image_paths)} image paths."
        self.raster_path = image_paths[0]
        self.src = rio.open(self.raster_path)

        longitude, latitude = self.get_gps_coords(
            row=int(self.src.height / 2),
            col=int(self.src.width / 2),
            altitude=self.loader_config.flight_specs.flight_height,
            as_decimal=True,
        )

        self.drone_image = DroneImage.from_image_path(
            image_path=self.raster_path,
            flight_specs=self.loader_config.flight_specs,
            width=self.src.width,
            height=self.src.height,
            latitude=latitude,
            longitude=longitude,
            gsd=self.config.flight_specs.gsd,
            is_raster=True,
        )

    def close_reader(self):
        try:
            if self.src is not None:
                self.src.close()
                self.src = None
        except Exception:
            logger.error(traceback.format_exc())
        return None

    def get_gps_coords(
        self, row: int, col: int, altitude: float, as_decimal: bool = False
    ) -> Union[Tuple[float, float], str]:
        longitude, latitude = self.src.xy(row, col)
        longitude, latitude = transform(
            src_crs=self.src.crs, dst_crs="EPSG:4326", xs=[longitude], ys=[latitude]
        )
        if as_decimal:
            return longitude[0], latitude[0]
        else:
            return str(
                geopy.Point(
                    latitude=latitude[0],
                    longitude=longitude[0],
                    altitude=altitude / 1e3,
                )
            )

    def run_detection(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        override_loading_config: bool = True,
    ) -> List[DroneImage]:
        """Run detection on a raster file.

        Args:
            raster_path: Path to the raster file
            save_path: Optional path to save results
            override_loading_config: Whether to override loader config

        Returns:
            Dictionary with raster path, detections with spatial bounds
        """
        assert (image_paths is None) ^ (
            image_dir is None
        ), "One of image_paths and image_dir must be None"
        logger.info("Starting raster detection pipeline")
        if image_dir is not None:
            image_paths = get_images_paths(image_dir)

        # set drone image
        self.set_drone_image(image_paths)

        # Update config from metadata if available
        if override_loading_config:
            self.override_loading_config()

        self.save_path = save_path
        if self.save_path:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating raster dataloader for: {image_paths}")

        # Create data loader with raster_path
        data_loader = DataLoader(
            raster_path=self.raster_path,
            config=self.loader_config,
        )

        total_batches = len(data_loader)
        logger.info(f"Total batches to process: {total_batches}")

        if total_batches == 0:
            logger.warning("No batches to process")
            return []

        # Process all batches
        for batch_data, batch_bounds, gps_coords in tqdm(
            data_loader, desc="Processing raster patches", unit="batch"
        ):
            try:
                # Move batch to device
                batch_tensor = batch_data.to(self.config.device, non_blocking=True)
                # Run detection
                detections = self._process_batch(batch_tensor)
                # add detections to drone image
                self._add_batch_detections_to_drone_image(
                    detections, batch_bounds, gps_coords
                )

            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to process batch: {e}")
                logger.debug(traceback.format_exc())

                if self.error_count > self.config.max_errors:
                    raise ValueError(
                        f"Too many errors. Stopping. {self.error_count} > {self.config.max_errors}"
                    )

        logger.info(
            f"Completed processing {total_batches} batches "
            f"with {self.error_count} errors"
        )
        logger.info(f"Found {len(self.drone_image.tiles)} patches with detections")

        # Save results if path provided
        if save_path:
            self._save_results(self.get_drone_images(), save_path)

        self.close_reader()

        return self.get_drone_images()

    def _add_batch_detections_to_drone_image(
        self,
        detections: List[List[Detection]],
        batch_bounds: torch.LongTensor,
        gps_coords: torch.Tensor,
    ) -> None:
        for detection, bound, (lon, lat) in zip(
            detections, batch_bounds.cpu().tolist(), gps_coords.cpu().tolist()
        ):
            tile = Tile(
                x_offset=bound[0],
                y_offset=bound[1],
                width=bound[2],
                height=bound[3],
                longitude=lon,
                latitude=lat,
                gsd=self.config.flight_specs.gsd,
                is_raster=True,
            )

            if len(detection) > 0:
                for det in detection:
                    det.gps_loc = self.get_gps_coords(
                        row=det.y_center + tile.y_offset,
                        col=det.x_center + tile.x_offset,
                        altitude=self.drone_image.altitude,
                        as_decimal=False,
                    )
                tile.set_predictions(detection, update_gps=False)
            else:
                tile.set_predictions([], update_gps=False)

            self.drone_image.add_tile(
                tile=tile,
                x_offset=tile.x_offset,
                y_offset=tile.y_offset,
                offset_detections=True,
                nms_threshold=self.config.nms_threshold,
            )

    def get_drone_images(self) -> List[DroneImage]:
        return [self.drone_image]


class MultiThreadedRasterDetectionPipeline(RasterDetectionPipeline):
    """Multi-threaded detection pipeline for processing large raster files.

    This pipeline uses separate threads for data loading and detection processing,
    allowing for better GPU utilization and faster processing times.
    """

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        """Initialize the multi-threaded raster detection pipeline.

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

        logger.info(f"Initialized MultiThreadedRasterDetectionPipeline")

    def _data_loading_worker(
        self,
        data_loader: DataLoader,
        progress_bar: tqdm,
    ) -> None:
        """Standalone data loading worker function for raster processing.

        Args:
            data_loader: Data loader instance
            progress_bar: tqdm progress bar
        """
        logger.info("Starting data loading process")

        try:
            for batch_data, batch_bounds, gps_coords in data_loader:
                if self.stop_event.is_set():
                    logger.info("Data loading process stopped by stop event")
                    break

                # Package the batch data
                batch = {
                    "batch_data": batch_data,
                    "batch_bounds": batch_bounds,
                    "gps_coords": gps_coords,
                }

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
            self.data_queue.put(None)
            logger.info("Data loading process finished")

    def _detection_worker(
        self,
        progress_bar: tqdm,
    ) -> None:
        """Standalone detection worker function for raster processing.

        Args:
            progress_bar: tqdm progress bar
        """

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
                except KeyboardInterrupt:
                    logger.info("Detection process stopped by keyboard interrupt")
                    self.stop_event.set()
                    break
                except Exception as e:
                    raise e

                # Check for sentinel value (end of data signal)
                if batch is None:
                    logger.info("Received end of data signal")
                    break

                # Process batch
                try:
                    batch_data = batch["batch_data"]
                    batch_bounds = batch["batch_bounds"]
                    gps_coords = batch["gps_coords"]

                    # Move batch to device
                    batch_tensor = batch_data.to(self.config.device, non_blocking=True)

                    # Run detection
                    detections = self._process_batch(
                        batch_tensor, progress_bar=progress_bar
                    )

                    # Add detections to drone image
                    self._add_batch_detections_to_drone_image(
                        detections, batch_bounds, gps_coords
                    )

                except Exception as e:
                    self.error_count += 1

                if self.error_count > self.config.max_errors:
                    raise ValueError(
                        f"Too many errors. Stopping detection worker. {self.error_count} > {self.config.max_errors}"
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
        """Run detection on a raster file using multi-threaded pipeline.

        Args:
            image_paths: List containing a single raster path
            image_dir: Directory containing a single raster file
            save_path: Optional path to save results
            override_loading_config: Whether to override loader config

        Returns:
            List containing a single DroneImage with detections
        """
        assert (image_paths is None) ^ (
            image_dir is None
        ), "One of image_paths and image_dir must be None"
        logger.info("Starting multi-threaded raster detection pipeline")

        if image_dir is not None:
            image_paths = get_images_paths(image_dir)

        self.set_drone_image(image_paths)

        # Update config from metadata if available
        if override_loading_config:
            self.override_loading_config()

        self.save_path = save_path
        if self.save_path:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating raster dataloader for: {image_paths}")

        # Create data loader with raster_path
        data_loader = DataLoader(
            raster_path=image_paths[0],
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
            total=total_batches, desc="Loading raster patches", position=0
        )
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
            f"Completed processing {total_batches} batches "
            f"with {self.error_count} errors"
        )
        logger.info(f"Found {len(self.drone_image.tiles)} patches with detections")

        # Save results if path provided
        if save_path:
            self._save_results(self.get_drone_images(), save_path)

        self.close_reader()

        return self.get_drone_images()
