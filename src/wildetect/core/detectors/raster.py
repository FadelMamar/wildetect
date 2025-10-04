import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import rasterio as rio
import torch
from tqdm import tqdm

from ..config import LoaderConfig, PredictionConfig
from ..data import DataLoader, Detection, DroneImage, Tile
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
        self.raster_path: Optional[str] = None
        self.drone_image: Optional[DroneImage] = None

    def run_detection(
        self,
        raster_path: str,
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
        logger.info("Starting raster detection pipeline")
        with rio.open(raster_path) as src:
            self.drone_image = DroneImage.from_image_path(
                image_path=raster_path,
                flight_specs=self.loader_config.flight_specs,
                width=src.width,
                height=src.height,
            )

        # Update config from metadata if available
        if override_loading_config:
            self.override_loading_config()

        self.raster_path = raster_path
        self.save_path = save_path
        if self.save_path:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating raster dataloader for: {raster_path}")

        # Create data loader with raster_path
        data_loader = DataLoader(
            raster_path=raster_path,
            config=self.loader_config,
        )

        total_batches = len(data_loader)
        logger.info(f"Total batches to process: {total_batches}")

        if total_batches == 0:
            logger.warning("No batches to process")
            return []

        # Process all batches
        for batch_data, batch_bounds in tqdm(
            data_loader, desc="Processing raster patches", unit="batch"
        ):
            try:
                # Move batch to device
                batch_tensor = batch_data.to(self.config.device, non_blocking=True)
                # Run detection
                detections = self._process_batch(batch_tensor)
                # add detections to drone image
                self._add_batch_detections_to_drone_image(detections, batch_bounds)

            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to process batch: {e}")
                logger.debug(traceback.format_exc())

                if self.error_count > 5:
                    raise RuntimeError(
                        f"Too many errors. Stopping. {traceback.format_exc()}"
                    )

        # update gps of detections
        self.drone_image.update_detection_gps("predictions")

        logger.info(
            f"Completed processing {total_batches} batches "
            f"with {self.error_count} errors"
        )
        logger.info(f"Found {len(self.drone_image.tiles)} patches with detections")

        # Save results if path provided
        if save_path:
            self._save_results([self.drone_image], save_path)

        return [self.drone_image]

    def _add_batch_detections_to_drone_image(
        self, detections: List[List[Detection]], batch_bounds: torch.LongTensor
    ) -> None:
        for detection, bound in zip(detections, batch_bounds.cpu().tolist()):
            tile = Tile(
                x_offset=bound[0],
                y_offset=bound[1],
                width=bound[2],
                height=bound[3],
            )
            if detection:
                tile.set_predictions(detection, update_gps=False)
            else:
                tile.set_predictions([], update_gps=False)
            self.drone_image.add_tile(
                tile,
                tile.x_offset,
                tile.y_offset,
            )
