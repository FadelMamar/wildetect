import json
import logging
import traceback
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import supervision as sv
import torch
from tqdm import tqdm
from wildtrain.models.detector import Detector
from wildtrain.utils.mlflow import load_registered_model

from ..config import LoaderConfig, PredictionConfig
from ..data import Detection, DroneImage, Tile
from ..data.loader import DataLoader

logger = logging.getLogger(__name__)


class DetectionPipeline(object):
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

        self.data_loader: Optional[DataLoader] = None
        if config.inference_service_url is None:
            self.detection_system = load_registered_model(
                alias=config.mlflow_model_alias,
                name=config.mlflow_model_name,
                load_unwrapped=True,
            )
            self.metadata = self.detection_system.metadata
            logger.info("Loading weights from MLFlow")
            self.detection_system.set_device(config.device)
        else:
            self.detection_system = partial(
                Detector.predict_inference_service,
                url=config.inference_service_url,
                timeout=config.timeout,
            )
            logger.info(f"Using inference service @ {config.inference_service_url}")
            self.metadata = dict()

        self.error_count = 0

    def _convert_to_detection(
        self, detections: List[sv.Detections]
    ) -> List[List[Detection]]:
        """Convert a list of detections to a list of Detection objects."""
        return [Detection.from_supervision(det) for det in detections]

    def _process_batch(
        self,
        batch: torch.Tensor,
        progress_bar: Optional[tqdm] = None,
    ) -> List[List[Detection]]:
        """Process a single batch of tiles."""
        # Run inference on tiles
        if self.detection_system is None:
            raise ValueError("Detection system not initialized")
        if self.config.inference_service_url is None:
            detections = self.detection_system.predict(batch, return_as_dict=False)
        else:
            detections = self.detection_system(batch)

        if progress_bar:
            progress_bar.update(batch.shape[0])

        # Convert to Detection objects
        detections = self._convert_to_detection(detections)

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
            detections = batch.get("detections", [])
            for tile_data, tile_detections in zip(batch["tiles"], detections):
                tile = Tile.from_dict(tile_data)
                assert (
                    tile.parent_image is not None
                ), "Parent image is None. Error in dataloader."

                # Create or get drone image for this parent
                if tile.parent_image not in drone_images:
                    drone_image = DroneImage.from_image_path(
                        image_path=tile.parent_image,
                        flight_specs=self.loader_config.flight_specs,
                    )
                    drone_images[tile.parent_image] = drone_image

                # Set detections on the tile
                if tile_detections:
                    tile.set_predictions(tile_detections, update_gps=False)
                else:
                    tile.set_predictions([], update_gps=False)

                # Add tile to drone image with its offset
                offset = (tile.x_offset or 0, tile.y_offset or 0)
                drone_images[tile.parent_image].add_tile(tile, offset[0], offset[1])

        # Update detections
        for drone_image in drone_images.values():
            drone_image.update_detection_gps("predictions")

        return list(drone_images.values())

    def run_detection(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        override_loading_config: bool = True,
    ) -> List[DroneImage]:
        """Run detection on images.

        Args:
            image_paths: List of image paths
            image_dir: Directory containing images
            save_path: Optional path to save results

        Returns:
            List of processed drone images with detections
        """
        logger.info("Starting single-threaded detection pipeline")

        # Update config from metadata if available
        if (self.metadata is not None) and override_loading_config:
            if "batch" in self.metadata:
                b = self.loader_config.batch_size
                logger.info(
                    f"Updating Loader Batch size: {b} -> {self.metadata.get('batch', b)}"
                )
                b = self.metadata.get("batch", b)
                self.loader_config.batch_size = int(b)

            if "tilesize" in self.metadata:
                tile_size = self.loader_config.tile_size
                logger.info(
                    f"Updating Loader Tile size: {tile_size} -> {self.metadata.get('tilesize', tile_size)}"
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
                detections = self._process_batch(batch.pop("images"))
                batch["detections"] = detections
                return batch
            except Exception:
                logger.error(f"Failed to process batch: {traceback.format_exc()}")
                return None

        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
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
        info: Dict[str, Any] = {
            "model_type": self.config.model_type,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "has_detection_system": self.detection_system is not None,
            "has_data_loader": self.data_loader is not None,
        }

        if self.detection_system:
            info["detection_system"] = self.detection_system.get_model_info()

        if self.data_loader:
            info["data_loader"] = {
                "total_batches": len(self.data_loader) if self.data_loader else 0,
                "config": vars(self.loader_config) if self.loader_config else {},
            }

        return info
