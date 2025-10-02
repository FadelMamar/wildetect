import json
import logging
import traceback
from abc import ABC, abstractmethod
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


class BaseDetectionPipeline(ABC):
    """Base detection pipeline."""

    def __init__(self, config: PredictionConfig, loader_config: LoaderConfig):
        self.config = config
        self.loader_config = loader_config
        self.results_stats = []
        self.save_path: Optional[str] = None

        logger.info(f"Loading pipeline of type: {self.__class__.__name__}")

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

    @abstractmethod
    def run_detection(self, *args, **kwargs) -> List[DroneImage]:
        """Run detection."""
        pass

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

        for drone_image in drone_images:
            stats = drone_image.get_statistics()
            self.results_stats.append(stats)

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path_obj, "w", encoding="utf-8") as f:
            json.dump(self.results_stats, f, indent=2)

        logger.info(f"Results saved to: {save_path}")
        self.save_path = save_path

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

    def override_loading_config(
        self,
    ) -> None:
        if self.metadata is not None:
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
        return None


class DetectionPipeline(BaseDetectionPipeline):
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
        super().__init__(config, loader_config)

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
        if override_loading_config:
            self.override_loading_config()

        logger.info(f"Creating dataloader")

        data_loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
            use_tile_dataset=True,
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


class SimpleDetectionPipeline(BaseDetectionPipeline):
    def __init__(self, config: PredictionConfig, loader_config: LoaderConfig):
        super().__init__(config, loader_config)
        self.total_batches = 0
        self.total_tiles = 0

    def _run_one_image(self, loader: DataLoader) -> List[List[Detection]]:
        """Run detection on one image."""

        def process_one_image(image_as_patches: torch.Tensor) -> List[List[Detection]]:
            result = []
            b = min(self.loader_config.batch_size, image_as_patches.shape[0])
            for i in range(0, image_as_patches.shape[0], b):
                batch = image_as_patches[i : i + b]
                result.extend(self._process_batch(batch))
            return result

        for img, idx in loader:
            self.total_batches += 1
            self.total_tiles += img.shape[0]
            yield process_one_image(img), idx

    def _postprocess_one_image(
        self, detections: List[List[Detection]], offset_info: Dict
    ) -> List[DroneImage]:
        num_tiles = len(detections)
        assert (
            num_tiles == len(offset_info["y_offset"])
        ), f"len(detections) != len(offset_info['y_offset']) -> {num_tiles} != {len(offset_info['y_offset'])}"

        y_offset = offset_info["y_offset"]
        x_offset = offset_info["x_offset"]
        y_end = offset_info["y_end"]
        x_end = offset_info["x_end"]
        file_name = offset_info["file_name"]

        drone_image = DroneImage.from_image_path(
            image_path=file_name,
            flight_specs=self.loader_config.flight_specs,
        )

        # add tiles to drone image
        for i in range(num_tiles):
            tile = Tile(
                x_offset=x_offset[i],
                y_offset=y_offset[i],
                width=x_end[i] - x_offset[i],
                height=y_end[i] - y_offset[i],
            )
            if detections[i]:
                tile.set_predictions(detections[i], update_gps=False)
            else:
                tile.set_predictions([], update_gps=False)
            drone_image.add_tile(
                tile, tile.x_offset or 0, tile.y_offset or 0, nms_threshold=0.5
            )
        drone_image.update_detection_gps("predictions")

        return drone_image

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
        if override_loading_config:
            self.override_loading_config()

        self.save_path = save_path
        if self.save_path:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating dataloader")

        loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
            use_tile_dataset=False,
        )

        # Process batches
        all_drone_images = []
        for detections, idx in self._run_one_image(
            tqdm(loader, desc="Processing images", unit="image")
        ):
            offset_info = loader.get_offset_info(idx=idx)
            drone_image = self._postprocess_one_image(detections, offset_info)
            all_drone_images.append(drone_image)

            # Save results when completed
            if self.save_path:
                self._save_results(drone_image, mode="a")

        logger.info(
            f"Completed processing {self.total_batches} batches with {self.total_tiles} tiles"
        )

        return all_drone_images

    def _save_results(
        self,
        drone_image: DroneImage,
        mode: str = "a",
    ) -> None:
        """Save detection results.

        Args:
            drone_images: List of drone images with detections
            save_path: Path to save results
        """
        stats = drone_image.get_statistics()
        self.results_stats.append(stats)
        with open(self.save_path, mode, encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Results saved to: {self.save_path}")
