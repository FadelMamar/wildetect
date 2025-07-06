"""
Detection Pipeline for end-to-end wildlife detection processing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from .config import LoaderConfig, PredictionConfig
from .data.detection import Detection
from .data.drone_image import DroneImage
from .data.loader import DataLoader
from .detectors.object_detection_system import ObjectDetectionSystem
from .factory import build_detector

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """End-to-end detection pipeline combining dataloader with inference engine."""

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
        device: str = "cpu",
    ):
        """Initialize the detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
            device: Device to run inference on
        """
        self.config = config
        self.loader_config = loader_config
        self.device = device or config.device

        assert config.model_path is not None, "Model path must be provided"
        assert config.model_type is not None, "Model type must be provided"

        # Initialize components
        self.detection_system: Optional[ObjectDetectionSystem] = None
        self.data_loader: Optional[DataLoader] = None

        self.setup()

        logger.info(
            f"Initialized DetectionPipeline with model_type={config.model_type}"
        )

    def setup(self) -> None:
        """Set up the inference engine with model and processors."""
        try:
            # Build detector
            detector = build_detector(
                config=self.config,
            )
            # Create object detection system
            self.detection_system = ObjectDetectionSystem(
                config=self.config,
            )
            self.detection_system.set_model(detector)

            logger.info("Detection pipeline setup completed")

        except Exception as e:
            logger.error(f"Failed to setup inference engine: {e}")
            raise

    def process_batch(
        self,
        batch: Dict[str, Any],
        progress_bar: Optional[tqdm] = None,
    ) -> List[List[Detection]]:
        """Process a single batch of tiles."""
        # Run inference on tiles
        try:
            if self.detection_system is None:
                raise ValueError("Detection system not initialized")

            detections = self.detection_system.predict(batch["images"])
            # Update progress bar
            if progress_bar:
                progress_bar.update(len(batch["tiles"]))
            return detections

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            return []

    def _postprocess(self, batch: Dict[str, Any]) -> List[DroneImage]:
        """Post-process batch results and convert to DroneImage objects.

        Args:
            batch: Batch containing tiles and detections

        Returns:
            List of DroneImage objects with detections
        """
        if not batch.get("detections"):
            logger.warning("No detections found in batch")
            return []

        # Group tiles by parent image
        drone_images: Dict[str, DroneImage] = {}

        # Process each tile and its detections
        for i, tile in enumerate(batch["tiles"]):
            parent_image = tile.parent_image or tile.image_path

            # Create or get drone image for this parent
            if parent_image not in drone_images:
                drone_image = DroneImage.from_image_path(
                    image_path=parent_image,
                    flight_specs=self.loader_config.flight_specs,
                )
                drone_images[parent_image] = drone_image

            # Get detections for this tile
            tile_detections = batch["detections"][i]

            # Set detections on the tile
            if tile_detections:
                tile.set_predictions(tile_detections)
            else:
                # Set empty detection if no detections found
                tile.set_predictions([])

            # Add tile to drone image with its offset
            offset = (tile.x_offset or 0, tile.y_offset or 0)
            drone_images[parent_image].add_tile(tile, offset[0], offset[1])

        return list(drone_images.values())

    def run_detection(
        self,
        image_paths: List[str],
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

        data_loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
        )

        # Process batches
        all_detections = []
        total_batches = len(data_loader)
        batch = None  # Initialize batch variable

        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for batch_idx, batch in enumerate(data_loader):
                try:
                    detections = self.process_batch(batch, pbar)
                    all_detections.append(detections)
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_idx}: {e}")
                    continue

        # Add detections to the last batch for postprocessing
        if batch is not None:
            batch["detections"] = all_detections
            all_drone_images = self._postprocess(batch=batch)
        else:
            logger.warning("No batches were processed")
            all_drone_images = []

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

        with open(save_path_obj, "w") as f:
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
