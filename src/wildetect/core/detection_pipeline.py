"""
Detection Pipeline for end-to-end wildlife detection processing.
"""

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from ..utils.utils import load_registered_model
from .config import LoaderConfig, PredictionConfig
from .data.detection import Detection
from .data.drone_image import DroneImage
from .data.loader import DataLoader
from .detectors.object_detection_system import ObjectDetectionSystem
from .factory import build_detector
from .processor.processor import Classifier, RoIPostProcessor

logger = logging.getLogger(__name__)


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
            self.config.cls_imgsz = roi_metadata.get("cls_imgsz", self.config.cls_imgsz)
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

        data_loader = DataLoader(
            image_paths=image_paths,
            image_dir=image_dir,
            config=self.loader_config,
        )

        # Process batches
        all_batches = []
        total_batches = len(data_loader)
        batch = None  # Initialize batch variable

        #with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for batch_idx, batch in tqdm(enumerate(data_loader), desc="Processing batches"):
            try:
                detections = self._process_batch(batch)
                batch["detections"] = detections
                all_batches.append(batch)
            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                self.error_count += 1
                continue
            finally:
                if self.error_count > 5:
                    raise RuntimeError("Too many errors. Stopping.")

        # Add detections to the last batch for postprocessing
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
