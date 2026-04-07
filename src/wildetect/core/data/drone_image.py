"""
Drone image representation with tiles and geographic footprint.
"""

import logging
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from wildata.converters.labelstudio.labelstudio_schemas import Task

from ..config import FlightSpecs
from ..config_models import ExifGPSUpdateConfig, LabelStudioConfigModel
from .detection import Detection
from .tile import Tile

logger = logging.getLogger(__name__)


@dataclass
class DroneImage(Tile):
    """Represents a drone image with tiles and geographic footprint.

    DroneImage extends Tile to add tile management functionality.
    A DroneImage represents the full original image, while containing
    multiple sub-tiles that were extracted from it.
    """

    # DroneImage-specific fields only (all other fields inherited from Tile)
    tiles: List[Tile] = field(default_factory=list)
    tile_offsets: List[Tuple[int, int]] = field(
        default_factory=list
    )  # (x_offset, y_offset)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize the drone image."""
        # Call parent's __post_init__ first
        super().__post_init__()

        # DroneImage-specific initialization
        if not self.tiles:
            # If no tiles provided, create a single tile from this image
            self._create_initial_tile()

    @property
    def geo_polygon_points(self) -> Optional[List[Tuple[float, float]]]:
        """Get the polygon points for the geographic footprint."""
        if self.geographic_footprint is not None:
            return self.geographic_footprint.get_polygon_points()
        else:
            logger.warning(f"No geographic footprint for {self.image_path}")
            return None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(vars(self))
        d.pop("image_data")
        d["tiles"] = [tile.to_dict() for tile in self.tiles]
        d["tile_offsets"] = self.tile_offsets
        d["type"] = "DroneImage"
        d["flight_specs"] = vars(self.flight_specs)
        d["predictions"] = [det.to_dict() for det in self.get_non_empty_predictions()]
        if self.geographic_footprint is not None:
            d["geographic_footprint"] = self.geographic_footprint.to_dict()
        return d

    def get_non_empty_predictions(self) -> List[Detection]:
        """Get all non-empty predictions from all tiles."""
        return [det for det in self.get_all_predictions() if not det.is_empty]

    def get_non_empty_annotations(self) -> List[Detection]:
        """Get all non-empty annotations from all tiles."""
        return [det for det in self.get_all_annotations() if not det.is_empty]

    def _create_initial_tile(self):
        """Create initial tile from the drone image itself."""
        # Create a tile representing the full image
        full_tile = Tile.from_image_path(
            image_path=self.image_path,
            flight_specs=self.flight_specs,
            width=self.width,
            height=self.height,
            latitude=self.latitude,
            longitude=self.longitude,
            gsd=self.gsd,
            is_raster=self.is_raster,
            altitude=self.altitude,
        )

        # Add it as the first tile with no offset
        self.add_tile(full_tile, 0, 0)

    def add_tile(
        self,
        tile: Tile,
        x_offset: int,
        y_offset: int,
        nms_threshold: float = 0.5,
        offset_detections: bool = True,
    ) -> None:
        """Add a tile with its offset to the drone image.

        Args:
            tile (Tile): Tile to add
            x_offset (int): x offset of the tile
            y_offset (int): y offset of the tile
            nms_threshold (float): nms threshold. NMS is disabled if 0.0
            offset_detections (bool): whether to offset detections
        """
        if not isinstance(tile, Tile):
            raise TypeError(f"Expected Tile object, got {type(tile)}")

        # Set the tile's parent image and offset
        tile.parent_image = self.image_path

        # set offsets
        tile.set_offsets(x_offset, y_offset)

        # offset detections -> DroneImage reference coordinates
        if offset_detections:
            tile.offset_detections()

        # add tile to drone image
        self.tiles.append(tile)
        self.tile_offsets.append((x_offset, y_offset))

        # deepcopy predictions
        predictions = deepcopy(tile.predictions)
        for det in predictions:
            if det.parent_image != self.image_path:
                det.set_distance_to_centroid(
                    self.image_path, image_width=self.width, image_height=self.height
                )

        # add predictions to drone image
        self.predictions.extend(predictions)

        # filter detections
        if nms_threshold > 0.0:
            self.filter_detections(
                method="nms",
                threshold=nms_threshold,
                clamp=True,
                confidence_threshold=0.0,
            )

        logger.debug(f"Added tile {tile.id} at offset {x_offset}, {y_offset}")

    def get_tiles_at_position(self, x: int, y: int) -> List[Tile]:
        """Get tile at a specific position.

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            Optional[Tile]: Tile at the position, or None if not found
        """
        tiles = []
        for tile, (offset_x, offset_y) in zip(self.tiles, self.tile_offsets):
            if offset_x <= x < (offset_x + tile.width) and offset_y <= y < (
                offset_y + tile.height
            ):
                tiles.append(tile)
        return tiles

    def get_tiles_in_region(self, x1: int, y1: int, x2: int, y2: int) -> List[Tile]:
        """Get all tiles that overlap with a region.

        Args:
            x1, y1 (int): Top-left corner of region
            x2, y2 (int): Bottom-right corner of region

        Returns:
            List[Tile]: List of tiles overlapping with the region
        """
        overlapping_tiles = []

        for tile, (offset_x, offset_y) in zip(self.tiles, self.tile_offsets):
            tile_x1, tile_y1 = offset_x, offset_y
            tile_x2, tile_y2 = (offset_x + tile.width), (offset_y + tile.height)

            # Check for overlap
            if not (tile_x2 < x1 or tile_x1 > x2 or tile_y2 < y1 or tile_y1 > y2):
                overlapping_tiles.append(tile)

        return overlapping_tiles

    def offset_detections(
        self,
    ):
        """Offset detections based on tile offsets if not offset already."""
        if not self.tile_offsets:
            logger.warning(
                "No tile offsets found for drone image. Skipping offsetting detections."
            )
            return

        for offset, tile in zip(self.tile_offsets, self.tiles):
            if not tile.x_offset or not tile.y_offset:
                tile.set_offsets(offset[0], offset[1])
                tile.offset_detections()

    def get_all_predictions(self) -> List[Detection]:
        """Get all predictions from all tiles.

        Returns:
            List[Detection]: All predictions from all tiles
        """
        if self.predictions:
            return self.predictions
        all_detections = []
        for tile in self.tiles:
            if tile.predictions:
                all_detections.extend(
                    [det for det in tile.predictions if not det.is_empty]
                )
        # set predictions
        self.predictions = all_detections
        return self.predictions

    def get_all_annotations(self) -> List[Detection]:
        """Get all annotations from all tiles."""
        if self.annotations:
            return self.annotations
        all_detections = []
        for tile in self.tiles:
            if tile.annotations:
                all_detections.extend(
                    [det for det in tile.annotations if not det.is_empty]
                )
        # set annotations
        self.annotations = all_detections
        return self.annotations

    def get_predictions_in_region(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> List[Detection]:
        """Get all predictions in a specific region.

        Args:
            x1, y1 (int): Top-left corner of region
            x2, y2 (int): Bottom-right corner of region

        Returns:
            List[Detection]: Predictions in the region
        """
        region_detections = []
        overlapping_tiles = self.get_tiles_in_region(x1, y1, x2, y2)

        for tile in overlapping_tiles:
            for detection in tile.predictions or []:
                # Check if detection is in region
                if x1 <= detection.x_center <= x2 and y1 <= detection.y_center <= y2:
                    region_detections.append(detection)

            for detection in tile.annotations or []:
                # Check if detection is in region
                if x1 <= detection.x_center <= x2 and y1 <= detection.y_center <= y2:
                    region_detections.append(detection)

        return region_detections

    def merge_detections(
        self, method="nms", threshold=0.5, clamp=True
    ) -> List[Detection]:
        """Merge detections from all tiles, handling overlaps.

        Returns:
            List[Detection]: Merged detections
        """
        self.get_all_predictions()
        self.filter_detections(
            method=method, threshold=threshold, clamp=clamp, confidence_threshold=0.0
        )
        return self.predictions

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the drone image.

        Returns:
            Dict[str, Any]: Statistics about the image
        """
        all_detections = self.get_non_empty_predictions()

        # Count detections by class
        class_counts = {}
        for detection in all_detections:
            class_name = detection.class_name or "unknown"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "image_id": self.id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "num_tiles": len(self.tiles),
            "total_detections": len(all_detections),
            "class_counts": class_counts,
            "all_detections": [
                det.bbox + [det.class_name, det.confidence, *det.gps_as_decimals]
                for det in all_detections
            ],
            "has_gps": (self.latitude is not None) and (self.longitude is not None),
            # "has_geographic_footprint": self.geographic_footprint is not None,
            "gps_loc": self.tile_gps_loc,
            # "polygon_points": self.geo_polygon_points,
            "gsd": self.gsd,
            "timestamp": self.timestamp,
        }

    def draw_detections(
        self,
        image: Optional[np.ndarray] = None,
        predictions: bool = True,
        prediction_color: Tuple[int, int, int] = (0, 255, 0),
        line_thickness: int = 2,
        font_scale: float = 0.5,
        show_confidence: bool = True,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw all detections on the full image.

        Args:
            image: Input image as numpy array. If None, uses drone image data
            predictions: Whether to draw predictions
            annotations: Whether to draw annotations
            prediction_color: BGR color for prediction boxes
            annotation_color: BGR color for annotation boxes
            line_thickness: Thickness of bounding box lines
            font_scale: Scale of the font for labels
            show_confidence: Whether to show confidence scores in labels
            save_path: Optional path to save the resulting image

        Returns:
            Image with detections drawn
        """
        import cv2

        # Load full image if not provided
        if image is None:
            pil_image = self.load_image_data()
            image = np.array(pil_image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = image.copy()

        # Draw detections from all tiles
        all_detections = self.get_non_empty_predictions()

        for detection in all_detections:
            # Get absolute coordinates
            bbox = detection.bbox
            class_name = detection.class_name or "unknown"
            confidence = detection.confidence

            # Draw bounding box
            cv2.rectangle(
                result,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                prediction_color,
                line_thickness,
            )

            # Draw label
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name

            # Calculate text position
            text_x = bbox[0]
            text_y = bbox[1] - 10 if bbox[1] - 10 > 0 else bbox[1] + 20

            # Draw text background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )
            cv2.rectangle(
                result,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + 5),
                prediction_color,
                -1,
            )

            # Draw text
            cv2.putText(
                result,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
            )

        # Save image if path is provided
        if save_path:
            cv2.imwrite(save_path, result)
            logger.info(f"Image with detections saved to: {save_path}")

        return result

    @classmethod
    def from_image_path(cls, image_path: str, **kwargs) -> "DroneImage":
        """Create drone image from image path."""
        return cls(image_path=image_path, **kwargs)

    @classmethod
    def from_image_data(cls, image_data: Image.Image, **kwargs) -> "DroneImage":
        """Create drone image from image data."""
        return cls(image_data=image_data, **kwargs)

    @classmethod
    def from_ls(
        cls,
        flight_specs: FlightSpecs,
        labelstudio_config: Optional[LabelStudioConfigModel] = None,
        exif_gps_update: Optional[ExifGPSUpdateConfig] = None,
        load_annotations: bool = True,
        load_predictions: bool = True,
    ) -> List["DroneImage"]:
        from ..visualization.labelstudio_manager import LabelStudioManager

        if load_predictions:
            raise NotImplementedError

        csv_dict = None
        errors_gps = 0
        if exif_gps_update is not None:
            df = exif_gps_update.to_df()
            cfg_rename = {
                exif_gps_update.lat_col: "latitude",
                exif_gps_update.lon_col: "longitude",
                exif_gps_update.alt_col: "altitude",
                exif_gps_update.filename_col: "image_path",
            }
            csv_dict = (
                df.rename(columns=cfg_rename)
                .set_index("image_path")
                .to_dict(orient="index")
            )

        def get_image_gps_coords(
            img_path: str,
        ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
            if csv_dict is None:
                return None, None, None
            else:
                if img_path not in csv_dict:
                    logger.error(f"Image path {img_path} not found in csv_dict")
                    errors_gps += 1
                    if errors_gps > 5:
                        raise Exception("Image paths not found in csv_dict.")
                    return None, None, None
                return (
                    csv_dict[img_path]["latitude"],
                    csv_dict[img_path]["longitude"],
                    csv_dict[img_path]["altitude"],
                )

        assert (labelstudio_config.project_id is not None) ^ (
            labelstudio_config.json_path is not None
        ), "Provide either `project_id` or `json_path`"

        ls_client = LabelStudioManager(
            url=labelstudio_config.url,
            api_key=labelstudio_config.api_key,
            download_resources=labelstudio_config.download_resources,
        )
        if isinstance(labelstudio_config.project_id, int):
            all_tasks = ls_client.get_tasks(labelstudio_config.project_id)
            logger.info("Loading from project_id")
        else:
            all_tasks = ls_client.get_tasks_from_json(labelstudio_config.json_path)
            logger.info("Loading from json file")

        all_drone_images = []
        errors = 0
        failed_paths = []
        for task in tqdm(
            all_tasks, total=len(all_tasks), desc="Loading images from Label Studio"
        ):
            try:
                latitude, longitude, altitude = get_image_gps_coords(task.image_path)
                image = DroneImage.from_ls_task(
                    task=task,
                    flight_specs=flight_specs,
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    load_predictions=load_predictions,
                    load_annotations=load_annotations,
                )
                all_drone_images.append(image)
            except Exception as e:
                logger.error(f"Failed to load image {task.image_path}: {e}")
                logger.debug(f"task dump: {task.model_dump()}")
                failed_paths.append(task.image_path)
                errors += 1
        
        if len(failed_paths):
            logger.error(f"Failed for these images: {failed_paths}")

        return all_drone_images

    @classmethod
    def from_ls_task(
        cls,
        task: Task,
        flight_specs: FlightSpecs,
        load_annotations: bool = True,
        load_predictions: bool = True,
        **kwargs,
    ) -> "DroneImage":
        """Create a DroneImage from a Label Studio task ID.

        Uses the Task schema from wildata to parse the Label Studio task
        and extract annotations and predictions.

        Args:
            task_id: Label Studio task ID to fetch
            flight_specs: Flight specifications for the drone image
            labelstudio_config: Label Studio configuration with url and api_key

        Returns:
            DroneImage with annotations and predictions from the task
        """

        # Get image path from task data
        image_path = task.image_path

        # Create DroneImage
        drone_image = cls(image_path=image_path, flight_specs=flight_specs, **kwargs)
        update_gps = drone_image.tile_gps_loc is not None

        # Extract annotations from the task
        if load_annotations:
            annotations = []
            # for annotation in task.annotations:
            annotations.extend(Detection.from_ls(task.annotations, image_path))
            drone_image.set_annotations(annotations, update_gps=update_gps)

        # Extract predictions from the task
        if load_predictions:
            raise NotImplementedError("Not yet implemented")
            predictions = []
            predictions.extend(Detection.from_ls(task.predictions, image_path))
            drone_image.set_predictions(predictions, update_gps=update_gps)

        return drone_image
