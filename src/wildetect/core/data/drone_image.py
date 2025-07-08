"""
Drone image representation with tiles and geographic footprint.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

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
    tile_offsets: List[Tuple[int, int]] = field(default_factory=list)
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

    def get_non_empty_predictions(self) -> List[Detection]:
        """Get all non-empty predictions from all tiles."""
        return [det for det in self.get_all_predictions() if not det.is_empty]

    def _create_initial_tile(self):
        """Create initial tile from the drone image itself."""
        # Create a tile representing the full image
        full_tile = Tile.from_image_path(
            image_path=self.image_path, flight_specs=self.flight_specs
        )

        # Add it as the first tile with no offset
        self.add_tile(full_tile, 0, 0)

    def add_tile(self, tile: Tile, x_offset: int, y_offset: int) -> None:
        """Add a tile with its offset to the drone image.

        Args:
            tile (Tile): Tile to add
            x_offset (int): x offset of the tile
            y_offset (int): y offset of the tile
        """
        if not isinstance(tile, Tile):
            raise TypeError(f"Expected Tile object, got {type(tile)}")

        # Set the tile's parent image and offset
        tile.parent_image = self.image_path
        # set offsets
        tile.set_offsets(x_offset, y_offset)
        # offset detections -> DrroneImage reference coordinates
        tile.offset_detections()
        # add tile to drone image
        self.tiles.append(tile)
        self.tile_offsets.append((x_offset, y_offset))
        # add predictions to drone image
        self.predictions.extend(tile.predictions)

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

        # self.offset_detections()
        # self.update_detection_gps("predictions")
        for tile in self.tiles:
            if tile.predictions:
                all_detections.extend(
                    [det for det in tile.predictions if not det.is_empty]
                )

        self.predictions = all_detections
        return self.predictions

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
            "has_gps": self.latitude is not None and self.longitude is not None,
            "has_geographic_footprint": self.geographic_footprint is not None,
            "gps_loc": self.tile_gps_loc,
            "polygon_points": self.geo_polygon_points,
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
