"""
Detection data structures for wildlife detection results.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import fiftyone as fo
import numpy as np
from PIL import Image

from wildetect.core.gps.geographic_bounds import GeographicBounds
from wildetect.core.gps.gps_utils import GPSUtils

if TYPE_CHECKING:
    from .tile import Tile

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Enhanced detection result with optional GPS features."""

    # Core detection fields
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    metadata: Optional[Dict[str, Any]] = None

    # Optional GPS fields
    geographic_footprint: Optional[GeographicBounds] = None
    gps_loc: Optional[str] = None
    image_gps_loc: Optional[str] = None
    parent_image: Optional[str] = None

    def __post_init__(self):
        """Calculate derived fields if not provided."""
        assert (
            isinstance(self.bbox, list) and len(self.bbox) == 4
        ), f"bbox must be a list of 4 elements, but got {self.bbox}"
        assert all(
            [isinstance(coord, int) for coord in self.bbox]
        ), f"bbox must be a list of integers, but got {self.bbox}"

        assert all(
            map(lambda x: isinstance(x, int), self.bbox)
        ), f"bbox must be a list of integers, but got {self.bbox}"

        x1, y1, x2, y2 = self.bbox
        if x1 > x2 or y1 > y2:
            raise ValueError(
                "x2 must be greater than x1 and y2 must be greater than y1, "
                f"but got x1={x1}, y1={y1}, x2={x2}, y2={y2}"
            )

        self.set_distance_to_centroid()

    def set_distance_to_centroid(self, parent_image: Optional[str] = None) -> None:
        if parent_image:
            self.parent_image = parent_image
        self.distance_to_centroid = self._get_distance_to_centroid()

    @property
    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        d = dict(vars(self))
        if "bbox" in d:
            x1, y1, x2, y2 = d["bbox"]
            d["x_min"] = x1
            d["y_min"] = y1
            d["x_max"] = x2
            d["y_max"] = y2
        if isinstance(d.get("geographic_footprint"), GeographicBounds):
            d["geographic_footprint"] = d["geographic_footprint"].to_dict()
        d["type"] = "Detection"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        attributes = [
            "bbox",
            "confidence",
            "class_id",
            "class_name",
            "metadata",
            "geographic_footprint",
            "gps_loc",
            "image_gps_loc",
            "parent_image",
        ]
        cfg = {k: data.get(k) for k in attributes}
        if isinstance(cfg.get("geographic_footprint"), dict):
            cfg["geographic_footprint"] = GeographicBounds.from_dict(
                cfg["geographic_footprint"]
            )
        return cls(**cfg)

    @property
    def x_center(self) -> int:
        return int((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def y_center(self) -> int:
        return int((self.bbox[1] + self.bbox[3]) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_empty(self) -> bool:
        """Check if detection is empty."""
        return sum(self.bbox) == 0

    @property
    def gps_as_decimals(self) -> Optional[Tuple[float, float, float]]:
        """Return GPS coordinates as decimal degrees."""
        if self.gps_loc is None:
            return None, None, None
        return GPSUtils.to_decimal(self.gps_loc)

    def get_bbox(self) -> List[int]:
        """Return bbox as list for compatibility with Tile class."""
        return self.bbox

    def _get_distance_to_centroid(
        self,
    ) -> float:
        """Compute the distance from the detection to the centroid of the parent image."""
        if self.parent_image is None:
            return 99999

        with Image.open(self.parent_image) as image:
            width, height = image.size
            distance_to_centroid = math.sqrt(
                (self.x_center - width / 2) ** 2 + (self.y_center - height / 2) ** 2
            )

        return distance_to_centroid

    @property
    def geo_box(self):
        if self.geographic_footprint is None:
            return None
        return self.geographic_footprint.box

    def update_values_from_tile(self, tile: "Tile") -> None:
        """Update detection with tile context."""
        self.parent_image = tile.image_path
        self.image_gps_loc = tile.tile_gps_loc

    def clamp_bbox(self, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> None:
        """Clamp bbox coordinates to image bounds."""
        x1, y1, x2, y2 = self.bbox
        x1 = max(x_range[0], min(x_range[1], x1))
        y1 = max(y_range[0], min(y_range[1], y1))
        x2 = max(x_range[0], min(x_range[1], x2))
        y2 = max(y_range[0], min(y_range[1], y2))
        self.bbox = [x1, y1, x2, y2]

    @classmethod
    def empty(cls, parent_image: str) -> "Detection":
        """Create an empty detection."""
        return cls(
            bbox=[0, 0, 0, 0],
            confidence=0.0,
            class_id=0,
            class_name="EMPTY",
            parent_image=parent_image,
        )

    def to_fiftyone(self, image_width: int, image_height: int) -> fo.Detection:
        """Convert to FiftyOne detection format.

        Returns:
            FiftyOne detection object (if fiftyone is available)
        """

        # Extract bounding box coordinates
        x1, y1, x2, y2 = self.bbox

        # Create FiftyOne detection
        fo_detection = fo.Detection(
            label=self.class_name,
            confidence=self.confidence,
            bounding_box=[
                x1 / image_width,
                y1 / image_height,
                (x2 - x1) / image_width,
                (y2 - y1) / image_height,
            ],  # [x, y, width, height]
            metadata={
                "class_id": self.class_id,
                "area": self.area,
                "x_center": self.x_center,
                "y_center": self.y_center,
                "original_bbox": self.bbox,
                "gps_loc": self.gps_loc,
                "image_gps_loc": self.image_gps_loc,
                "parent_image": self.parent_image,
                **(self.metadata or {}),
            },
        )

        return fo_detection

    @classmethod
    def from_fiftyone(
        cls, fo_detection, class_id: Optional[int] = None
    ) -> Optional["Detection"]:
        """Create Detection from FiftyOne detection.

        Args:
            fo_detection: FiftyOne detection object
            class_id: Optional class ID (if not provided, will be set to 0)

        Returns:
            Detection object or None if FiftyOne is not available
        """
        try:
            # Extract bounding box (FiftyOne uses [x, y, width, height])
            x, y, width, height = fo_detection.bounding_box

            # Convert to [x1, y1, x2, y2] format
            bbox = [int(x), int(y), int(x + width), int(y + height)]

            # Extract metadata
            metadata = fo_detection.metadata or {}
            original_bbox = metadata.get("original_bbox", bbox)
            gps_loc = metadata.get("gps_loc")
            image_gps_loc = metadata.get("image_gps_loc")
            parent_image = metadata.get("parent_image")

            # Use provided class_id or extract from metadata, default to 0
            if class_id is None:
                class_id = metadata.get("class_id", 0)

            return cls(
                bbox=original_bbox,
                confidence=fo_detection.confidence,
                class_id=class_id or 0,
                class_name=fo_detection.label,
                gps_loc=gps_loc,
                image_gps_loc=image_gps_loc,
                parent_image=parent_image,
                metadata=metadata,
            )

        except ImportError:
            logger.warning(
                "FiftyOne not available, cannot convert from FiftyOne format"
            )
            return None

    def set_gps_location(
        self,
        gps_loc: str,
    ) -> None:
        """Set the GPS location for this detection.

        Args:
            gps_loc (str): GPS location string
        """
        assert isinstance(
            gps_loc, str
        ), f"gps_loc must be a string, but got {type(gps_loc)}"
        self.gps_loc = gps_loc

    def set_image_gps_location(self, image_gps_loc: str) -> None:
        """Set the image GPS location for this detection.

        Args:
            image_gps_loc (str): Image GPS location string
        """
        self.image_gps_loc = image_gps_loc

    def set_parent_image(self, parent_image: str) -> None:
        """Set the parent image path for this detection.

        Args:
            parent_image (str): Parent image path
        """
        self.parent_image = parent_image

    def set_geographic_footprint(self, geographic_footprint: GeographicBounds) -> None:
        """Set the geographic footprint for this detection.

        Args:
            geographic_footprint (GeographicBounds): Geographic footprint
        """
        assert isinstance(
            geographic_footprint, GeographicBounds
        ), f"geographic_footprint must be a GeographicBounds, but got {type(geographic_footprint)}"
        self.geographic_footprint = geographic_footprint

    def has_gps_data(self) -> bool:
        """Check if this detection has GPS data.

        Returns:
            bool: True if GPS data is available
        """
        return (
            self.gps_loc is not None
            or self.image_gps_loc is not None
            or self.geographic_footprint is not None
        )

    def get_gps_summary(self) -> Dict[str, Any]:
        """Get a summary of GPS data for this detection.

        Returns:
            dict: GPS data summary
        """
        return {
            "has_gps": self.has_gps_data(),
            "gps_loc": self.gps_loc,
            "image_gps_loc": self.image_gps_loc,
            "parent_image": self.parent_image,
            "has_geographic_footprint": self.geographic_footprint is not None,
        }

    def to_absolute_coords(self, x_offset: int, y_offset: int) -> None:
        """Convert relative coordinates to absolute image coordinates by applying offsets.

        Args:
            x_offset (int): Offset to add to x coordinates.
            y_offset (int): Offset to add to y coordinates.
        """
        self.bbox[0] += x_offset
        self.bbox[1] += y_offset
        self.bbox[2] += x_offset
        self.bbox[3] += y_offset
