"""
Geographic bounds utilities for wildlife detection.
"""

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from wildetect.utils.utils import compute_iou

logger = logging.getLogger(__name__)


@dataclass
class GeographicBounds:
    """Geographic bounding box for image footprint in UTM coordinates"""

    north: float  # Max latitude
    south: float  # Min latitude
    east: float  # Max longitude
    west: float  # Min longitude

    # Additional metadata for polygon computation
    lat_center: Optional[float] = None  # Center latitude of the image
    lon_center: Optional[float] = None  # Center longitude of the image
    width_px: Optional[int] = None  # Image width in pixels
    height_px: Optional[int] = None  # Image height in pixels
    gsd: Optional[float] = None  # Ground sample distance in cm/px

    # Cached polygon points
    _polygon_points: Optional[List[Tuple[float, float]]] = field(
        default=None, repr=False
    )

    def __post_init__(self):
        self._polygon_points = self.get_polygon_points()

    def to_dict(
        self
    ) -> Dict[str, Union[List[Tuple[float, float]], List[float], float]]:
        d = vars(self)
        d["polygon_points"] = self.get_polygon_points()
        d["box"] = self.box
        d["center"] = self.center
        d["area"] = self.area
        d["type"] = "GeographicBounds"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeographicBounds":
        attributes = [
            "north",
            "south",
            "east",
            "west",
            "lat_center",
            "lon_center",
            "width_px",
            "height_px",
            "gsd",
        ]
        cfg = {k: data.get(k) for k in attributes}
        return cls(**cfg)

    @property
    def area(self) -> float:
        """Calculate area in square degrees covered by the bounding box."""
        return (self.east - self.west) * (self.north - self.south)

    @property
    def box(self) -> List[float]:
        """Get bounding box as [west, south, east, north]."""
        return [self.west, self.south, self.east, self.north]

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates of the bounding box."""
        return ((self.north + self.south) / 2, (self.east + self.west) / 2)

    def overlap_ratio(self, other: "GeographicBounds") -> float:
        """Calculate overlap ratio (IoU) with another bounds using torchmetrics IntersectionOverUnion.
        Args:
            other (GeographicBounds): Another geographic bounds object.
        Returns:
            float: Overlap ratio (IoU) between the two bounds.
        """
        box_self = self.box
        box_other = other.box

        return compute_iou(box_self, box_other)

    def overlap_area(self, other: "GeographicBounds") -> float:
        main_bounds = self
        overlap_bounds = other

        overlap_west = max(main_bounds.west, overlap_bounds.west)
        overlap_east = min(main_bounds.east, overlap_bounds.east)
        overlap_south = max(main_bounds.south, overlap_bounds.south)
        overlap_north = min(main_bounds.north, overlap_bounds.north)

        overlap_area = 0.0
        if overlap_west < overlap_east and overlap_south < overlap_north:
            overlap_area = (overlap_east - overlap_west) * (
                overlap_north - overlap_south
            )

        return overlap_area

    def expand(self, margin: float) -> "GeographicBounds":
        """Expand the bounds by a margin.

        Args:
            margin (float): Margin to expand by

        Returns:
            GeographicBounds: Expanded bounds
        """
        return GeographicBounds(
            north=self.north + margin,
            south=self.south - margin,
            east=self.east + margin,
            west=self.west - margin,
            lat_center=self.lat_center,
            lon_center=self.lon_center,
            width_px=self.width_px,
            height_px=self.height_px,
            gsd=self.gsd,
        )

    def get_polygon_points(
        self, num_points_per_edge: int = 10
    ) -> Optional[List[Tuple[float, float]]]:
        """Get polygon points representing the image footprint."""
        if self._polygon_points is not None:
            return self._polygon_points

        if not self._can_compute_polygon():
            logger.warning("Cannot compute polygon: missing required metadata")
            return None

        try:
            from ..gps.gps_utils import get_pixel_gps_coordinates

            # Generate polygon boundary coordinates
            width, height = self.width_px or 0, self.height_px or 0

            # Create edge coordinates efficiently
            edges = [
                (
                    np.linspace(0, width, num_points_per_edge),
                    np.zeros(num_points_per_edge),
                ),  # Top
                (
                    np.full(num_points_per_edge, width),
                    np.linspace(0, height, num_points_per_edge),
                ),  # Right
                (
                    np.linspace(width, 0, num_points_per_edge),
                    np.full(num_points_per_edge, height),
                ),  # Bottom
                (
                    np.zeros(num_points_per_edge),
                    np.linspace(height, 0, num_points_per_edge),
                ),  # Left
            ]

            x_coords = np.concatenate([edge[0] for edge in edges])
            y_coords = np.concatenate([edge[1] for edge in edges])

            # Compute GPS coordinates
            result = get_pixel_gps_coordinates(
                x=x_coords,
                y=y_coords,
                lat_center=self.lat_center or 0.0,
                lon_center=self.lon_center or 0.0,
                W=width,
                H=height,
                gsd=self.gsd or 0.0,
                return_as_utm=False,
            )

            # Convert result to polygon points
            if isinstance(result, tuple) and len(result) == 2:
                lats, lons = result
                if isinstance(lats, np.ndarray) and isinstance(lons, np.ndarray):
                    polygon_points = [
                        (float(lats[i]), float(lons[i])) for i in range(lats.size)
                    ]
                elif isinstance(lats, (list, np.ndarray)) and isinstance(
                    lons, (list, np.ndarray)
                ):
                    # Handle list/array inputs
                    lats_list = lats if isinstance(lats, list) else lats.tolist()
                    lons_list = lons if isinstance(lons, list) else lons.tolist()
                    polygon_points = [
                        (float(lats_list[i]), float(lons_list[i]))
                        for i in range(len(lats_list))
                    ]
                else:
                    # Handle single values
                    polygon_points = [(float(lats), float(lons))]
            else:
                # Fallback for unexpected result format
                logger.warning(
                    f"Unexpected result format from get_pixel_gps_coordinates: {type(result)}"
                )
                return None

            self._polygon_points = polygon_points
            return polygon_points

        except Exception as e:
            logger.warning(f"Failed to create polygon from GPS: {e}")
            traceback.print_exc()
            raise Exception
            return None

    def _can_compute_polygon(self) -> bool:
        """Check if we have the necessary metadata to compute polygon."""
        return all(
            [
                self.lat_center is not None,
                self.lon_center is not None,
                self.width_px is not None,
                self.height_px is not None,
                self.gsd is not None,
            ]
        )

    def clear_polygon_cache(self) -> None:
        """Clear the cached polygon points."""
        self._polygon_points = None

    @classmethod
    def from_image_metadata(
        cls,
        lat_center: float,
        lon_center: float,
        width_px: int,
        height_px: int,
        gsd: float,
    ) -> Optional["GeographicBounds"]:
        """Create GeographicBounds from image metadata with polygon computation.

        Args:
            lat_center: Center latitude of the image
            lon_center: Center longitude of the image
            width_px: Image width in pixels
            height_px: Image height in pixels
            gsd: Ground sample distance in cm/px
            num_points_per_edge: Number of points to sample along each edge

        Returns:
            GeographicBounds instance or None if creation fails
        """
        try:
            from ..gps.gps_utils import get_pixel_gps_coordinates

            # Calculate the four corner coordinates
            corners_x = np.array([0, width_px, width_px, 0])
            corners_y = np.array([0, 0, height_px, height_px])

            # Get GPS coordinates for corners
            lats, lons = get_pixel_gps_coordinates(
                x=corners_x,
                y=corners_y,
                lat_center=lat_center,
                lon_center=lon_center,
                W=width_px,
                H=height_px,
                gsd=gsd,
                return_as_utm=True,
            )

            # Convert to float values
            lats = (
                [float(lat) for lat in lats]
                if isinstance(lats, (list, np.ndarray))
                else [float(lats)]
            )
            lons = (
                [float(lon) for lon in lons]
                if isinstance(lons, (list, np.ndarray))
                else [float(lons)]
            )

            # Create bounds
            bounds = cls(
                north=max(lats),
                south=min(lats),
                east=max(lons),
                west=min(lons),
                lat_center=lat_center,
                lon_center=lon_center,
                width_px=width_px,
                height_px=height_px,
                gsd=gsd,
            )

            return bounds

        except Exception as e:
            logger.error(f"Failed to create GeographicBounds from metadata: {e}")
            return None
