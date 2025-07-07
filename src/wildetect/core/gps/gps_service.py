"""
GPS service for handling GPS-related detection operations.
"""

import logging
import traceback
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import geopy
import numpy as np
from PIL import Image

from ..data.detection import Detection
from ..gps.geographic_bounds import GeographicBounds
from ..gps.gps_utils import GPSUtils, get_pixel_gps_coordinates

if TYPE_CHECKING:
    from ..data.tile import Tile

logger = logging.getLogger(__name__)


def create_geographic_footprint(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    lat_center_roi: float,
    long_center_roi: float,
    width_roi: int,
    height_roi: int,
    gsd: float,
) -> Optional[GeographicBounds]:
    if lat_center_roi is None or long_center_roi is None:
        return None

    try:
        # Use the new GeographicBounds.from_image_metadata method for better polygon support
        bounds = GeographicBounds.from_image_metadata(
            lat_center=lat_center_roi,
            lon_center=long_center_roi,
            width_px=width_roi,
            height_px=height_roi,
            gsd=gsd,
        )

        if bounds is None:
            # Fallback to original method if the new method fails
            # Calculate bounding box corners in GPS coordinates
            xs = np.array([x1, x2])
            ys = np.array([y1, y2])

            xs_utm, ys_utm = get_pixel_gps_coordinates(
                x=xs,
                y=ys,
                lat_center=lat_center_roi,
                lon_center=long_center_roi,
                W=width_roi,
                H=height_roi,
                gsd=gsd,
                return_as_utm=True,
            )

            # Handle both single values and arrays
            if isinstance(xs_utm, (list, np.ndarray)):
                east_max = float(np.max(xs_utm))
                east_min = float(np.min(xs_utm))
            else:
                east_max = east_min = float(xs_utm)

            if isinstance(ys_utm, (list, np.ndarray)):
                north_max = float(np.max(ys_utm))
                north_min = float(np.min(ys_utm))
            else:
                north_max = north_min = float(ys_utm)

            bounds = GeographicBounds(
                north=north_max,
                south=north_min,
                east=east_max,
                west=east_min,
                lat_center=lat_center_roi,
                lon_center=long_center_roi,
                width_px=width_roi,
                height_px=height_roi,
                gsd=gsd,
            )

        return bounds
    except Exception as e:
        traceback.print_exc()
        raise Exception


class GPSDetectionService:
    """Service for handling GPS-related detection operations."""

    @staticmethod
    def update_detection_gps(detection: Detection, tile: "Tile") -> None:
        """Update detection with GPS information from tile.

        Args:
            detection: Detection object to update
            tile: Tile object containing GPS context
        """

        if tile.tile_gps_loc is None:
            logger.debug(f"No GPS coordinate found in tile: {tile.image_path}")
            return

        if tile.flight_specs is None:
            logger.debug(f"No flight specs found in tile: {tile.image_path}")
            return

        detection.image_gps_loc = tile.tile_gps_loc

        try:
            detection.gps_loc = GPSDetectionService.compute_detection_gps(
                detection, tile
            )
        except Exception as e:
            logger.error(f"Failed to compute GPS location of detection: {e}")
            detection.gps_loc = None

        # Compute geographic footprint using tile's center coordinates as reference
        if tile.tile_gps_loc:
            # Get tile center coordinates
            lat_center, long_center, alt = GPSUtils.to_decimal(tile.tile_gps_loc)

            # Validate that detection bbox is within tile bounds
            x1, y1, x2, y2 = detection.bbox
            if (
                tile.width is not None
                and tile.height is not None
                and tile.gsd is not None
                and x1 >= 0
                and x2 <= tile.width
                and y1 >= 0
                and y2 <= tile.height
            ):
                try:
                    detection.geographic_footprint = create_geographic_footprint(
                        x1=x1,
                        x2=x2,
                        y1=y1,
                        y2=y2,
                        lat_center_roi=lat_center,
                        long_center_roi=long_center,
                        width_roi=tile.width,
                        height_roi=tile.height,
                        gsd=tile.gsd,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create geographic footprint for detection: {e}"
                    )
                    detection.geographic_footprint = None
            else:
                logger.warning(
                    f"Detection bbox {detection.bbox} is outside tile bounds {tile.width}x{tile.height}. geographic_footprint is set toNone"
                )
                detection.geographic_footprint = None

    @staticmethod
    def update_detections_by_type(
        tile: "Tile", detection_type: Literal["predictions", "annotations"]
    ) -> None:
        """Update GPS information for either predictions or annotations in a tile.

        Args:
            tile: Tile object containing detections to update
            detection_type: Type of detections to update ("predictions" or "annotations")
        """
        if not tile.tile_gps_loc or not tile.gsd:
            logger.debug(f"No GPS coordinate found in tile: {tile.image_path}")
            return

        detections = getattr(tile, detection_type, [])
        if len(detections) == 0:
            logger.debug(f"No {detection_type} found in tile: {tile.image_path}")
            return

        updated_count = 0
        for det in detections:
            if not det.is_empty:
                GPSDetectionService.update_detection_gps(det, tile)
                updated_count += 1

        logger.debug(
            f"Updated GPS information for {updated_count} {detection_type} in tile: {tile.image_path}"
        )

    @staticmethod
    def update_all_detections_gps(tile: "Tile") -> None:
        """Update GPS information for all detections in a tile.

        Args:
            tile: Tile object containing detections to update
        """
        if not tile.tile_gps_loc or not tile.gsd:
            logger.debug(f"No GPS coordinate found in tile: {tile.image_path}")
            return

        GPSDetectionService.update_detections_by_type(tile, "predictions")

        GPSDetectionService.update_detections_by_type(tile, "annotations")

    @staticmethod
    def compute_detection_gps(detection: Detection, tile: "Tile") -> Union[str, None]:
        """Compute GPS location for a detection.

        Args:
            detection: Detection object
            tile: Tile object containing GPS context

        Returns:
            GPS location string or None if computation fails
        """
        if tile.tile_gps_loc is None:
            logger.info(f"No GPS coordinate found in tile: {tile.image_path}")
            return None

        if tile.gsd is None:
            logger.info(f"No GSD found in tile: {tile.image_path}")
            return None

        image = tile.load_image_data()
        assert isinstance(image, Image.Image), "Provide PIL Image"

        # compute detection
        W, H = image.size

        lat_center, lon_center, alt = GPSUtils.to_decimal(tile.tile_gps_loc)
        alt = alt * 1e-3  # converting to km

        px_lat, px_long = get_pixel_gps_coordinates(
            x=detection.x_center,
            y=detection.y_center,
            lat_center=lat_center,
            lon_center=lon_center,
            W=W,
            H=H,
            gsd=tile.gsd,
        )

        gps_loc = str(geopy.Point(latitude=px_lat, longitude=px_long, altitude=alt))

        return gps_loc

    @staticmethod
    def validate_gps_data(tile: "Tile") -> List[str]:
        """Validate GPS data in a tile.

        Args:
            tile: Tile object to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if not tile.tile_gps_loc:
            errors.append("No GPS location found in tile")

        if tile.flight_specs is None:
            errors.append("No flight specs provided")

        if tile.gsd is None:
            errors.append("No GSD (ground sample distance) calculated")

        return errors
