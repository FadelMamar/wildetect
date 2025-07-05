"""
GPS service for handling GPS-related detection operations.
"""

from typing import Optional, List,Union
import logging
import geopy
import numpy as np
from .detection import Detection
from PIL import Image
from ..gps.gps_utils import GPSUtils,get_pixel_gps_coordinates
from ..gps.geographic_bounds import GeographicBounds

logger = logging.getLogger(__name__)

def create_geographic_footprint(
        x1:float,x2:float,
        y1:float,y2:float,
        lat_center_roi:float,long_center_roi:float,
        width_roi:int,height_roi:int,
        gsd: float
    ) -> Optional[GeographicBounds]:
    
    try:
        # Calculate bounding box corners in GPS coordinates
        xs = np.array([x1, x2])
        ys = np.array([y1, y2])
        
        xs_utm, ys_utm = get_pixel_gps_coordinates(
            x=xs, y=ys,
            lat_center=lat_center_roi,
            lon_center=long_center_roi,
            W=width_roi, H=height_roi,
            gsd=gsd,
            return_as_utm=True
        )        
    
        return GeographicBounds(
        north=max(ys_utm),
        south=min(ys_utm),
        east=max(xs_utm),
        west=min(xs_utm),
        )
    except Exception as e:
        logger.error(f"Failed to create geographic footprint: {e}")
        return None

class GPSDetectionService:
    """Service for handling GPS-related detection operations."""
    
    @staticmethod
    def update_detection_gps(detection: Detection, tile: 'Tile') -> None:
        """Update detection with GPS information from tile.
        
        Args:
            detection: Detection object to update
            tile: Tile object containing GPS context
        """
        if not tile.tile_gps_loc or not tile.flight_specs:
            logger.info(f"No GPS coordinate found in tile: {tile.image_path}")
            return
        
        detection.image_gps_loc = tile.tile_gps_loc
        
        try:
            detection.gps_loc = GPSDetectionService.compute_detection_gps(
                detection, tile
            )
        except Exception as e:
            logger.error(f"Failed to compute GPS location of detection: {e}")
            detection.gps_loc = None
        
        if detection.gps_loc and tile.gsd and tile.width and tile.height:
            lat,long,alt = GPSUtils.to_decimal(tile.tile_gps_loc)
            detection.geographic_footprint = create_geographic_footprint(
                x1=detection.bbox[0],
                x2=detection.bbox[2],
                y1=detection.bbox[1],
                y2=detection.bbox[3],
                lat_center_roi=lat,
                long_center_roi=long,
                width_roi=tile.width,
                height_roi=tile.height,
                gsd=tile.gsd
            )

    @staticmethod
    def update_all_detections_gps(tile: 'Tile') -> None:
        """Update GPS information for all detections in a tile.
        
        Args:
            tile: Tile object containing detections to update
        """
        if not tile.tile_gps_loc or not tile.flight_specs:
            logger.info(f"No GPS coordinate found in tile: {tile.image_path}")
            return
        
        # Update predictions
        if tile.predictions:
            for det in tile.predictions:
                if not det.is_empty:
                    GPSDetectionService.update_detection_gps(det, tile)
        
        # Update annotations
        if tile.annotations:
            for det in tile.annotations:
                if not det.is_empty:
                    GPSDetectionService.update_detection_gps(det, tile)
    
    @staticmethod
    def compute_detection_gps(detection: Detection, tile: 'Tile') -> Union[str,None]:
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
    def validate_gps_data(tile: 'Tile') -> List[str]:
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