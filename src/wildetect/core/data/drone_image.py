"""
Drone image representation with tiles and geographic footprint.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import uuid
import logging
from pathlib import Path

from PIL import Image
import numpy as np

from .tile import Tile
from .detection import Detection
from ..gps.geographic_bounds import GeographicBounds
from ..config import FlightSpecs

logger = logging.getLogger(__name__)


@dataclass
class DroneImage:
    """Represents a drone image with tiles and geographic footprint."""
    
    # Core image information
    image_path: str
    image_data: Optional[Image.Image] = None
    
    # Unique identifier
    id: Optional[str] = None
    
    # Image dimensions
    width: Optional[int] = None
    height: Optional[int] = None
    
    # GPS and geographic information
    gps_loc: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    geographic_footprint: Optional[GeographicBounds] = None
    
    # Flight specifications
    flight_specs: Optional[FlightSpecs] = None
    gsd: Optional[float] = None  # cm/px
    
    # Tiles and offsets
    tiles: List[Tile] = field(default_factory=list)
    tile_offsets: List[Tuple[int, int]] = field(default_factory=list)
    
    # Metadata
    date: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize the drone image."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        
        # Load image dimensions
        if self.image_data is None and self.image_path:
            try:
                with Image.open(self.image_path) as img:
                    self.width, self.height = img.size
                    # Try to extract date from EXIF
                    try:
                        self.date = img._getexif()[36867] if img._getexif() else None
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Could not load image dimensions from {self.image_path}: {e}")
        elif self.image_data:
            self.width, self.height = self.image_data.size
        
        # Extract GPS coordinates
        self._extract_gps_coords()
        
        # Calculate geographic footprint if GPS data is available
        if self.latitude and self.longitude and self.gsd:
            self._calculate_geographic_footprint()
    
    def _extract_gps_coords(self) -> None:
        """Extract GPS coordinates from the image."""
        try:
            from ..gps.gps_utils import GPSUtils
            import geopy
            
            coords = GPSUtils.get_gps_coord(
                file_name=self.image_path,
                image=self.image_data,
                return_as_decimal=True,
            )
            
            if coords is not None:
                self.latitude, self.longitude, self.altitude = coords[0]
                self.gps_loc = str(
                    geopy.Point(self.latitude, self.longitude, self.altitude / 1e3)
                )
            else:
                self.latitude = self.longitude = self.altitude = None
                logger.debug(f"Failed to extract GPS coordinates from {self.image_path}")
                
        except ImportError:
            logger.warning("GPS utilities not available, skipping GPS extraction")
        except Exception as e:
            logger.warning(f"Error extracting GPS coordinates: {e}")
            self.latitude = self.longitude = self.altitude = None
    
    def _calculate_geographic_footprint(self) -> None:
        """Calculate geographic footprint based on GPS and GSD."""
        try:
            from ..gps.gps_service import create_geographic_footprint
            
            if not all([self.latitude, self.longitude, self.gsd, self.width, self.height]):
                logger.debug("Missing required data for geographic footprint calculation")
                return
            
            self.geographic_footprint = create_geographic_footprint(
                x1=0, x2=self.width,
                y1=0, y2=self.height,
                lat_center_roi=self.latitude,
                long_center_roi=self.longitude,
                width_roi=self.width,
                height_roi=self.height,
                gsd=self.gsd
            )
            
        except ImportError:
            logger.warning("GPS service not available, skipping geographic footprint calculation")
        except Exception as e:
            logger.warning(f"Error calculating geographic footprint: {e}")
    
    def load_image_data(self) -> Image.Image:
        """Load the full image data."""
        if self.image_data is not None:
            return self.image_data
        else:
            return Image.open(self.image_path)
    
    def add_tile(self, tile: Tile, offset: Tuple[int, int]) -> None:
        """Add a tile with its offset to the drone image.
        
        Args:
            tile (Tile): Tile to add
            offset (Tuple[int, int]): (x, y) offset of the tile
        """
        if not isinstance(tile, Tile):
            raise TypeError(f"Expected Tile object, got {type(tile)}")
        
        if not isinstance(offset, tuple) or len(offset) != 2:
            raise ValueError("Offset must be a tuple of (x, y) coordinates")
        
        # Set the tile's parent image and offset
        tile.parent_image = self.image_path
        tile.set_offsets(offset[0], offset[1])
        
        self.tiles.append(tile)
        self.tile_offsets.append(offset)
        
        logger.debug(f"Added tile {tile.id} at offset {offset}")
    
    def get_tile_at_position(self, x: int, y: int) -> Optional[Tile]:
        """Get tile at a specific position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            Optional[Tile]: Tile at the position, or None if not found
        """
        for tile, (offset_x, offset_y) in zip(self.tiles, self.tile_offsets):
            if (offset_x <= x < offset_x + tile.width and 
                offset_y <= y < offset_y + tile.height):
                return tile
        return None
    
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
            tile_x2, tile_y2 = offset_x + tile.width, offset_y + tile.height
            
            # Check for overlap
            if not (tile_x2 < x1 or tile_x1 > x2 or tile_y2 < y1 or tile_y1 > y2):
                overlapping_tiles.append(tile)
        
        return overlapping_tiles
    
    def get_all_detections(self) -> List[Detection]:
        """Get all detections from all tiles.
        
        Returns:
            List[Detection]: All detections from all tiles
        """
        all_detections = []
        
        for tile in self.tiles:
            if tile.predictions:
                all_detections.extend(tile.predictions)
            if tile.annotations:
                all_detections.extend(tile.annotations)
        
        return all_detections
    
    def get_detections_in_region(self, x1: int, y1: int, x2: int, y2: int) -> List[Detection]:
        """Get all detections in a specific region.
        
        Args:
            x1, y1 (int): Top-left corner of region
            x2, y2 (int): Bottom-right corner of region
            
        Returns:
            List[Detection]: Detections in the region
        """
        region_detections = []
        overlapping_tiles = self.get_tiles_in_region(x1, y1, x2, y2)
        
        for tile in overlapping_tiles:
            for detection in tile.predictions or []:
                # Check if detection is in region
                if (x1 <= detection.x_center <= x2 and 
                    y1 <= detection.y_center <= y2):
                    region_detections.append(detection)
            
            for detection in tile.annotations or []:
                # Check if detection is in region
                if (x1 <= detection.x_center <= x2 and 
                    y1 <= detection.y_center <= y2):
                    region_detections.append(detection)
        
        return region_detections
    
    def merge_detections(self) -> List[Detection]:
        """Merge detections from all tiles, handling overlaps.
        
        Returns:
            List[Detection]: Merged detections
        """
        all_detections = self.get_all_detections()
        
        # Simple deduplication based on position and class
        merged = []
        for detection in all_detections:
            # Check if similar detection already exists
            is_duplicate = False
            for existing in merged:
                if (detection.class_name == existing.class_name and
                    abs(detection.x_center - existing.x_center) < 50 and
                    abs(detection.y_center - existing.y_center) < 50):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(detection)
        
        return merged
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the drone image.
        
        Returns:
            Dict[str, Any]: Statistics about the image
        """
        all_detections = self.get_all_detections()
        
        # Count detections by class
        class_counts = {}
        for detection in all_detections:
            class_name = detection.class_name or "unknown"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'image_id': self.id,
            'image_path': self.image_path,
            'width': self.width,
            'height': self.height,
            'num_tiles': len(self.tiles),
            'total_detections': len(all_detections),
            'class_counts': class_counts,
            'has_gps': self.latitude is not None and self.longitude is not None,
            'has_geographic_footprint': self.geographic_footprint is not None,
            'gsd': self.gsd,
            'date': self.date
        }
    
    def draw_detections(self, 
                       image: Optional[np.ndarray] = None,
                       predictions: bool = True,
                       annotations: bool = True,
                       prediction_color: Tuple[int, int, int] = (0, 255, 0),
                       annotation_color: Tuple[int, int, int] = (255, 0, 0),
                       line_thickness: int = 2,
                       font_scale: float = 0.5,
                       show_confidence: bool = True,
                       save_path: Optional[str] = None) -> np.ndarray:
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
        all_detections = self.get_all_detections()
        
        for detection in all_detections:
            if detection.is_empty:
                continue
            
            # Determine color based on detection type
            if detection in [d for tile in self.tiles for d in (tile.predictions or [])]:
                color = prediction_color
            else:
                color = annotation_color
            
            # Get absolute coordinates
            bbox = detection.bbox
            class_name = detection.class_name or "unknown"
            confidence = detection.confidence
            
            # Draw bounding box
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         color, line_thickness)
            
            # Draw label
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Calculate text position
            text_x = bbox[0]
            text_y = bbox[1] - 10 if bbox[1] - 10 > 0 else bbox[1] + 20
            
            # Draw text background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                          font_scale, line_thickness)
            cv2.rectangle(result, (text_x, text_y - text_height - 5), 
                         (text_x + text_width, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(result, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness)
        
        # Save image if path is provided
        if save_path:
            cv2.imwrite(save_path, result)
            logger.info(f"Image with detections saved to: {save_path}")
        
        return result
    
    @classmethod
    def from_image_path(cls, image_path: str, **kwargs) -> 'DroneImage':
        """Create drone image from image path."""
        return cls(image_path=image_path, **kwargs)
    
    @classmethod
    def from_image_data(cls, image_data: Image.Image, **kwargs) -> 'DroneImage':
        """Create drone image from image data."""
        return cls(image_data=image_data, **kwargs) 