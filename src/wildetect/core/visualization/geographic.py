"""
Geographic bounds visualization for drone images using Folium maps.

This module provides functionality to visualize the geographic footprints
of DroneImage instances on interactive maps, including overlap analysis
and coverage statistics.
"""

import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import folium
import numpy as np
from folium import plugins
from pyproj import Transformer

from ..data.drone_image import DroneImage
from ..flight.geographic_merger import GPSOverlapStrategy
from ..gps.geographic_bounds import GeographicBounds

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for geographic bounds visualization."""

    # Map settings
    map_center: Optional[Tuple[float, float]] = None
    zoom_start: int = 13
    tiles: str = "OpenStreetMap"

    # Visualization colors
    image_bounds_color: str = "blue"
    image_center_color: str = "red"
    overlap_color: str = "orange"
    detection_color: str = "green"

    # Display settings
    show_image_centers: bool = True
    show_bounding_boxes: bool = True
    use_polygons: bool = True  # Use polygons instead of rectangles for more accuracy
    show_detections: bool = False
    show_overlaps: bool = True
    show_statistics: bool = True

    # Popup settings
    popup_max_width: int = 300
    show_image_path: bool = True
    show_detection_count: bool = True
    show_gps_info: bool = True


class GeographicVisualizer:
    """Visualize geographic bounds of drone images on interactive maps."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the geographic visualizer.

        Args:
            config: Configuration for visualization settings
        """
        self.config = config or VisualizationConfig()
        self.transformer = Transformer.from_crs(
            "EPSG:32632", "EPSG:4326", always_xy=True
        )
        self.overlap_strategy = GPSOverlapStrategy()

    def _transform_utm_to_wgs84(
        self, easting: float, northing: float
    ) -> Optional[Tuple[float, float]]:
        """Transform UTM coordinates to WGS84 lat/lon.

        Args:
            easting: UTM easting coordinate
            northing: UTM northing coordinate

        Returns:
            Tuple of (longitude, latitude) in WGS84, or None if transformation fails
        """
        try:
            lon, lat = self.transformer.transform(easting, northing)
            return lon, lat
        except Exception as e:
            logger.warning(f"Failed to transform coordinates: {e}")
            return None

    def _extract_geographic_data(
        self, drone_images: List[DroneImage]
    ) -> Dict[str, Any]:
        """Extract geographic data from drone images.

        Args:
            drone_images: List of DroneImage instances

        Returns:
            Dictionary containing extracted geographic data
        """
        geographic_data = {
            "images": [],
            "bounds": [],
            "centers": [],
            "statistics": {
                "total_images": len(drone_images),
                "images_with_gps": 0,
                "images_with_footprints": 0,
                "total_detections": 0,
                "detections_with_gps": 0,
                "coverage_area": 0.0,
                "overlap_count": 0,
            },
        }

        for drone_image in drone_images:
            image_data = {
                "id": drone_image.id,
                "path": drone_image.image_path,
                "width": drone_image.width,
                "height": drone_image.height,
                "detection_count": len(drone_image.predictions),
                "gps_location": drone_image.tile_gps_loc,
                "latitude": drone_image.latitude,
                "longitude": drone_image.longitude,
                "geographic_footprint": drone_image.geographic_footprint,
                "polygon_points": drone_image.geo_polygon_points,
                "gsd": drone_image.gsd,
            }

            # Update statistics
            if drone_image.latitude is not None and drone_image.longitude is not None:
                geographic_data["statistics"]["images_with_gps"] += 1
                geographic_data["centers"].append(
                    (drone_image.longitude, drone_image.latitude)
                )

            if drone_image.geographic_footprint is not None:
                geographic_data["statistics"]["images_with_footprints"] += 1

            geographic_data["statistics"]["total_detections"] += len(
                drone_image.predictions
            )

            # Count detections with GPS
            for detection in drone_image.predictions:
                if detection.geographic_footprint is not None:
                    geographic_data["statistics"]["detections_with_gps"] += 1

            geographic_data["images"].append(image_data)

        return geographic_data

    def _create_popup_content(self, image_data: Dict[str, Any]) -> str:
        """Create HTML popup content for an image.

        Args:
            image_data: Image data dictionary

        Returns:
            HTML string for popup content
        """
        content = f"<div style='max-width: {self.config.popup_max_width}px;'>"

        if self.config.show_image_path:
            content += (
                f"<strong>Image:</strong> {image_data['path'].split('/')[-1]}<br>"
            )

        content += (
            f"<strong>Size:</strong> {image_data['width']}x{image_data['height']}<br>"
        )

        if self.config.show_detection_count:
            content += (
                f"<strong>Detections:</strong> {image_data['detection_count']}<br>"
            )

        if self.config.show_gps_info and image_data["latitude"] is not None:
            content += f"<strong>GPS:</strong> {image_data['latitude']:.6f}, {image_data['longitude']:.6f}<br>"

        if image_data.get("gsd"):
            content += f"<strong>GSD:</strong> {image_data['gsd']:.2f} cm/px<br>"

        if image_data.get("bounds_wgs84"):
            bounds = image_data["bounds_wgs84"]
            content += f"<strong>Coverage:</strong> {bounds['area']:.2f} m²<br>"

        content += "</div>"
        return content

    def _find_overlaps_using_strategy(
        self, drone_images: List[DroneImage]
    ) -> Dict[str, List[str]]:
        """Find overlapping images using GPSOverlapStrategy.

        Args:
            drone_images: List of DroneImage instances

        Returns:
            Dictionary mapping image paths to list of overlapping image paths
        """
        try:
            return self.overlap_strategy.find_overlapping_images(drone_images)
        except Exception as e:
            logger.warning(f"Failed to find overlaps using GPSOverlapStrategy: {e}")
            return {}

    def _create_overlap_visualizations(
        self,
        map_obj: folium.Map,
        drone_images: List[DroneImage],
        overlap_map: Dict[str, List[str]],
    ) -> int:
        """Create overlap visualizations on the map.

        Args:
            map_obj: Folium map object
            drone_images: List of DroneImage instances
            overlap_map: Dictionary mapping image paths to overlapping image paths

        Returns:
            Number of overlap visualizations created
        """
        overlap_count = 0
        image_path_to_image = {img.image_path: img for img in drone_images}

        for image_path, overlapping_paths in overlap_map.items():
            if image_path not in image_path_to_image:
                continue

            main_image = image_path_to_image[image_path]

            for overlap_path in overlapping_paths:
                if overlap_path not in image_path_to_image:
                    continue

                overlap_image = image_path_to_image[overlap_path]

                # Create overlap visualization if both images have geographic footprints
                if (
                    main_image.geographic_footprint is not None
                    and overlap_image.geographic_footprint is not None
                ):
                    # Calculate overlap bounds
                    main_bounds = main_image.geographic_footprint
                    overlap_bounds = overlap_image.geographic_footprint

                    # Find intersection
                    overlap_west = max(main_bounds.west, overlap_bounds.west)
                    overlap_east = min(main_bounds.east, overlap_bounds.east)
                    overlap_south = max(main_bounds.south, overlap_bounds.south)
                    overlap_north = min(main_bounds.north, overlap_bounds.north)

                    if overlap_west < overlap_east and overlap_south < overlap_north:
                        # Calculate overlap area
                        overlap_area = (overlap_east - overlap_west) * (
                            overlap_north - overlap_south
                        )

                        # Create rectangle for overlap
                        folium.Rectangle(
                            bounds=[
                                [overlap_south, overlap_west],
                                [overlap_north, overlap_east],
                            ],
                            color=self.config.overlap_color,
                            weight=3,
                            fill=True,
                            fillOpacity=0.3,
                            popup=f"Overlap Area: {overlap_area:.2f} m²<br>Images: {image_path.split('/')[-1]}, {overlap_path.split('/')[-1]}",
                        ).add_to(map_obj)
                        overlap_count += 1

        return overlap_count

    def create_map(self, drone_images: List[DroneImage]) -> folium.Map:
        """Create a Folium map with geographic bounds visualization.

        Args:
            drone_images: List of DroneImage instances to visualize

        Returns:
            Folium map object
        """
        if not drone_images:
            logger.warning("No drone images provided for visualization")
            return folium.Map()

        # Extract geographic data
        geo_data = self._extract_geographic_data(drone_images)

        # Determine map center
        if self.config.map_center:
            center = self.config.map_center
        elif geo_data["centers"]:
            # Use average of image centers
            lons = [c[0] for c in geo_data["centers"]]
            lats = [c[1] for c in geo_data["centers"]]
            center = (np.mean(lats), np.mean(lons))
        else:
            # Default center (can be adjusted)
            center = (0, 0)

        # Create base map
        map_obj = folium.Map(
            location=[float(center[0]), float(center[1])],
            zoom_start=self.config.zoom_start,
            tiles=self.config.tiles,
        )

        # Add image bounding boxes
        if self.config.show_bounding_boxes:
            bounds_count = 0
            polygon_count = 0
            for i, image_data in enumerate(geo_data["images"]):
                popup_content = self._create_popup_content(image_data)

                # Try to create polygon first (more accurate) if enabled
                polygon_points = image_data["polygon_points"]
                if self.config.use_polygons and polygon_points is not None:
                    # Create polygon
                    folium.Polygon(
                        locations=polygon_points,
                        color=self.config.image_bounds_color,
                        weight=2,
                        fill=True,
                        fillOpacity=0.1,
                        popup=folium.Popup(
                            popup_content, max_width=self.config.popup_max_width
                        ),
                    ).add_to(map_obj)
                    polygon_count += 1
                    logger.debug(f"Added polygon for image {i}")

            logger.info(
                f"Added {bounds_count} rectangles and {polygon_count} polygons to map"
            )

        # Add image centers
        if self.config.show_image_centers:
            for i, image_data in enumerate(geo_data["images"]):
                if (
                    image_data["latitude"] is not None
                    and image_data["longitude"] is not None
                ):
                    popup_content = self._create_popup_content(image_data)

                    folium.CircleMarker(
                        location=[image_data["latitude"], image_data["longitude"]],
                        radius=5,
                        color=self.config.image_center_color,
                        fill=True,
                        popup=folium.Popup(
                            popup_content, max_width=self.config.popup_max_width
                        ),
                    ).add_to(map_obj)

        # Add overlap visualization using GPSOverlapStrategy
        if self.config.show_overlaps:
            overlap_map = self._find_overlaps_using_strategy(drone_images)
            overlap_count = self._create_overlap_visualizations(
                map_obj, drone_images, overlap_map
            )
            geo_data["statistics"]["overlap_count"] = len(overlap_map)
            logger.info(
                f"Found {len(overlap_map)} overlapping image pairs, created {overlap_count} overlap visualizations"
            )

        # Add statistics layer
        if self.config.show_statistics:
            stats = geo_data["statistics"]
            overlap_map = (
                self._find_overlaps_using_strategy(drone_images)
                if self.config.show_overlaps
                else {}
            )
            stats_html = f"""
            <div style="background-color: white; padding: 10px; border-radius: 5px;">
                <h4>Coverage Statistics</h4>
                <p><strong>Total Images:</strong> {stats['total_images']}</p>
                <p><strong>Images with GPS:</strong> {stats['images_with_gps']}</p>
                <p><strong>Images with Footprints:</strong> {stats['images_with_footprints']}</p>
                <p><strong>Total Detections:</strong> {stats['total_detections']}</p>
                <p><strong>Detections with GPS:</strong> {stats['detections_with_gps']}</p>
                <p><strong>Coverage Area:</strong> {stats['coverage_area']:.2f} m²</p>
                <p><strong>Overlapping Image Pairs:</strong> {len(overlap_map)}</p>
            </div>
            """

            folium.Element(stats_html).add_to(map_obj)

        return map_obj

    def save_map(self, drone_images: List[DroneImage], output_path: str) -> None:
        """Create and save a map visualization to file.

        Args:
            drone_images: List of DroneImage instances to visualize
            output_path: Path to save the HTML map file
        """
        map_obj = self.create_map(drone_images)
        map_obj.save(output_path)
        logger.info(f"Map saved to: {output_path}")

    def get_coverage_statistics(self, drone_images: List[DroneImage]) -> Dict[str, Any]:
        """Get detailed coverage statistics for drone images.

        Args:
            drone_images: List of DroneImage instances

        Returns:
            Dictionary containing coverage statistics
        """
        geo_data = self._extract_geographic_data(drone_images)
        overlap_map = self._find_overlaps_using_strategy(drone_images)

        stats = geo_data["statistics"].copy()
        stats["overlap_count"] = len(overlap_map)

        # Calculate overlap statistics
        total_overlap_area = 0.0
        overlap_areas = []

        image_path_to_image = {img.image_path: img for img in drone_images}
        for image_path, overlapping_paths in overlap_map.items():
            if image_path not in image_path_to_image:
                continue

            main_image = image_path_to_image[image_path]

            for overlap_path in overlapping_paths:
                if overlap_path not in image_path_to_image:
                    continue

                overlap_image = image_path_to_image[overlap_path]

                if (
                    main_image.geographic_footprint is not None
                    and overlap_image.geographic_footprint is not None
                ):
                    # Calculate overlap area
                    main_bounds = main_image.geographic_footprint
                    overlap_bounds = overlap_image.geographic_footprint

                    overlap_west = max(main_bounds.west, overlap_bounds.west)
                    overlap_east = min(main_bounds.east, overlap_bounds.east)
                    overlap_south = max(main_bounds.south, overlap_bounds.south)
                    overlap_north = min(main_bounds.north, overlap_bounds.north)

                    if overlap_west < overlap_east and overlap_south < overlap_north:
                        overlap_area = (overlap_east - overlap_west) * (
                            overlap_north - overlap_south
                        )
                        overlap_areas.append(overlap_area)
                        total_overlap_area += overlap_area

        stats["overlap_areas"] = overlap_areas
        stats["total_overlap_area"] = total_overlap_area

        if stats["coverage_area"] > 0:
            stats["average_overlap_area"] = (
                np.mean(overlap_areas) if overlap_areas else 0
            )
            stats["overlap_percentage"] = (
                total_overlap_area / stats["coverage_area"]
            ) * 100
        else:
            stats["average_overlap_area"] = 0
            stats["overlap_percentage"] = 0

        return stats


def visualize_geographic_bounds(
    drone_images: List[DroneImage],
    output_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
) -> folium.Map:
    """Convenience function to visualize geographic bounds.

    Args:
        drone_images: List of DroneImage instances to visualize
        output_path: Optional path to save the HTML map file
        config: Optional visualization configuration

    Returns:
        Folium map object
    """
    visualizer = GeographicVisualizer(config)
    map_obj = visualizer.create_map(drone_images)

    if output_path:
        visualizer.save_map(drone_images, output_path)

    return map_obj
