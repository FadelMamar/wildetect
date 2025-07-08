"""
Geographic bounds visualization for drone images using Folium maps.

This module provides functionality to visualize the geographic footprints
of DroneImage instances on interactive maps, including overlap analysis
and coverage statistics.
"""

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import numpy as np

from ..data.drone_image import DroneImage
from ..flight.geographic_merger import GPSOverlapStrategy

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
    show_image_centers: bool = False
    show_image_bounds: bool = True
    show_statistics: bool = True
    show_detections: bool = True  # Show individual detections on the map

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
        self.overlap_strategy = GPSOverlapStrategy()

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
                drone_image.get_non_empty_predictions()
            )

            # Count detections with GPS
            for detection in drone_image.get_non_empty_predictions():
                if detection.gps_loc is not None:
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
            content += f"<strong>Image:</strong> {Path(image_data['path']).name}<br>"

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

    def _visualize_drone_image_polygon(
        self,
        map_obj: folium.Map,
        drone_image: DroneImage,
        color: Optional[str] = None,
        weight: int = 2,
        fill_opacity: float = 0.1,
        popup_content: Optional[str] = None,
    ) -> bool:
        """Visualize a DroneImage bounds using its polygon.

        Args:
            map_obj: Folium map object to add the visualization to
            drone_image: DroneImage instance to visualize
            color: Color for the polygon (defaults to config image_bounds_color)
            weight: Line weight for the polygon border
            fill_opacity: Opacity for polygon fill
            popup_content: Optional custom popup content

        Returns:
            bool: True if visualization was successful, False otherwise
        """
        if drone_image.geographic_footprint is None:
            logger.warning(
                f"DroneImage {drone_image.image_path} has no geographic footprint"
            )
            return False

        if drone_image.geo_polygon_points is None:
            logger.warning(f"DroneImage {drone_image.image_path} has no polygon points")
            return False

        try:
            # Use provided color or default from config
            polygon_color = color or self.config.image_bounds_color

            # Create popup content if not provided
            if popup_content is None:
                image_data = {
                    "path": drone_image.image_path,
                    "width": drone_image.width,
                    "height": drone_image.height,
                    "detection_count": len(drone_image.predictions),
                    "latitude": drone_image.latitude,
                    "longitude": drone_image.longitude,
                    "gsd": drone_image.gsd,
                }
                popup_content = self._create_popup_content(image_data)

            # Create polygon visualization
            folium.Polygon(
                locations=drone_image.geo_polygon_points,
                color=polygon_color,
                weight=weight,
                fill=True,
                fillOpacity=fill_opacity,
                popup=folium.Popup(
                    popup_content, max_width=self.config.popup_max_width
                ),
            ).add_to(map_obj)

            logger.debug(
                f"Added polygon visualization for image: {drone_image.image_path}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to visualize polygon for {drone_image.image_path}: {e}"
            )
            return False

    def _visualize_detections(
        self,
        map_obj: folium.Map,
        drone_image: DroneImage,
        color: Optional[str] = None,
        radius: int = 5,
        weight: int = 2,
        fill_opacity: float = 0.7,
    ) -> int:
        """Visualize detections for a drone image.

        Args:
            map_obj: Folium map object to add the visualization to
            drone_image: DroneImage instance containing detections
            color: Color for the detection markers (defaults to config detection_color)
            radius: Radius of the detection markers
            weight: Line weight for the marker border
            fill_opacity: Opacity for marker fill

        Returns:
            int: Number of detections visualized
        """
        if not drone_image.get_non_empty_predictions():
            return 0

        detection_color = color or self.config.detection_color
        visualized_count = 0

        for detection in drone_image.get_non_empty_predictions():
            try:
                # Get detection center coordinates
                if detection.gps_loc is None:
                    logger.warning(
                        f"Can't visualize detection. No GPS location for detection {detection}"
                    )
                    continue

                lat, lon, alt = detection.gps_as_decimals

                # Create popup content for detection
                popup_content = self._create_detection_popup_content(
                    detection, drone_image
                )

                # Create marker for detection
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color=detection_color,
                    fill=True,
                    fillColor=detection_color,
                    fillOpacity=fill_opacity,
                    weight=weight,
                    popup=folium.Popup(
                        popup_content, max_width=self.config.popup_max_width
                    ),
                ).add_to(map_obj)

                visualized_count += 1
                logger.debug(f"Added detection marker at {lat}, {lon}")

            except Exception as e:
                logger.warning(f"Failed to visualize detection: {e}")
                continue

        return visualized_count

    def _create_detection_popup_content(
        self, detection, drone_image: DroneImage
    ) -> str:
        """Create HTML popup content for a detection.

        Args:
            detection: Detection object
            drone_image: Parent DroneImage

        Returns:
            HTML string for popup content
        """
        content = f"<div style='max-width: {self.config.popup_max_width}px;'>"

        content += f"<strong>Detection:</strong><br>"
        content += f"<strong>Class:</strong> {detection.class_name}<br>"
        content += f"<strong>Confidence:</strong> {detection.confidence:.3f}<br>"
        content += f"<strong>BBox:</strong> {detection.bbox}<br>"
        content += f"<strong>Area:</strong> {detection.area} px²<br>"

        if detection.gps_loc:
            content += f"<strong>GPS:</strong> {detection.gps_loc}<br>"

        content += f"<strong>Image:</strong> {Path(drone_image.image_path).name}<br>"

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

    def _create_map_center(self, drone_images: List[DroneImage]) -> Tuple[float, float]:
        """Create a map center from the drone images."""
        valid_coordinates = []
        for drone_image in drone_images:
            if drone_image.latitude is not None and drone_image.longitude is not None:
                valid_coordinates.append((drone_image.latitude, drone_image.longitude))

        if valid_coordinates:
            mean_lat = float(np.mean([coord[0] for coord in valid_coordinates]))
            mean_lon = float(np.mean([coord[1] for coord in valid_coordinates]))
            map_center = (mean_lat, mean_lon)
        else:
            map_center = (
                -23.988208339433463,
                31.55495477272327,
            )  # Fallback to Kruger National Park coordinates

        return map_center

    def create_map(
        self, drone_images: List[DroneImage], save_path: Optional[str] = None
    ) -> folium.Map:
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
        else:
            center = self._create_map_center(drone_images)

        # Create base map
        map_obj = folium.Map(
            location=[float(center[0]), float(center[1])],
            zoom_start=self.config.zoom_start,
            tiles=self.config.tiles,
        )

        # Add image
        if self.config.show_image_bounds:
            failed_count = 0
            polygon_count = 0
            for i, drone_image in enumerate(drone_images):
                # Use the new polygon visualization method
                success = self._visualize_drone_image_polygon(
                    map_obj=map_obj,
                    drone_image=drone_image,
                    color=self.config.image_bounds_color,
                    weight=2,
                    fill_opacity=0.1,
                )
                if success:
                    polygon_count += 1
                else:
                    failed_count += 1

            logger.debug(
                f"Added {polygon_count} image polygons"
                f"and {failed_count} failed to add polygons to map"
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
                        radius=3,
                        color=self.config.image_center_color,
                        fill=False,
                        popup=folium.Popup(
                            popup_content, max_width=self.config.popup_max_width
                        ),
                    ).add_to(map_obj)

        # Add detections
        if self.config.show_detections:
            total_detections_visualized = 0
            total_detections = 0
            for drone_image in drone_images:
                eligible_detections = [
                    det
                    for det in drone_image.get_all_predictions()
                    if not det.is_empty and det.gps_loc is not None
                ]
                detections_count = self._visualize_detections(
                    map_obj=map_obj,
                    drone_image=drone_image,
                    color=self.config.detection_color,
                    radius=3,
                    weight=2,
                    fill_opacity=0.7,
                )
                total_detections += len(eligible_detections)
                total_detections_visualized += detections_count
                logger.debug(
                    f"Visualized {detections_count} detections for {drone_image.image_path}"
                )

            logger.info(
                f"Total detections visualized: {total_detections_visualized}/{total_detections}"
            )

        # Add statistics layer
        if self.config.show_statistics:
            stats = geo_data["statistics"]
            stats_html = f"""
            <div style="background-color: white; padding: 10px; border-radius: 5px;">
                <h4>Coverage Statistics</h4>
                <p><strong>Total Images:</strong> {stats['total_images']}</p>
                <p><strong>Images with GPS:</strong> {stats['images_with_gps']}</p>
                <p><strong>Images with Footprints:</strong> {stats['images_with_footprints']}</p>
                <p><strong>Total Detections:</strong> {stats['total_detections']}</p>
                <p><strong>Detections with GPS:</strong> {stats['detections_with_gps']}</p>
                <p><strong>Coverage Area:</strong> {stats['coverage_area']:.2f} m²</p>
            </div>
            """

            folium.Element(stats_html).add_to(map_obj)

        if save_path:
            self.save_map(map_obj, save_path)

        return map_obj

    def save_map(self, map_obj: folium.Map, output_path: str) -> None:
        """Create and save a map visualization to file.

        Args:
            drone_images: List of DroneImage instances to visualize
            output_path: Path to save the HTML map file
        """
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
                    overlap_area = main_image.geographic_footprint.overlap_area(
                        overlap_image.geographic_footprint
                    )
                    overlap_areas.append(overlap_area)
                    total_overlap_area += overlap_area

        stats["overlap_areas"] = overlap_areas
        stats["total_overlap_area"] = total_overlap_area
        stats["average_overlap_area"] = sum(overlap_areas) / max(len(overlap_areas), 1)

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
    map_obj = visualizer.create_map(drone_images, output_path)

    return map_obj
