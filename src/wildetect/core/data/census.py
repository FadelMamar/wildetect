"""
Dataset management for drone image analysis campaigns.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..config import LoaderConfig
from ..data.detection import Detection
from ..flight.flight_analyzer import FlightEfficiency, FlightPath, FlightPathAnalyzer
from ..flight.geographic_merger import GeographicMerger
from .drone_image import DroneImage
from .loader import load_images_as_drone_images
from .utils import get_images_paths

logger = logging.getLogger(__name__)


@dataclass
class CampaignMetadata:
    """Metadata for a drone flight campaign."""

    campaign_id: str
    flight_date: datetime
    pilot_info: Dict
    weather_conditions: Dict
    mission_objectives: List[str]
    target_species: List[str]
    flight_parameters: Dict  # altitude, speed, overlap settings
    equipment_info: Dict  # camera, drone model


@dataclass
class DetectionResults:
    """Results from a detection campaign."""

    total_images: int
    total_detections: int
    detection_by_class: Dict[str, int]
    processing_time: float
    detection_confidence_stats: Dict[str, Any]
    geographic_coverage: Dict[str, Any]
    campaign_id: str
    metadata: Dict[str, Any]


class CensusDataManager:
    """High-level data management for drone flight campaigns."""

    def __init__(
        self,
        campaign_id: str,
        loading_config: LoaderConfig,
        metadata: Optional[Dict] = None,
    ):
        """Initialize CensusData for a campaign.

        Args:
            campaign_id (str): Unique identifier for the campaign
            loading_config (LoaderConfig): Configuration for data loading
            metadata (Optional[Dict]): Campaign metadata
        """
        self.campaign_id = campaign_id
        self.metadata = metadata or {}

        # Core data structures
        self.drone_images: List[DroneImage] = []
        self.image_paths: List[str] = []

        # Processing configuration
        self.loading_config = loading_config

        # Campaign metadata
        self.campaign_metadata: Optional[CampaignMetadata] = None

        # Detection results
        self.detection_results: Optional[DetectionResults] = None

        # Phase 2: Flight analysis and geographic merging
        self.flight_analyzer = FlightPathAnalyzer()
        self.geographic_merger = GeographicMerger()
        self.flight_path: Optional[FlightPath] = None
        self.flight_efficiency: Optional[FlightEfficiency] = None

        logger.info(f"Initialized CensusData for campaign: {campaign_id}")

    def add_images_from_paths(self, image_paths: List[str]) -> None:
        """Add images from a list of file paths.

        Args:
            image_paths (List[str]): List of image file paths
        """
        # Validate paths
        valid_paths = []
        for path in image_paths:
            if not Path(path).exists():
                logger.warning(f"Image path does not exist: {path}")
                continue
            if Path(path).is_file():
                valid_paths.append(path)
            else:
                logger.warning(f"Image path is not a file: {path}")

        self.image_paths.extend(valid_paths)
        logger.info(
            f"Added {len(valid_paths)} valid image paths to campaign {self.campaign_id}"
        )

    def add_images_from_directory(self, directory_path: str) -> None:
        """Add all images from a directory."""
        image_paths = get_images_paths(directory_path)
        image_paths = [str(path) for path in image_paths]
        self.add_images_from_paths(image_paths)
        logger.info(f"Added {len(image_paths)} images from directory: {directory_path}")

    def create_drone_images(
        self,
        gps_coords_loader:Optional[Callable[[str], Tuple[float, float, float]]] = None,
    ) -> None:
        """Create DroneImage instances from the loaded image paths.

        Args:
            tile_size (Optional[int]): Size of tiles to extract
            overlap (Optional[float]): Overlap ratio between tiles
        """
        if not self.image_paths:
            logger.warning(
                "No image paths available for DroneImage creation. Please add images first."
            )
            return

        logger.info(f"Creating DroneImages for {len(self.image_paths)} images...")

        self.drone_images = load_images_as_drone_images(
            image_paths=self.image_paths,
            flight_specs=self.loading_config.flight_specs,
            gps_coords_loader=gps_coords_loader,
        )

        logger.info(f"Successfully created {len(self.drone_images)} DroneImages")

    def analyze_flight_path(self) -> Optional[FlightPath]:
        """Analyze the flight path from drone images.

        Returns:
            Optional[FlightPath]: Analyzed flight path or None if no GPS data
        """
        if not self.drone_images:
            logger.warning("No drone images available for flight path analysis")
            return None

        logger.info(f"Analyzing flight path for {len(self.drone_images)} drone images")

        self.flight_path = self.flight_analyzer.analyze_flight_path(self.drone_images)

        if self.flight_path.coordinates:
            logger.info(
                f"Flight path analysis completed: {len(self.flight_path.coordinates)} waypoints"
            )
        else:
            logger.warning("No GPS coordinates found for flight path analysis")

        return self.flight_path

    def calculate_flight_efficiency(self) -> Optional[FlightEfficiency]:
        """Calculate flight efficiency metrics.

        Returns:
            Optional[FlightEfficiency]: Flight efficiency metrics or None if no flight path
        """
        if not self.flight_path:
            logger.warning("No flight path available. Run analyze_flight_path() first.")
            return None

        logger.info("Calculating flight efficiency metrics")

        self.flight_efficiency = self.flight_analyzer.calculate_flight_efficiency(
            self.flight_path, self.drone_images
        )

        logger.info(f"Flight efficiency calculated:")
        logger.info(
            f"  Total distance: {self.flight_efficiency.total_distance_km:.2f} km"
        )
        logger.info(
            f"  Area covered: {self.flight_efficiency.total_area_covered_sqkm:.2f} sq km"
        )
        logger.info(
            f"  Overlap percentage: {self.flight_efficiency.overlap_percentage:.1%}"
        )

        return self.flight_efficiency

    def merge_detections_geographically(self, iou_threshold: float = 0.8) -> None:
        """Merge detections across overlapping geographic regions."""
        if not self.drone_images:
            logger.warning("No drone images available for geographic merging")
            return None

        merged_drone_images = self.geographic_merger.run(
            self.drone_images, iou_threshold=iou_threshold
        )
        self.drone_images = merged_drone_images
        return None

    def set_detection_results(self, results: DetectionResults) -> None:
        """Set detection results from a detection campaign.

        Args:
            results: Detection results to store
        """
        self.detection_results = results
        logger.info(f"Set detection results for campaign {self.campaign_id}")

    def get_drone_image_by_path(self, image_path: str) -> Optional[DroneImage]:
        """Get a drone image by its path.

        Args:
            image_path: Path to the image

        Returns:
            Optional[DroneImage]: The drone image if found, None otherwise
        """
        for drone_image in self.drone_images:
            if drone_image.image_path == image_path:
                return drone_image
        return None

    def _calculate_geographic_coverage(self) -> Dict[str, Any]:
        """Calculate geographic coverage statistics."""
        coverage = {
            "total_area_covered": 0.0,
            "images_with_gps": 0,
            "geographic_bounds": None,
        }

        gps_images = [
            img for img in self.drone_images if img.latitude and img.longitude
        ]
        coverage["images_with_gps"] = len(gps_images)

        if gps_images:
            # Calculate bounding box - filter out None values
            lats = [img.latitude for img in gps_images if img.latitude is not None]
            lons = [img.longitude for img in gps_images if img.longitude is not None]

            if lats and lons:  # Check that we have valid coordinates
                coverage["geographic_bounds"] = {
                    "min_lat": min(lats),
                    "max_lat": max(lats),
                    "min_lon": min(lons),
                    "max_lon": max(lons),
                }

        return coverage

    def get_campaign_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the campaign.

        Returns:
            Dict[str, Any]: Campaign statistics
        """
        stats = {
            "campaign_id": self.campaign_id,
            "total_images": len(self.drone_images),
            "total_image_paths": len(self.image_paths),
            "tile_configuration": vars(self.loading_config),
            "metadata": self.metadata,
        }

        if self.detection_results:
            stats["detection_results"] = {
                "total_detections": self.detection_results.total_detections,
                "detection_by_class": self.detection_results.detection_by_class,
                "processing_time": self.detection_results.processing_time,
                "confidence_stats": self.detection_results.detection_confidence_stats,
            }

        # Add flight analysis statistics
        if self.flight_path:
            stats["flight_analysis"] = {
                "total_waypoints": len(self.flight_path.coordinates),
                "total_distance_km": self.flight_path.metadata.get(
                    "total_distance_km", 0.0
                ),
                "average_altitude_m": self.flight_path.metadata.get(
                    "average_altitude_m", 0.0
                ),
                "num_images_with_gps": len(
                    [img for img in self.drone_images if img.latitude and img.longitude]
                ),
            }

        # Add flight efficiency statistics
        if self.flight_efficiency:
            stats["flight_efficiency"] = {
                "total_distance_km": self.flight_efficiency.total_distance_km,
                "total_area_covered_sqkm": self.flight_efficiency.total_area_covered_sqkm,
                "overlap_percentage": self.flight_efficiency.overlap_percentage,
                "average_altitude_m": self.flight_efficiency.average_altitude_m,
                "image_density_per_sqkm": self.flight_efficiency.image_density_per_sqkm,
            }

        return stats

    def export_detection_report(self, output_path: str) -> None:
        """Export detection results to a file."""
        if not self.detection_results:
            logger.warning("No detection results to export")
            return

        report = {
            "campaign_id": self.campaign_id,
            "metadata": self.metadata,
            "detection_results": {
                "total_images": self.detection_results.total_images,
                "total_detections": self.detection_results.total_detections,
                "detection_by_class": self.detection_results.detection_by_class,
                "processing_time": self.detection_results.processing_time,
                "confidence_stats": self.detection_results.detection_confidence_stats,
                "geographic_coverage": self.detection_results.geographic_coverage,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        gps_coords = [
            list(detection.gps_as_decimals) + [detection.parent_image]
            for detection in self.get_all_detections()
        ]
        df = pd.DataFrame(
            gps_coords, columns=["latitude", "longitude", "altitude", "image_path"]
        )
        df.to_csv(Path(output_path).with_suffix(".csv"), index=False)

        logger.info(f"Detection report exported to: {output_path}")

    def get_all_detections(
        self,
    ) -> List[Detection]:
        """Get all detections from all DroneImages."""
        all_detections = []
        for drone_image in self.drone_images:
            all_detections.extend(drone_image.get_non_empty_predictions())
        return all_detections
