"""
Flight path analysis for drone image campaigns.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from geopy.distance import geodesic
from torchmetrics.functional.detection import complete_intersection_over_union
from tqdm import tqdm

from ..data.detection import Detection
from ..data.drone_image import DroneImage

logger = logging.getLogger(__name__)


@dataclass
class FlightPath:
    """Represents a flight path with GPS coordinates and metadata."""

    coordinates: List[Tuple[float, float, float]]  # lat, lon, altitude
    timestamps: List[datetime]
    image_paths: List[DroneImage]
    metadata: Dict[str, Any]


@dataclass
class FlightEfficiency:
    """Flight efficiency metrics."""

    total_distance_km: float
    total_area_covered_sqkm: float
    coverage_efficiency: float  # area_covered / distance_flown
    overlap_percentage: float
    average_altitude_m: float
    image_density_per_sqkm: float


# TODO
class FlightPathAnalyzer:
    """Analyzes flight paths and calculates efficiency metrics."""

    def __init__(self):
        """Initialize the flight path analyzer."""
        pass

    def analyze_flight_path(self, drone_images: List[DroneImage]) -> FlightPath:
        """Analyze the flight path from DroneImage GPS data.

        Args:
            drone_images (List[DroneImage]): List of drone images with GPS data

        Returns:
            FlightPath: Analyzed flight path
        """
        # Extract GPS coordinates and metadata
        coordinates = []
        timestamps = []
        image_paths = []

        for drone_image in drone_images:
            if drone_image.latitude and drone_image.longitude:
                coordinates.append(
                    (
                        drone_image.latitude,
                        drone_image.longitude,
                        drone_image.altitude or 0.0,
                    )
                )
                timestamps.append(
                    datetime.now()
                )  # Placeholder - should extract from EXIF
                image_paths.append(drone_image.image_path)

        if not coordinates:
            logger.warning("No GPS coordinates found in drone images")
            return FlightPath([], [], [], {})

        # Sort by timestamp if available
        if timestamps:
            sorted_data = sorted(zip(coordinates, timestamps, image_paths))
            coordinates, timestamps, image_paths = zip(*sorted_data)
            # Convert back to lists
            coordinates = list(coordinates)
            timestamps = list(timestamps)
            image_paths = list(image_paths)

        # Calculate flight metrics
        metadata = self._calculate_flight_metrics(coordinates, timestamps)

        return FlightPath(
            coordinates=list(coordinates),
            timestamps=list(timestamps),
            image_paths=list(image_paths),
            metadata=metadata,
        )

    def calculate_flight_efficiency(
        self, flight_path: FlightPath, drone_images: List[DroneImage]
    ) -> FlightEfficiency:
        """Calculate flight efficiency metrics.

        Args:
            flight_path (FlightPath): Analyzed flight path
            drone_images (List[DroneImage]): List of drone images

        Returns:
            FlightEfficiency: Flight efficiency metrics
        """
        if not flight_path.coordinates:
            return FlightEfficiency(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Calculate total distance
        total_distance_km = self._calculate_total_distance(flight_path.coordinates)

        # Calculate area covered
        total_area_covered_sqkm = self._calculate_area_covered(drone_images)

        # Calculate coverage efficiency
        coverage_efficiency = (
            total_area_covered_sqkm / total_distance_km
            if total_distance_km > 0
            else 0.0
        )

        # Calculate overlap percentage
        overlap_percentage = self._calculate_overlap_percentage(drone_images)

        # Calculate flight duration
        flight_duration_hours = self._calculate_flight_duration(flight_path.timestamps)

        # Calculate average altitude
        altitudes = [coord[2] for coord in flight_path.coordinates if coord[2] > 0]
        average_altitude_m = sum(altitudes) / len(altitudes) if altitudes else 0.0

        # Calculate image density
        image_density_per_sqkm = (
            len(drone_images) / total_area_covered_sqkm
            if total_area_covered_sqkm > 0
            else 0.0
        )

        return FlightEfficiency(
            total_distance_km=total_distance_km,
            total_area_covered_sqkm=total_area_covered_sqkm,
            coverage_efficiency=coverage_efficiency,
            overlap_percentage=overlap_percentage,
            average_altitude_m=average_altitude_m,
            image_density_per_sqkm=image_density_per_sqkm,
        )

    def _calculate_flight_metrics(
        self, coordinates: List[Tuple], timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate flight path metrics."""
        if len(coordinates) < 2:
            return {}

        # Calculate distances between consecutive points
        distances = []
        for i in range(len(coordinates) - 1):
            dist = self._calculate_distance(
                coordinates[i][0],
                coordinates[i][1],
                coordinates[i + 1][0],
                coordinates[i + 1][1],
            )
            distances.append(dist)

        # Calculate flight statistics
        total_distance = sum(distances)
        avg_distance = total_distance / len(distances) if distances else 0.0
        max_distance = max(distances) if distances else 0.0
        min_distance = min(distances) if distances else 0.0

        # Calculate altitude statistics
        altitudes = [coord[2] for coord in coordinates if coord[2] > 0]
        avg_altitude = sum(altitudes) / len(altitudes) if altitudes else 0.0
        max_altitude = max(altitudes) if altitudes else 0.0
        min_altitude = min(altitudes) if altitudes else 0.0

        return {
            "total_distance_km": total_distance,
            "average_distance_km": avg_distance,
            "max_distance_km": max_distance,
            "min_distance_km": min_distance,
            "average_altitude_m": avg_altitude,
            "max_altitude_m": max_altitude,
            "min_altitude_m": min_altitude,
            "num_waypoints": len(coordinates),
        }

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two GPS coordinates in kilometers."""
        return geodesic((lat1, lon1), (lat2, lon2)).km

    def _estimate_overlap(
        self,
        img1: DroneImage,
        img2: DroneImage,
    ) -> float:
        """Estimate overlap between two images based on distance and size."""
        img1_footprint = img1.geographic_footprint
        img2_footprint = img2.geographic_footprint

        if img1_footprint is None or img2_footprint is None:
            return 0.0

        overlap_ratio = img1_footprint.overlap_ratio(img2_footprint)

        return overlap_ratio

    def _calculate_total_distance(self, coordinates: List[Tuple]) -> float:
        """Calculate total distance of flight path."""
        if len(coordinates) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(coordinates) - 1):
            total_distance += self._calculate_distance(
                coordinates[i][0],
                coordinates[i][1],
                coordinates[i + 1][0],
                coordinates[i + 1][1],
            )

        return total_distance

    def _calculate_area_covered(self, drone_images: List[DroneImage]) -> float:
        """Calculate total area covered by drone images."""

        for drone_image in drone_images:
            geographic_footprint = drone_image.geographic_footprint
            if geographic_footprint is not None:
                total_area += geographic_footprint.area
            else:
                logger.warning(
                    f"No geographic footprint found for image: {drone_image.image_path}"
                )

        return total_area

    def _calculate_overlap_percentage(self, drone_images: List[DroneImage]) -> float:
        """Calculate overall overlap percentage."""
        if len(drone_images) < 2:
            return 0.0

        # Count overlapping pairs
        overlapping_pairs = 0
        total_pairs = len(drone_images) * (len(drone_images) - 1) / 2

        for i, img1 in enumerate(drone_images):
            for j, img2 in enumerate(drone_images[i + 1 :], i + 1):
                if (
                    img1.latitude
                    and img1.longitude
                    and img2.latitude
                    and img2.longitude
                ):
                    distance = self._calculate_distance(
                        img1.latitude, img1.longitude, img2.latitude, img2.longitude
                    )
                    if distance < 0.2:  # 200m threshold for overlap
                        overlapping_pairs += 1

        return overlapping_pairs / total_pairs if total_pairs > 0 else 0.0

    def _calculate_flight_duration(self, timestamps: List[datetime]) -> float:
        """Calculate flight duration in hours."""
        if len(timestamps) < 2:
            return 0.0

        # Sort timestamps
        sorted_timestamps = sorted(timestamps)

        # Calculate duration
        duration = sorted_timestamps[-1] - sorted_timestamps[0]
        return duration.total_seconds() / 3600.0  # Convert to hours
