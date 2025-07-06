"""
Performance optimization utilities for Phase 2 features.
"""

import gc
import logging
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .drone_image import DroneImage
from .flight_analyzer import FlightEfficiency, FlightPath
from .geographic_merger import GeographicDataset

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Performance optimization utilities for Phase 2 analysis."""

    def __init__(
        self, cache_dir: Optional[str] = None, max_workers: Optional[int] = None
    ):
        """Initialize the performance optimizer.

        Args:
            cache_dir (Optional[str]): Directory for caching results
            max_workers (Optional[int]): Maximum number of worker processes
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine optimal number of workers
        if max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
        else:
            self.max_workers = max_workers

        # Cache for expensive computations
        self._flight_path_cache: Optional[FlightPath] = None
        self._efficiency_cache: Optional[FlightEfficiency] = None
        self._geographic_dataset_cache: Optional[GeographicDataset] = None

        logger.info(f"PerformanceOptimizer initialized with {self.max_workers} workers")

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self._flight_path_cache = None
        self._efficiency_cache = None
        self._geographic_dataset_cache = None
        gc.collect()
        logger.info("All caches cleared")

    def cache_flight_path(self, flight_path: FlightPath, campaign_id: str) -> None:
        """Cache flight path analysis results.

        Args:
            flight_path (FlightPath): Flight path to cache
            campaign_id (str): Campaign identifier
        """
        cache_file = self.cache_dir / f"flight_path_{campaign_id}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(flight_path, f)
            logger.info(f"Flight path cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache flight path: {e}")

    def load_cached_flight_path(self, campaign_id: str) -> Optional[FlightPath]:
        """Load cached flight path analysis results.

        Args:
            campaign_id (str): Campaign identifier

        Returns:
            Optional[FlightPath]: Cached flight path or None if not found
        """
        cache_file = self.cache_dir / f"flight_path_{campaign_id}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    flight_path = pickle.load(f)
                logger.info(f"Loaded cached flight path from {cache_file}")
                return flight_path
            except Exception as e:
                logger.warning(f"Failed to load cached flight path: {e}")

        return None

    def cache_geographic_dataset(
        self, dataset: GeographicDataset, campaign_id: str
    ) -> None:
        """Cache geographic dataset results.

        Args:
            dataset (GeographicDataset): Dataset to cache
            campaign_id (str): Campaign identifier
        """
        cache_file = self.cache_dir / f"geographic_dataset_{campaign_id}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dataset, f)
            logger.info(f"Geographic dataset cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache geographic dataset: {e}")

    def load_cached_geographic_dataset(
        self, campaign_id: str
    ) -> Optional[GeographicDataset]:
        """Load cached geographic dataset results.

        Args:
            campaign_id (str): Campaign identifier

        Returns:
            Optional[GeographicDataset]: Cached dataset or None if not found
        """
        cache_file = self.cache_dir / f"geographic_dataset_{campaign_id}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    dataset = pickle.load(f)
                logger.info(f"Loaded cached geographic dataset from {cache_file}")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cached geographic dataset: {e}")

        return None

    def parallel_flight_path_analysis(
        self, drone_images: List[DroneImage], flight_analyzer
    ) -> FlightPath:
        """Perform flight path analysis using parallel processing.

        Args:
            drone_images (List[DroneImage]): List of drone images
            flight_analyzer: Flight path analyzer instance

        Returns:
            FlightPath: Analyzed flight path
        """
        logger.info(
            f"Starting parallel flight path analysis with {self.max_workers} workers"
        )
        start_time = time.time()

        # Extract GPS data in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit GPS extraction tasks
            future_to_image = {
                executor.submit(self._extract_gps_data, img): img
                for img in drone_images
            }

            gps_data = []
            for future in as_completed(future_to_image):
                try:
                    result = future.result()
                    if result:
                        gps_data.append(result)
                except Exception as e:
                    logger.warning(f"Failed to extract GPS data: {e}")

        # Create flight path from collected data
        if gps_data:
            coordinates = [data["coordinates"] for data in gps_data]
            timestamps = [data["timestamp"] for data in gps_data]
            image_paths = [data["image_path"] for data in gps_data]

            # Calculate metadata
            metadata = flight_analyzer._calculate_flight_metrics(
                coordinates, timestamps
            )

            flight_path = FlightPath(
                coordinates=coordinates,
                timestamps=timestamps,
                image_paths=image_paths,
                metadata=metadata,
            )
        else:
            flight_path = FlightPath([], [], [], {})

        elapsed_time = time.time() - start_time
        logger.info(f"Parallel flight path analysis completed in {elapsed_time:.2f}s")

        return flight_path

    def _extract_gps_data(self, drone_image: DroneImage) -> Optional[Dict[str, Any]]:
        """Extract GPS data from a drone image.

        Args:
            drone_image (DroneImage): Drone image to extract GPS data from

        Returns:
            Optional[Dict[str, Any]]: GPS data or None if not available
        """
        if drone_image.latitude and drone_image.longitude:
            return {
                "coordinates": (
                    drone_image.latitude,
                    drone_image.longitude,
                    drone_image.altitude or 0.0,
                ),
                "timestamp": drone_image.date or time.time(),
                "image_path": drone_image.image_path,
            }
        return None

    def parallel_overlap_detection(
        self,
        drone_images: List[DroneImage],
        flight_analyzer,
        overlap_threshold: float = 0.1,
    ) -> List:
        """Perform overlap detection using parallel processing.

        Args:
            drone_images (List[DroneImage]): List of drone images
            flight_analyzer: Flight path analyzer instance
            overlap_threshold (float): Minimum overlap percentage

        Returns:
            List: List of overlapping regions
        """
        logger.info(
            f"Starting parallel overlap detection with {self.max_workers} workers"
        )
        start_time = time.time()

        # Create image pairs for parallel processing
        image_pairs = []
        for i, img1 in enumerate(drone_images):
            for j, img2 in enumerate(drone_images[i + 1 :], i + 1):
                image_pairs.append((img1, img2))

        overlapping_regions = []

        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(
                    self._check_overlap, pair, flight_analyzer, overlap_threshold
                ): pair
                for pair in image_pairs
            }

            for future in as_completed(future_to_pair):
                try:
                    result = future.result()
                    if result:
                        overlapping_regions.append(result)
                except Exception as e:
                    logger.warning(f"Failed to check overlap: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Parallel overlap detection completed in {elapsed_time:.2f}s")
        logger.info(f"Found {len(overlapping_regions)} overlapping regions")

        return overlapping_regions

    def _check_overlap(
        self,
        image_pair: Tuple[DroneImage, DroneImage],
        flight_analyzer,
        overlap_threshold: float,
    ):
        """Check overlap between two drone images.

        Args:
            image_pair (Tuple[DroneImage, DroneImage]): Pair of drone images
            flight_analyzer: Flight path analyzer instance
            overlap_threshold (float): Minimum overlap percentage

        Returns:
            Optional overlap region or None
        """
        img1, img2 = image_pair

        if not (img1.latitude and img1.longitude and img2.latitude and img2.longitude):
            return None

        # Calculate distance
        distance = flight_analyzer._calculate_distance(
            img1.latitude, img1.longitude, img2.latitude, img2.longitude
        )

        # Estimate overlap
        overlap_info = flight_analyzer._estimate_overlap(img1, img2, distance)

        if overlap_info["overlap_percentage"] >= overlap_threshold:
            from .flight_analyzer import OverlapRegion

            return OverlapRegion(
                center_lat=(img1.latitude + img2.latitude) / 2,
                center_lon=(img1.longitude + img2.longitude) / 2,
                radius_meters=overlap_info["radius_meters"],
                image_paths=[img1.image_path, img2.image_path],
                overlap_area_sqm=overlap_info["overlap_area_sqm"],
                overlap_percentage=overlap_info["overlap_percentage"],
            )

        return None

    def optimize_memory_usage(self, drone_images: List[DroneImage]) -> None:
        """Optimize memory usage by clearing unnecessary data.

        Args:
            drone_images (List[DroneImage]): List of drone images to optimize
        """
        logger.info("Optimizing memory usage")

        # Clear image data that's not needed for analysis
        for drone_image in drone_images:
            if hasattr(drone_image, "image_data") and drone_image.image_data:
                drone_image.image_data = None

            # Clear tile image data
            for tile in drone_image.tiles:
                if hasattr(tile, "image_data") and tile.image_data:
                    tile.image_data = None

        # Force garbage collection
        gc.collect()

        logger.info("Memory optimization completed")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "cpu_count": mp.cpu_count(),
            "max_workers": self.max_workers,
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cache_dir": str(self.cache_dir),
            "cache_files": len(list(self.cache_dir.glob("*.pkl"))),
        }

    def cleanup_cache(self, max_age_days: int = 7) -> None:
        """Clean up old cache files.

        Args:
            max_age_days (int): Maximum age of cache files in days
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        cleaned_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            if current_time - cache_file.stat().st_mtime > max_age_seconds:
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleaned up {cleaned_count} old cache files")

    def export_performance_report(self, output_path: str) -> None:
        """Export performance report to a file.

        Args:
            output_path (str): Path to save the report
        """
        import json

        metrics = self.get_performance_metrics()

        report = {
            "timestamp": time.time(),
            "performance_metrics": metrics,
            "cache_info": {
                "cache_dir": str(self.cache_dir),
                "cache_files": len(list(self.cache_dir.glob("*.pkl"))),
                "cache_size_mb": sum(
                    f.stat().st_size for f in self.cache_dir.glob("*.pkl")
                )
                / 1024
                / 1024,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report exported to: {output_path}")


# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_global_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance.

    Returns:
        PerformanceOptimizer: Global optimizer instance
    """
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def set_global_optimizer(optimizer: PerformanceOptimizer) -> None:
    """Set the global performance optimizer instance.

    Args:
        optimizer (PerformanceOptimizer): Optimizer instance to set
    """
    global _global_optimizer
    _global_optimizer = optimizer
