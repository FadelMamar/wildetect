"""
Campaign Manager for orchestrating complete wildlife detection campaigns.

This module provides a high-level interface that integrates data management,
detection processing, flight analysis, and reporting into a unified pipeline.
"""

import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import ROOT, LoaderConfig, PredictionConfig
from .data.census import CensusDataManager, DetectionResults
from .data.detection import Detection
from .data.drone_image import DroneImage
from .detection_pipeline import DetectionPipeline, MultiThreadedDetectionPipeline
from .flight.flight_analyzer import FlightEfficiency, FlightPath
from .visualization.fiftyone_manager import FiftyOneManager
from .visualization.geographic import GeographicVisualizer, VisualizationConfig

logger = logging.getLogger(__name__)


@dataclass
class CampaignConfig:
    """Configuration for a complete campaign."""

    # Campaign identification
    campaign_id: str

    # Data loading configuration
    loader_config: LoaderConfig

    # Detection configuration
    prediction_config: PredictionConfig

    detection_merging_threshold: float = 0.2

    # Optional configurations
    metadata: Optional[Dict[str, Any]] = None
    visualization_config: Optional[VisualizationConfig] = None
    fiftyone_dataset_name: Optional[str] = None


class CampaignManager:
    """High-level campaign manager that orchestrates the complete detection pipeline."""

    def __init__(self, config: CampaignConfig):
        """Initialize the campaign manager.

        Args:
            config: Campaign configuration
        """
        self.config = config
        self.campaign_id = config.campaign_id

        # Initialize components
        self.census_manager = CensusDataManager(
            campaign_id=config.campaign_id,
            loading_config=config.loader_config,
            metadata=config.metadata or {},
        )

        # Choose pipeline type based on configuration
        if config.prediction_config.pipeline_type == "multi":
            self.detection_pipeline = MultiThreadedDetectionPipeline(
                config=config.prediction_config,
                loader_config=config.loader_config,
                queue_size=config.prediction_config.queue_size,
            )
        else:
            self.detection_pipeline = DetectionPipeline(
                config=config.prediction_config, loader_config=config.loader_config
            )

        # Optional components
        self.fiftyone_manager = None
        if config.fiftyone_dataset_name:
            self.fiftyone_manager = FiftyOneManager(
                config.fiftyone_dataset_name, persistent=True
            )

        self.geographic_visualizer = GeographicVisualizer(
            config=config.visualization_config
        )

        logger.info(f"Initialized CampaignManager for campaign: {self.campaign_id}")

    def get_drone_images(
        self, as_dict: bool = False
    ) -> Union[List[DroneImage], Dict[str, Dict[str, Any]]]:
        """Get the drone images for the campaign."""
        drone_images = self.census_manager.drone_images
        if as_dict:
            return {img.image_path: img.to_dict() for img in drone_images}
        else:
            return drone_images

    def set_drone_images(self, drone_images: List[DroneImage]) -> None:
        """Set the drone images for the campaign."""
        self.census_manager.drone_images = drone_images

    def add_images_from_paths(self, image_paths: List[str]) -> None:
        """Add images from file paths.

        Args:
            image_paths: List of image file paths
        """
        for image_path in image_paths:
            assert Path(image_path).is_file(), "image_paths must be file paths"

        self.census_manager.add_images_from_paths(image_paths)

    def add_images_from_directory(self, directory_path: str) -> None:
        """Add all images from a directory.

        Args:
            directory_path: Path to directory containing images
        """
        assert Path(directory_path).is_dir(), "directory_path must be a directory"
        self.census_manager.add_images_from_directory(directory_path)

    def prepare_data(
        self,
    ) -> None:
        """Prepare data for detection by creating DroneImage instances.

        Args:
            tile_size: Optional tile size override
            overlap: Optional overlap ratio override
        """
        self.census_manager.create_drone_images()
        logger.info(
            f"Prepared {len(self.census_manager.drone_images)} drone images for detection"
        )

    def run_detection(
        self, save_results: bool = True, output_dir: Optional[str] = None
    ) -> DetectionResults:
        """Run detection on all images in the campaign.

        Args:
            save_results: Whether to save results to files
            output_dir: Optional output directory for results

        Returns:
            DetectionResults: Results from the detection campaign
        """

        images = self.get_drone_images(as_dict=False)
        if not images:
            raise ValueError("No drone images available. Call prepare_data() first.")

        logger.info(f"Starting detection campaign for {len(images)} images")
        start_time = time.time()

        # Run detection using the pipeline
        image_paths = [img.image_path for img in images]
        processed_drone_images = self.detection_pipeline.run_detection(
            image_paths=image_paths,
            save_path=output_dir + "/detection_results.json"
            if save_results and output_dir
            else None,
        )

        # Update census manager with processed images
        self.set_drone_images(processed_drone_images)

        # Calculate detection statistics
        total_detections = 0
        detection_by_class = {}
        confidence_scores = []

        for drone_image in processed_drone_images:
            detections = drone_image.get_non_empty_predictions()
            total_detections += len(detections)

            for detection in detections:
                class_name = detection.class_name
                detection_by_class[class_name] = (
                    detection_by_class.get(class_name, 0) + 1
                )
                confidence_scores.append(detection.confidence)

        # Create detection results
        processing_time = time.time() - start_time
        confidence_stats = {
            "mean": sum(confidence_scores) / max(len(confidence_scores), 1)
            if confidence_scores
            else 0.0,
            "min": min(confidence_scores) if confidence_scores else 0.0,
            "max": max(confidence_scores) if confidence_scores else 0.0,
            "count": len(confidence_scores),
        }

        detection_results = DetectionResults(
            total_images=len(processed_drone_images),
            total_detections=total_detections,
            detection_by_class=detection_by_class,
            processing_time=processing_time,
            detection_confidence_stats=confidence_stats,
            geographic_coverage=self.census_manager._calculate_geographic_coverage(),
            campaign_id=self.campaign_id,
            metadata=self.config.metadata or {},
        )

        # Store results in census manager
        self.census_manager.set_detection_results(detection_results)

        logger.info(
            f"Detection completed: {total_detections} detections in {processing_time:.2f}s"
        )
        return detection_results

    def analyze_flight_path(self) -> Optional[FlightPath]:
        """Analyze the flight path from GPS data.

        Returns:
            Optional[FlightPath]: Analyzed flight path or None if no GPS data
        """
        return self.census_manager.analyze_flight_path()

    def calculate_flight_efficiency(self) -> Optional[FlightEfficiency]:
        """Calculate flight efficiency metrics.

        Returns:
            Optional[FlightEfficiency]: Flight efficiency metrics or None if no flight path
        """
        return self.census_manager.calculate_flight_efficiency()

    def merge_detections_geographically(
        self,
    ) -> None:
        """Merge detections across overlapping geographic regions."""
        self.census_manager.merge_detections_geographically(
            iou_threshold=self.config.detection_merging_threshold
        )
        return None

    def create_geographic_visualization(self, output_path: Optional[str] = None) -> str:
        """Create geographic visualization of the campaign.

        Args:
            output_path: Optional path to save the visualization

        Returns:
            str: Path to the saved visualization file
        """
        images = self.get_drone_images(as_dict=False)
        if not images:
            raise ValueError("No drone images available for visualization")

        if output_path is None:
            output_path = f"campaign_{self.campaign_id}_visualization.html"

        self.geographic_visualizer.create_map(images, save_path=output_path)

        logger.info(f"Geographic visualization saved to: {output_path}")
        return output_path

    def export_to_fiftyone(self) -> None:
        """Export campaign data to FiftyOne for visualization and analysis."""
        if self.fiftyone_manager is None:
            logger.warning("FiftyOne manager not initialized. Skipping export.")
            return

        images = self.get_drone_images(as_dict=False)
        if not images:
            logger.warning("No drone images available for FiftyOne export.")
            return
        self.fiftyone_manager.add_drone_images(images)
        self.fiftyone_manager.save_dataset()
        logger.info(f"Exported {len(images)} images to FiftyOne")

    def export_to_labelstudio(self, dotenv_path: Optional[str] = None) -> Optional[str]:
        """Export campaign data to LabelStudio for annotation/review."""

        if self.fiftyone_manager is None:
            logger.warning("FiftyOne manager not initialized. Skipping export.")
            return

        try:
            annot_key = f"campaign_{self.campaign_id}_review"
            self.fiftyone_manager.send_predictions_to_labelstudio(
                annot_key, dotenv_path=dotenv_path
            )
            logger.info(
                f"Exported FiftyOne dataset to LabelStudio with annot_key: {annot_key}"
            )
            return annot_key

        except Exception:
            logger.error(f"Error exporting to LabelStudio: {traceback.format_exc()}")
            return None

    def export_detection_report(self, output_path: str) -> None:
        """Export a comprehensive detection report.

        Args:
            output_path: Path to save the report
        """
        self.census_manager.export_detection_report(output_path)

        #
        results = self.get_drone_images(as_dict=True)
        path = Path(output_path).with_name("detections_and_images.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Images and their detections saved to: {path}")

    def get_campaign_statistics(self) -> Dict[str, Any]:
        """Get comprehensive campaign statistics.

        Returns:
            Dict[str, Any]: Complete campaign statistics
        """
        return self.census_manager.get_campaign_statistics()

    def get_all_detections(
        self,
    ) -> List[Detection]:
        """Get all detections from the campaign.

        Args:
            force_compute: Whether to force recomputation of detections

        Returns:
            List: All detections across the campaign
        """
        return self.census_manager.get_all_detections()

    def validate_image_paths(self, image_paths: List[str]) -> List[str]:
        """Validate image paths.

        Args:
            image_paths: List of image paths to validate
        Returns:
            List[str]: List of valid image paths
        Raises:
            ValueError: If none of the images have GPS coordinates
        """
        from PIL import Image

        from .gps.gps_utils import GPSUtils

        valid_images = []
        invalid_images = []
        no_gps_count = 0
        for image_path in image_paths:
            reason = None
            # Check if file is readable as an image
            assert isinstance(image_path, str), "image_paths must be a list of strings"
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify image integrity
            except Exception as e:
                reason = f"Unreadable image: {e}"
                invalid_images.append((image_path, reason))
                logger.warning(f"Invalid image: {image_path} | Reason: {reason}")
                continue
            # Check for GPS coordinates
            try:
                gps_result = GPSUtils.get_gps_coord(file_name=image_path)
                if gps_result is None:
                    reason = "No GPS coordinates found"
                    no_gps_count += 1
                    invalid_images.append((image_path, reason))
                    logger.warning(f"Invalid image: {image_path} | Reason: {reason}")
                    continue
            except Exception as e:
                reason = f"Error extracting GPS: {e}"
                invalid_images.append((image_path, reason))
                logger.warning(f"Invalid image: {image_path} | Reason: {reason}")
                continue
            # If both checks pass, add to valid images
            valid_images.append(image_path)

        logger.info(
            f"Validation complete: {len(valid_images)} valid images, {len(invalid_images)} invalid images."
        )
        if len(valid_images) == 0 or no_gps_count == len(image_paths):
            raise ValueError(
                "None of the images have GPS coordinates. Validation failed."
            )
        return valid_images

    def run_complete_campaign(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        export_to_fiftyone: bool = True,
    ) -> Dict[str, Any]:
        """Run a complete campaign from start to finish.

        Args:
            image_paths: List of image paths to process
            output_dir: Optional output directory for results
            run_flight_analysis: Whether to run flight path analysis
            run_geographic_merging: Whether to run geographic merging
            create_visualization: Whether to create geographic visualization
            export_to_fiftyone: Whether to export to FiftyOne

        Returns:
            Dict[str, Any]: Complete campaign results
        """
        logger.info(f"Starting complete campaign: {self.campaign_id}")

        # Validate image paths
        assert isinstance(image_paths, list), "image_paths must be a list"
        image_paths = self.validate_image_paths(image_paths)

        if output_dir is None:
            output_dir = ROOT / "census_campaign_results" / self.campaign_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(output_dir)
            logger.info(f"No output directory provided, using default: {output_dir}")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Add images
        self.add_images_from_paths(image_paths)

        # Step 2: Prepare data
        self.prepare_data()

        # Step 3: Run detection
        detection_results = self.run_detection(save_results=True, output_dir=output_dir)

        # Step 4: Flight analysis (optional)
        try:
            flight_path = None
            flight_efficiency = None
            flight_path = self.analyze_flight_path()
            if flight_path:
                flight_efficiency = self.calculate_flight_efficiency()
        except Exception as e:
            logger.error(f"Error analyzing flight path: {e}")
            flight_path = None
            flight_efficiency = None

        # Step 5: Geographic merging (optional)
        try:
            self.merge_detections_geographically()
        except Exception as e:
            logger.error(f"Error merging detections: {e}")
            # logger.error(f"Error merging detections: {traceback.format_exc()}")

        # Step 6: Create visualization (optional)
        if output_dir:
            try:
                visualization_path = Path(output_dir) / "geographic_visualization.html"
                visualization_path = self.create_geographic_visualization(
                    str(visualization_path)
                )
            except Exception as e:
                logger.error(f"Error creating geographic visualization: {e}")
                visualization_path = None

        # Step 7: Export to FiftyOne (optional)
        annot_key = None
        try:
            if export_to_fiftyone:
                self.export_to_fiftyone()
                annot_key = self.export_to_labelstudio()
        except Exception as e:
            logger.error(f"Error exporting to FiftyOne: {e}")
            annot_key = None

        # Step 8: Export final report
        try:
            if output_dir:
                report_path = Path(output_dir) / "campaign_report.json"
                self.export_detection_report(str(report_path))
        except Exception as e:
            logger.error(f"Error exporting detection report: {e}")
            report_path = None

        # Compile results
        results = {
            "campaign_id": self.campaign_id,
            "detection_results": detection_results,
            "flight_path": flight_path,
            "flight_efficiency": flight_efficiency,
            "merged_images": self.get_drone_images(as_dict=True),
            "visualization_path": visualization_path,
            "statistics": self.get_campaign_statistics(),
            "fiftyone_annot_key": annot_key,
        }

        logger.info(
            f"Campaign completed successfully: {detection_results.total_detections} detections found"
        )
        return results
