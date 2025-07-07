"""
CLI-UI Integration Module for WildDetect.

This module provides integration between the CLI functionality and the Streamlit UI,
allowing the UI to call CLI functions and display results in a web interface.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core.campaign_manager import CampaignConfig, CampaignManager
from .core.config import FlightSpecs, LoaderConfig, PredictionConfig
from .core.data.census import CensusDataManager
from .core.detection_pipeline import DetectionPipeline
from .core.visualization.geographic import GeographicVisualizer, VisualizationConfig

console = Console()


class CLIUIIntegration:
    """Integration class for CLI and UI functionality."""

    def __init__(self):
        """Initialize the CLI-UI integration."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def run_detection_ui(
        self,
        images: List[str],
        model_path: Optional[str] = None,
        model_type: str = "yolo",
        confidence: float = 0.25,
        device: str = "auto",
        batch_size: int = 8,
        tile_size: int = 640,
        output: Optional[str] = None,
        max_images: Optional[int] = None,
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Run detection from UI with progress tracking."""
        try:
            # Determine if input is directory or file paths
            if len(images) == 1 and Path(images[0]).is_dir():
                image_dir = images[0]
                image_paths = None
                if status_text:
                    status_text.text(f"Processing directory: {image_dir}")
            else:
                image_dir = None
                image_paths = images
                if status_text:
                    status_text.text(f"Processing {len(images)} images")

            # Create configurations
            pred_config = PredictionConfig(
                model_path=model_path,
                model_type=model_type,
                confidence_threshold=confidence,
                device=device,
                batch_size=batch_size,
                tilesize=tile_size,
            )

            loader_config = LoaderConfig(
                tile_size=tile_size,
                batch_size=batch_size,
            )

            # Create detection pipeline
            pipeline = DetectionPipeline(
                config=pred_config,
                loader_config=loader_config,
            )

            # Run detection with UI progress tracking
            if progress_bar:
                progress_bar.progress(0)
                if status_text is not None:
                    status_text.text("Running detection...")

            drone_images = pipeline.run_detection(
                image_paths=image_paths or [],
                image_dir=image_dir,
                save_path=output + "/results.json" if output else None,
            )

            if progress_bar:
                progress_bar.progress(1.0)
                if status_text is not None:
                    status_text.text("Detection completed!")

            # Convert results to UI-friendly format
            results = self._convert_drone_images_to_ui_format(drone_images)

            return {
                "success": True,
                "drone_images": drone_images,
                "results": results,
                "total_images": len(drone_images),
                "total_detections": sum(
                    result.get("total_detections", 0) for result in results
                ),
            }

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "drone_images": [],
                "results": [],
                "total_images": 0,
                "total_detections": 0,
            }

    def run_census_ui(
        self,
        campaign_id: str,
        images: List[str],
        model_path: Optional[str] = None,
        model_type: str = "yolo",
        confidence: float = 0.25,
        device: str = "auto",
        batch_size: int = 8,
        tile_size: int = 640,
        output: Optional[str] = None,
        pilot_name: Optional[str] = None,
        target_species: Optional[List[str]] = None,
        create_map: bool = True,
        progress_bar=None,
        status_text=None,
        sensor_height: float = 24.0,
        focal_length: float = 35.0,
        flight_height: float = 180.0,
        equipment_info: Optional[Dict[str, Union[str, int, float]]] = None,
    ) -> Dict[str, Any]:
        """Run census campaign from UI with progress tracking."""
        try:
            flight_specs = FlightSpecs(
                sensor_height=sensor_height,
                focal_length=focal_length,
                flight_height=flight_height,
            )

            if status_text:
                status_text.text(f"Starting Wildlife Census Campaign: {campaign_id}")

            # Determine if input is directory or file paths
            if len(images) == 1 and Path(images[0]).is_dir():
                image_dir = images[0]
                image_paths = None
                if status_text:
                    status_text.text(f"Processing directory: {image_dir}")
            else:
                image_dir = None
                image_paths = images
                if status_text:
                    status_text.text(f"Processing {len(images)} images")

            # Create campaign metadata
            campaign_metadata = {
                "pilot_info": {
                    "name": pilot_name or "Unknown",
                    "experience": "Unknown",
                },
                "weather_conditions": {
                    "temperature": 25,
                    "wind_speed": 5,
                    "visibility": "good",
                },
                "mission_objectives": ["wildlife_survey", "habitat_mapping"],
                "target_species": target_species,
                "flight_parameters": vars(flight_specs),
                "equipment_info": equipment_info,
            }

            # Create campaign configuration
            pred_config = PredictionConfig(
                model_path=model_path,
                model_type=model_type,
                confidence_threshold=confidence,
                device=device,
                batch_size=batch_size,
                tilesize=tile_size,
                flight_specs=flight_specs,
            )

            loader_config = LoaderConfig(
                tile_size=tile_size,
                batch_size=batch_size,
                flight_specs=flight_specs,
            )

            # Create campaign configuration
            campaign_config = CampaignConfig(
                campaign_id=campaign_id,
                loader_config=loader_config,
                prediction_config=pred_config,
                metadata=campaign_metadata,
                fiftyone_dataset_name=f"campaign_{campaign_id}",
            )

            # Initialize campaign manager
            campaign_manager = CampaignManager(campaign_config)

            # Run complete campaign with UI progress tracking
            if progress_bar:
                progress_bar.progress(0)
                if status_text is not None:
                    status_text.text("Running census campaign...")

            results = campaign_manager.run_complete_campaign(
                image_paths=image_paths or [],
                output_dir=output,
                tile_size=tile_size,
                overlap=0.2,
                run_flight_analysis=True,
                run_geographic_merging=True,
                create_visualization=create_map,
                export_to_fiftyone=False,
            )

            if progress_bar:
                progress_bar.progress(1.0)
                if status_text is not None:
                    status_text.text("Census campaign completed!")

            return {
                "success": True,
                "campaign_manager": campaign_manager,
                "results": results,
                "campaign_id": campaign_id,
            }

        except Exception as e:
            self.logger.error(f"Census campaign failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "campaign_manager": None,
                "results": {},
                "campaign_id": campaign_id,
            }

    def analyze_results_ui(
        self,
        results_path: str,
        output_dir: str = "analysis",
        create_map: bool = True,
    ) -> Dict[str, Any]:
        """Analyze detection results from UI."""
        try:
            results_path_obj = Path(results_path)
            if not results_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Results file not found: {results_path}",
                }

            # Load results
            with open(results_path_obj, "r") as f:
                results = json.load(f)

            # Create output directory
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)

            # Analyze results
            analysis_results = self._analyze_detection_results(results)

            # Create geographic visualization if requested
            if create_map and "drone_images" in results:
                self._create_geographic_visualization(
                    results["drone_images"], output_dir
                )

            # Export analysis report
            self._export_analysis_report(analysis_results, output_dir)

            return {
                "success": True,
                "analysis_results": analysis_results,
                "output_dir": output_dir,
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_results": {},
                "output_dir": output_dir,
            }

    def visualize_results_ui(
        self,
        results_path: str,
        output_dir: str = "visualizations",
        show_confidence: bool = True,
        create_map: bool = True,
    ) -> Dict[str, Any]:
        """Visualize detection results from UI."""
        try:
            results_path_obj = Path(results_path)
            if not results_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Results file not found: {results_path}",
                }

            # Load results
            with open(results_path_obj, "r") as f:
                results = json.load(f)

            # Create output directory
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)

            # Display basic statistics
            visualization_data = self._extract_visualization_data(results)

            # Create geographic visualization if requested
            if create_map and "drone_images" in results:
                self._create_geographic_visualization(
                    results["drone_images"], output_dir
                )

            # Export visualization report
            self._export_visualization_report(
                results_path, visualization_data, output_dir
            )

            return {
                "success": True,
                "visualization_data": visualization_data,
                "output_dir": output_dir,
            }

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "visualization_data": {},
                "output_dir": output_dir,
            }

    def get_system_info_ui(self) -> Dict[str, Any]:
        """Get system information for UI display."""
        system_info = {
            "components": {},
            "dependencies": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Check PyTorch
        try:
            import torch

            system_info["components"]["PyTorch"] = {
                "status": "✓",
                "details": f"Version {torch.__version__}",
            }
        except ImportError:
            system_info["components"]["PyTorch"] = {
                "status": "✗",
                "details": "Not installed",
            }

        # Check CUDA
        try:
            import torch

            if torch.cuda.is_available():
                system_info["components"]["CUDA"] = {
                    "status": "✓",
                    "details": f"Available - {torch.cuda.get_device_name()}",
                }
            else:
                system_info["components"]["CUDA"] = {
                    "status": "✗",
                    "details": "Not available",
                }
        except ImportError:
            system_info["components"]["CUDA"] = {
                "status": "✗",
                "details": "PyTorch not installed",
            }

        # Check other dependencies
        dependencies = [
            ("PIL", "PIL"),
            ("numpy", "numpy"),
            ("tqdm", "tqdm"),
            ("ultralytics", "ultralytics"),
            ("folium", "folium"),
            ("shapely", "shapely"),
        ]

        for name, module in dependencies:
            try:
                __import__(module)
                system_info["dependencies"][name] = {
                    "status": "✓",
                    "details": "Installed",
                }
            except ImportError:
                system_info["dependencies"][name] = {
                    "status": "✗",
                    "details": "Not installed",
                }

        return system_info

    def _convert_drone_images_to_ui_format(self, drone_images: List) -> List[Dict]:
        """Convert drone images to UI-friendly format."""
        results = []
        for drone_image in drone_images:
            stats = drone_image.get_statistics()
            results.append(
                {
                    "image_path": stats["image_path"],
                    "total_detections": stats.get("total_detections", 0),
                    "class_counts": stats.get("class_counts", {}),
                    "species_counts": stats.get("class_counts", {}),  # Alias for UI
                    "total_count": stats.get("total_detections", 0),  # Alias for UI
                }
            )
        return results

    def _analyze_detection_results(self, results: Union[dict, list]) -> dict:
        """Analyze detection results and generate insights."""
        analysis = {
            "total_images": 0,
            "total_detections": 0,
            "species_breakdown": {},
            "confidence_distribution": {},
            "geographic_coverage": {},
            "processing_efficiency": {},
        }

        # Extract statistics from results
        if isinstance(results, list):
            # Handle list of image results
            analysis["total_images"] = len(results)
            for result in results:
                if "total_detections" in result:
                    analysis["total_detections"] += result["total_detections"]
                if "class_counts" in result:
                    for species, count in result["class_counts"].items():
                        analysis["species_breakdown"][species] = (
                            analysis["species_breakdown"].get(species, 0) + count
                        )

        return analysis

    def _extract_visualization_data(self, results: Union[dict, list]) -> dict:
        """Extract visualization data from results."""
        visualization_data = {
            "total_images": 0,
            "total_detections": 0,
            "species_counts": {},
            "timestamp": datetime.now().isoformat(),
        }

        if isinstance(results, list):
            visualization_data["total_images"] = len(results)
            total_detections = sum(
                result.get("total_detections", 0) for result in results
            )
            visualization_data["total_detections"] = total_detections

            # Species breakdown
            species_counts = {}
            for result in results:
                for species, count in result.get("class_counts", {}).items():
                    species_counts[species] = species_counts.get(species, 0) + count

            visualization_data["species_counts"] = species_counts

        return visualization_data

    def _create_geographic_visualization(
        self, drone_images: List, output_dir: Optional[str] = None
    ):
        """Create geographic visualization of drone images."""
        if not drone_images:
            return

        try:
            # Create visualizer
            config = VisualizationConfig(
                show_image_bounds=True,
                show_image_centers=True,
                show_statistics=True,
            )
            visualizer = GeographicVisualizer(config)

            # Create map
            map_obj = visualizer.create_map(drone_images)

            # Save map if output directory provided
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                map_file = output_path / "geographic_visualization.html"
                map_obj.save(str(map_file))

        except Exception as e:
            self.logger.error(f"Failed to create geographic visualization: {e}")

    def _export_analysis_report(self, analysis_results: dict, output_dir: str):
        """Export analysis report."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            report_file = output_path / "analysis_report.json"
            with open(report_file, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to export analysis report: {e}")

    def _export_visualization_report(
        self, results_path: str, visualization_data: dict, output_dir: str
    ):
        """Export visualization report."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            report_file = output_path / "visualization_report.json"
            with open(report_file, "w") as f:
                json.dump(visualization_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to export visualization report: {e}")


# Global instance for easy access
cli_ui_integration = CLIUIIntegration()
