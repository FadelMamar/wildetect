"""
CLI-UI Integration Module for WildDetect.

This module provides integration between the CLI functionality and the Streamlit UI,
allowing the UI to call CLI functions through subprocesses and display results in a web interface.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from rich.console import Console

console = Console()


class CLIUIIntegration:
    """Integration class for CLI and UI functionality using subprocesses."""

    def __init__(self):
        """Initialize the CLI-UI integration."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.cli_module_path = Path(__file__).parent / "cli.py"

    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def _run_cli_command(
        self,
        command: str,
        args: List[str],
        timeout: int = 300,
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Run a CLI command through subprocess and return results."""
        try:
            # Build the command
            cmd = ["wildetect", command] + args

            if status_text:
                status_text.text(f"Running command: {' '.join(cmd)}")

            # Run the subprocess
            if progress_bar:
                progress_bar.progress(0.1)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ),
            )

            # Monitor progress
            start_time = time.time()
            while process.poll() is None:
                if progress_bar:
                    elapsed = time.time() - start_time
                    progress = min(0.9, elapsed / timeout)
                    progress_bar.progress(progress)

                if status_text:
                    status_text.text(f"Running... ({elapsed:.1f}s)")

                time.sleep(0.1)

            # Get output
            stdout, stderr = process.communicate()
            return_code = process.returncode

            if progress_bar:
                progress_bar.progress(1.0)

            if return_code != 0:
                error_msg = stderr.strip() if stderr else "Unknown error"
                self.logger.error(f"CLI command failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "return_code": return_code,
                    "stdout": stdout,
                    "stderr": stderr,
                }

            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
            }

        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "return_code": -1,
            }
        except Exception as e:
            self.logger.error(f"Subprocess error: {e}")
            return {
                "success": False,
                "error": str(e),
                "return_code": -1,
            }

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
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Run detection from UI using subprocess."""
        try:
            # Create temporary output directory if not provided
            if not output:
                output = tempfile.mkdtemp(prefix="wildetect_detection_")

            # Build CLI arguments
            args = []

            # Add image paths
            args.extend(images)

            # Add optional parameters
            if model_path:
                args.extend(["--model", model_path])

            args.extend(["--type", model_type])
            args.extend(["--confidence", str(confidence)])
            args.extend(["--device", device])
            args.extend(["--batch-size", str(batch_size)])
            args.extend(["--tile-size", str(tile_size)])
            args.extend(["--output", output])

            if status_text:
                status_text.text("Starting detection process...")

            # Run detection command
            result = self._run_cli_command(
                "detect",
                args,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            if not result["success"]:
                return result

            # Load results from output file
            results_file = Path(output) / "results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    results_data = json.load(f)

                # Extract statistics
                total_images = len(results_data.get("drone_images", []))
                total_detections = sum(
                    drone_img.get("total_detections", 0)
                    for drone_img in results_data.get("drone_images", [])
                )

                return {
                    "success": True,
                    "results": results_data,
                    "total_images": total_images,
                    "total_detections": total_detections,
                    "output_dir": output,
                }
            else:
                return {
                    "success": False,
                    "error": "Results file not found after detection",
                    "output_dir": output,
                }

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
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
        """Run census campaign from UI using subprocess."""
        try:
            # Create temporary output directory if not provided
            if not output:
                output = tempfile.mkdtemp(prefix="wildetect_census_")

            # Build CLI arguments
            args = [campaign_id]
            args.extend(images)

            # Add optional parameters
            if model_path:
                args.extend(["--model", model_path])

            args.extend(["--confidence", str(confidence)])
            args.extend(["--device", device])
            args.extend(["--batch-size", str(batch_size)])
            args.extend(["--tile-size", str(tile_size)])
            args.extend(["--output", output])

            if pilot_name:
                args.extend(["--pilot", pilot_name])

            if target_species:
                for species in target_species:
                    args.extend(["--species", species])

            if not create_map:
                args.extend(["--no-map"])

            if status_text:
                status_text.text(f"Starting census campaign: {campaign_id}")

            # Run census command
            result = self._run_cli_command(
                "census",
                args,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            if not result["success"]:
                return result

            # Load campaign results
            campaign_results = self._load_campaign_results(output, campaign_id)

            return {
                "success": True,
                "campaign_id": campaign_id,
                "results": campaign_results,
                "output_dir": output,
            }

        except Exception as e:
            self.logger.error(f"Census campaign failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "campaign_id": campaign_id,
            }

    def analyze_results_ui(
        self,
        results_path: str,
        output_dir: str = "analysis",
        create_map: bool = True,
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Analyze detection results from UI using subprocess."""
        try:
            # Build CLI arguments
            args = [results_path]
            args.extend(["--output", output_dir])

            if not create_map:
                args.extend(["--no-map"])

            if status_text:
                status_text.text("Starting analysis...")

            # Run analyze command
            result = self._run_cli_command(
                "analyze",
                args,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            if not result["success"]:
                return result

            # Load analysis results
            analysis_file = Path(output_dir) / "analysis_report.json"
            if analysis_file.exists():
                with open(analysis_file, "r") as f:
                    analysis_results = json.load(f)
            else:
                analysis_results = {}

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
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Visualize detection results from UI using subprocess."""
        try:
            # Build CLI arguments
            args = [results_path]
            args.extend(["--output", output_dir])

            if not show_confidence:
                args.extend(["--no-confidence"])

            if not create_map:
                args.extend(["--no-map"])

            if status_text:
                status_text.text("Creating visualizations...")

            # Run visualize command
            result = self._run_cli_command(
                "visualize",
                args,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            if not result["success"]:
                return result

            # Load visualization data
            visualization_file = Path(output_dir) / "visualization_report.json"
            if visualization_file.exists():
                with open(visualization_file, "r") as f:
                    visualization_data = json.load(f)
            else:
                visualization_data = {}

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
        """Get system information for UI display using subprocess."""
        try:
            result = self._run_cli_command("info", [])

            if result["success"]:
                # Parse the system info from stdout
                system_info = self._parse_system_info(result["stdout"])
                return {
                    "success": True,
                    "system_info": system_info,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to get system info"),
                    "system_info": {},
                }

        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {
                "success": False,
                "error": str(e),
                "system_info": {},
            }

    def _load_campaign_results(
        self, output_dir: str, campaign_id: str
    ) -> Dict[str, Any]:
        """Load campaign results from output directory."""
        try:
            campaign_file = Path(output_dir) / "campaign_report.json"
            if campaign_file.exists():
                with open(campaign_file, "r") as f:
                    return json.load(f)
            else:
                return {
                    "campaign_id": campaign_id,
                    "status": "completed",
                    "output_dir": output_dir,
                }
        except Exception as e:
            self.logger.error(f"Failed to load campaign results: {e}")
            return {
                "campaign_id": campaign_id,
                "status": "error",
                "error": str(e),
                "output_dir": output_dir,
            }

    def _parse_system_info(self, stdout: str) -> Dict[str, Any]:
        """Parse system information from CLI output."""
        system_info = {
            "components": {},
            "dependencies": {},
            "timestamp": datetime.now().isoformat(),
        }

        lines = stdout.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for table headers or section indicators
            if "Component" in line and "Status" in line and "Details" in line:
                current_section = "components"
            elif "PyTorch" in line or "CUDA" in line:
                # Parse component lines
                parts = line.split("|")
                if len(parts) >= 3:
                    component = parts[0].strip()
                    status = parts[1].strip()
                    details = parts[2].strip()

                    if current_section == "components":
                        system_info["components"][component] = {
                            "status": status,
                            "details": details,
                        }
            elif any(
                dep in line
                for dep in ["PIL", "numpy", "tqdm", "ultralytics", "folium", "shapely"]
            ):
                # Parse dependency lines
                parts = line.split("|")
                if len(parts) >= 3:
                    dependency = parts[0].strip()
                    status = parts[1].strip()
                    details = parts[2].strip()

                    system_info["dependencies"][dependency] = {
                        "status": status,
                        "details": details,
                    }

        return system_info

    def clear_results_ui(
        self,
        results_dir: str = "results",
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Clear results using subprocess."""
        try:
            if status_text:
                status_text.text(f"Clearing results in {results_dir}...")

            # Run clear command with confirmation
            result = self._run_cli_command(
                "clear-results",
                [results_dir],
                progress_bar=progress_bar,
                status_text=status_text,
            )

            return {
                "success": result["success"],
                "message": "Results cleared successfully"
                if result["success"]
                else result.get("error", "Failed to clear results"),
            }

        except Exception as e:
            self.logger.error(f"Failed to clear results: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def fiftyone_ui(
        self,
        dataset_name: str = "wildlife_detection",
        action: str = "launch",
        export_format: str = "coco",
        export_path: Optional[str] = None,
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Manage FiftyOne datasets from UI using subprocess."""
        try:
            # Build CLI arguments
            args = [f"--dataset", dataset_name, "--action", action]

            if action == "export":
                args.extend(["--format", export_format])
                if export_path:
                    args.extend(["--output", export_path])

            if status_text:
                status_text.text(f"Running FiftyOne {action}...")

            # Run fiftyone command
            result = self._run_cli_command(
                "fiftyone",
                args,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            return {
                "success": result["success"],
                "action": action,
                "dataset_name": dataset_name,
                "output": result.get("stdout", ""),
                "error": result.get("error", ""),
            }

        except Exception as e:
            self.logger.error(f"FiftyOne operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action,
                "dataset_name": dataset_name,
            }

    def labelstudio_ui(
        self,
        action: str = "status",
        project_name: str = "wildlife_detection",
        results_path: Optional[str] = None,
        export_format: str = "yolo",
        export_path: Optional[str] = None,
        progress_bar=None,
        status_text=None,
    ) -> Dict[str, Any]:
        """Manage LabelStudio projects from UI using subprocess."""
        try:
            # Build CLI arguments
            args = [f"--action", action, "--project", project_name]

            if action == "create" and results_path:
                args.extend(["--results", results_path])
            elif action == "export":
                args.extend(["--format", export_format])
                if export_path:
                    args.extend(["--output", export_path])

            if status_text:
                status_text.text(f"Running LabelStudio {action}...")

            # Run labelstudio command
            result = self._run_cli_command(
                "labelstudio",
                args,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            return {
                "success": result["success"],
                "action": action,
                "project_name": project_name,
                "output": result.get("stdout", ""),
                "error": result.get("error", ""),
            }

        except Exception as e:
            self.logger.error(f"LabelStudio operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action,
                "project_name": project_name,
            }


# Global instance for easy access
cli_ui_integration = CLIUIIntegration()
