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
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
    ) -> Dict[str, Any]:
        """Run a CLI command through subprocess and return results."""
        try:
            # Build the command
            cmd = ["wildetect", command] + args

            if status_text:
                status_text.text(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
                bufsize=1,
                universal_newlines=True,
            )

            if log_placeholder is not None:
                logs = ""
                for line in process.stdout:
                    logs += line
                    log_placeholder.code(
                        logs
                    )  # Update the Streamlit code block with new logs

            process.stdout.close()
            return_code = process.wait()

            return {
                "success": True,
                "stdout": logs,
                "stderr": "",
                "return_code": return_code,
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
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
        output: str = "results",
    ) -> Dict[str, Any]:
        """Run detection from UI using subprocess."""
        try:
            # Build CLI arguments
            args = []

            # Add image paths
            args.extend(images)

            if status_text:
                status_text.text("Starting detection process...")

            # Run detection command
            result = self._run_cli_command(
                "detect",
                args,
                log_placeholder=log_placeholder,
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
        output: str = "results",
        pilot_name: Optional[str] = None,
        target_species: Optional[List[str]] = None,
        create_map: bool = True,
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
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
                log_placeholder=log_placeholder,
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
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
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
                log_placeholder=log_placeholder,
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
        output_dir: str = "results",
        show_confidence: bool = True,
        create_map: bool = True,
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
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
                log_placeholder=log_placeholder,
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
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
    ) -> Dict[str, Any]:
        """Clear results using subprocess."""
        try:
            if status_text:
                status_text.text(f"Clearing results in {results_dir}...")

            # Run clear command with confirmation
            result = self._run_cli_command(
                "clear-results",
                [results_dir],
                log_placeholder=log_placeholder,
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
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
    ) -> Dict[str, Any]:
        """Manage FiftyOne datasets from UI using subprocess."""
        try:
            # Build CLI arguments
            args = [f"--dataset", dataset_name, "--action", action]

            if action == "export":
                args.extend(["--format", export_format])
                if export_path:
                    args.extend(["--output", export_path])

            # Run fiftyone command
            result = self._run_cli_command(
                "fiftyone",
                args,
                log_placeholder=log_placeholder,
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


# Global instance for easy access
cli_ui_integration = CLIUIIntegration()
