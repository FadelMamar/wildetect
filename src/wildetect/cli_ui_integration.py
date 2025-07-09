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
import traceback
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
                status_text.code(f"Running command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
                universal_newlines=True,
            )
            logs = ""
            if log_placeholder is not None:
                for line in process.stdout:
                    logs += line
                    log_placeholder.code(logs)

            process.stdout.close()
            return_code = process.wait()

            return {
                "success": True,
                "stdout": logs,
                "stderr": "",
                "return_code": return_code,
            }

        except Exception:
            self.logger.error(f"Subprocess error: {traceback.format_exc()}")
            return {
                "success": False,
                "error": traceback.format_exc(),
                "return_code": -1,
            }

    def run_detection_ui(
        self,
        images: List[str],
        dataset_name: Optional[str] = None,
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
    ) -> Dict[str, Any]:
        """Run detection from UI using subprocess."""
        try:
            # Build CLI arguments
            args = []

            # Add image paths
            args.extend(images)

            if dataset_name:
                args.extend(["--dataset", dataset_name])

            if status_text:
                status_text.text("Starting detection process...")

            # Run detection command
            result = self._run_cli_command(
                "detect",
                args,
                log_placeholder=log_placeholder,
                status_text=status_text,
            )
            return result

            # if not result["success"]:
            #    return result

            # Load results from output file
            # results_file = Path(output) / "results.json"
            # if results_file.exists():
            #    with open(results_file, "r", encoding="utf-8") as f:
            #        results_data = json.load(f)

            # return {
            #    "success": True,
            #    "results": results_data,
            #    "output_dir": output,
            # }
            # else:
            #    return {
            #        "success": False,
            #        "error": "Results file not found after detection",
            #        "output_dir": output,
            #    }

        except Exception:
            self.logger.error(f"Detection failed: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(traceback.format_exc()),
            }

    def run_census_ui(
        self,
        campaign_id: str,
        images: List[str],
        target_species: Optional[List[str]] = None,
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
    ) -> Dict[str, Any]:
        """Run census campaign from UI using subprocess."""
        try:
            # Build CLI arguments
            args = [campaign_id]
            args.extend(images)

            # Create temporary output directory if not provided

            if target_species:
                for species in target_species:
                    args.extend(["--species", species])

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
            # campaign_results = self._load_campaign_results(output, campaign_id)

            return {
                "success": True,
                "campaign_id": campaign_id,
                # "results": campaign_results,
                # "output_dir": output,
            }

        except Exception:
            self.logger.error(f"Census campaign failed: {traceback.format_exc()}")
            return {
                "success": False,
                "error": traceback.format_exc(),
                "campaign_id": campaign_id,
            }

    def _load_campaign_results(
        self, output_dir: str, campaign_id: str
    ) -> Dict[str, Any]:
        """Load campaign results from output directory."""
        try:
            campaign_file = Path(output_dir) / "campaign_report.json"
            if campaign_file.exists():
                with open(campaign_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                self.logger.error(f"Campaign results file not found: {campaign_file}")
                return {
                    "campaign_id": campaign_id,
                    "status": "completed",
                    "output_dir": output_dir,
                }
        except Exception as e:
            self.logger.error(
                f"Failed to load campaign results: {traceback.format_exc()}"
            )
            return {
                "campaign_id": campaign_id,
                "status": "error",
                "error": e,
                "output_dir": output_dir,
            }

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

        except Exception:
            self.logger.error(f"Failed to clear results: {traceback.format_exc()}")
            return {
                "success": False,
                "error": traceback.format_exc(),
            }

    def fiftyone_ui(
        self,
        dataset_name: Optional[str] = None,
        action: str = "launch",
        export_format: str = "coco",
        export_path: Optional[str] = None,
        log_placeholder: Optional["st.empty"] = None,
        status_text: Optional["st.empty"] = None,
    ) -> Dict[str, Any]:
        """Manage FiftyOne datasets from UI using subprocess."""
        try:
            # Build CLI arguments
            if dataset_name:
                args = [f"--dataset", dataset_name, "--action", action]
            else:
                args = [f"--action", action]

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

        except Exception:
            self.logger.error(f"FiftyOne operation failed: {traceback.format_exc()}")
            return {
                "success": False,
                "error": traceback.format_exc(),
                "action": action,
                "dataset_name": dataset_name,
            }


# Global instance for easy access
cli_ui_integration = CLIUIIntegration()
