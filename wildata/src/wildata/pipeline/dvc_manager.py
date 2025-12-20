"""
DVC (Data Version Control) integration for the WildTrain data pipeline.
Provides data versioning, remote storage, and pipeline management capabilities.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class DVCStorageType(Enum):
    """Supported DVC storage types."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    SSH = "ssh"
    HDFS = "hdfs"


@dataclass
class DVCConfig:
    """Configuration for DVC integration."""

    storage_type: DVCStorageType = DVCStorageType.S3
    storage_path: str = "s3://dvc-storage"
    cache_dir: Optional[str] = None
    remote_name: str = "dataregistry"
    auto_push: bool = False
    auto_pull: bool = False


class DVCManager:
    """
    Manages DVC integration for data versioning and remote storage.

    Features:
    - Data versioning with Git-like commands
    - Remote storage integration (S3, GCS, Azure, etc.)
    - Pipeline management with dvc.yaml
    - Experiment tracking
    - Data lineage tracking
    """

    def __init__(self, project_root: Path, config: Optional[DVCConfig] = None):
        """
        Initialize the DVC manager.

        Args:
            project_root: Root directory of the project
            config: DVC configuration
        """
        self.project_root = project_root
        self.config = config or DVCConfig(
            storage_type=DVCStorageType.LOCAL,
            storage_path=str(project_root / "dvc_storage"),
        )
        self.logger = logging.getLogger(__name__)

        # Ensure DVC is initialized
        self._ensure_dvc_initialized()

    def _ensure_dvc_initialized(self):
        """Ensure DVC is initialized in the project."""
        dvc_dir = self.project_root / ".dvc"
        if not dvc_dir.exists():
            self.logger.info("Initializing DVC...")
            self._run_dvc_command(["init"])

    def _run_dvc_command(
        self, args: List[str], capture_output: bool = True
    ) -> Tuple[int, str, str]:
        """
        Run a DVC command.

        Args:
            args: Command arguments
            capture_output: Whether to capture output

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        cmd = ["dvc"] + args
        self.logger.debug(f"Running DVC command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            raise RuntimeError(
                "DVC is not installed. Please install DVC first: pip install dvc"
            )

    def setup_remote_storage(self, force: bool = False) -> bool:
        """
        Setup remote storage for DVC.

        Args:
            force: Whether to force reconfiguration

        Returns:
            True if setup was successful
        """
        if self.config.storage_type != DVCStorageType.S3:
            raise ValueError(f"{self.config.storage_type} storage is not supported")
        try:
            # Check if remote already exists
            if not force:
                returncode, stdout, stderr = self._run_dvc_command(["remote", "list"])
                if self.config.remote_name in stdout:
                    self.logger.info(
                        f"Remote '{self.config.remote_name}' already exists"
                    )
                    return True

            # Add remote storage
            remote_args = [
                "remote",
                "add",
                "-d",
                self.config.remote_name,
                self.config.storage_path,
            ]

            returncode, stdout, stderr = self._run_dvc_command(remote_args)

            if returncode == 0:
                self.logger.info(
                    f"Successfully setup remote storage: {self.config.storage_path}"
                )
                return True
            else:
                self.logger.error(f"Failed to setup remote storage: {stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error setting up remote storage: {e}")
            return False

    def add_data_to_dvc(self, data_path: Path, dataset_name: str) -> bool:
        """
        Add data to DVC tracking.

        Args:
            data_path: Path to the data directory/file
            dataset_name: Name of the dataset

        Returns:
            True if successful
        """
        try:
            # Add to DVC
            returncode, stdout, stderr = self._run_dvc_command(["add", str(data_path)])

            if returncode == 0:
                self.logger.info(f"Added {data_path} to DVC tracking")

                # Commit to Git if auto-push is enabled
                if self.config.auto_push:
                    self._commit_and_push(dataset_name)

                return True
            else:
                self.logger.error(f"Failed to add {data_path} to DVC: {stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error adding data to DVC: {e}")
            return False

    def _commit_and_push(self, dataset_name: str):
        """Commit changes to Git and push to remote."""
        try:
            # Git add
            subprocess.run(["git", "add", "."], cwd=self.project_root, check=True)

            # Git commit
            commit_message = f"Add dataset: {dataset_name}"
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.project_root,
                check=True,
            )

            # Push to remote
            if self.config.auto_push:
                returncode, stdout, stderr = self._run_dvc_command(["push"])
                if returncode == 0:
                    self.logger.info("Pushed data to remote storage")
                else:
                    self.logger.warning(f"Failed to push to remote: {stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Git operations failed: {e}")
        except Exception as e:
            self.logger.error(f"Error in commit and push: {e}")

    def pull_data(self, dataset_name: Optional[str] = None) -> bool:
        """
        Pull data from remote storage.

        Args:
            dataset_name: Specific dataset to pull (None for all)

        Returns:
            True if successful
        """
        try:
            if dataset_name:
                # Pull specific dataset
                returncode, stdout, stderr = self._run_dvc_command(
                    ["pull", f"data/{dataset_name}"]
                )
            else:
                # Pull all data
                returncode, stdout, stderr = self._run_dvc_command(["pull"])

            if returncode == 0:
                self.logger.info(f"Successfully pulled data from remote")
                return True
            else:
                self.logger.error(f"Failed to pull data: {stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error pulling data: {e}")
            return False

    def create_pipeline(self, pipeline_name: str, stages: List[Dict[str, Any]]) -> bool:
        """
        Create a DVC pipeline for data processing.

        Args:
            pipeline_name: Name of the pipeline
            stages: List of pipeline stages

        Returns:
            True if successful
        """
        try:
            pipeline_file = self.project_root / "dvc.yaml"

            # Create pipeline configuration
            pipeline_config = {"stages": {}}

            for stage in stages:
                stage_name = stage["name"]
                pipeline_config["stages"][stage_name] = {
                    "cmd": stage["command"],
                    "deps": stage.get("deps", []),
                    "outs": stage.get("outs", []),
                    "params": stage.get("params", []),
                }

            # Write pipeline file
            with open(pipeline_file, "w") as f:
                yaml.dump(pipeline_config, f, default_flow_style=False)

            self.logger.info(f"Created DVC pipeline: {pipeline_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating pipeline: {e}")
            return False

    def run_pipeline(self, pipeline_name: str) -> bool:
        """
        Run a DVC pipeline.

        Args:
            pipeline_name: Name of the pipeline to run

        Returns:
            True if successful
        """
        try:
            returncode, stdout, stderr = self._run_dvc_command(["repro", pipeline_name])

            if returncode == 0:
                self.logger.info(f"Successfully ran pipeline: {pipeline_name}")
                return True
            else:
                self.logger.error(f"Failed to run pipeline: {stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error running pipeline: {e}")
            return False

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets tracked by DVC.

        Returns:
            List of dataset information
        """
        try:
            returncode, stdout, stderr = self._run_dvc_command(["list", "data"])

            if returncode == 0:
                datasets = []
                for line in stdout.strip().split("\n"):
                    if line.strip():
                        datasets.append(
                            {"name": line.strip(), "path": f"data/{line.strip()}"}
                        )
                return datasets
            else:
                self.logger.error(f"Failed to list datasets: {stderr}")
                return []

        except Exception as e:
            self.logger.error(f"Error listing datasets: {e}")
            return []

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset information or None if not found
        """
        try:
            dataset_path = self.project_root / "data" / dataset_name

            if not dataset_path.exists():
                return None

            # Get file size and modification time
            total_size = sum(
                f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()
            )

            return {
                "name": dataset_name,
                "path": str(dataset_path),
                "size_bytes": total_size,
                "size_mb": total_size / (1024 * 1024),
                "tracked_by_dvc": True,
            }

        except Exception as e:
            self.logger.error(f"Error getting dataset info: {e}")
            return None

    def remove_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Remove a dataset from DVC tracking.

        Args:
            dataset_name: Name of the dataset to remove
            force: Whether to force removal

        Returns:
            True if successful
        """
        try:
            dataset_path = self.project_root / "data" / dataset_name

            if not dataset_path.exists():
                self.logger.warning(f"Dataset {dataset_name} does not exist")
                return True

            # Remove from DVC
            returncode, stdout, stderr = self._run_dvc_command(
                ["remove", str(dataset_path)]
            )

            if returncode == 0:
                self.logger.info(f"Removed dataset {dataset_name} from DVC tracking")

                # Commit changes
                if self.config.auto_push:
                    self._commit_and_push(f"Remove dataset: {dataset_name}")

                return True
            else:
                self.logger.error(f"Failed to remove dataset: {stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error removing dataset: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get DVC status information.

        Returns:
            Status information dictionary
        """
        try:
            # Get DVC status
            returncode, stdout, stderr = self._run_dvc_command(["status"])

            status_info = {
                "dvc_initialized": True,
                "remote_configured": False,
                "data_tracked": False,
                "status_output": stdout if returncode == 0 else stderr,
            }

            # Check if remote is configured
            remote_returncode, remote_stdout, remote_stderr = self._run_dvc_command(
                ["remote", "list"]
            )
            if remote_returncode == 0 and self.config.remote_name in remote_stdout:
                status_info["remote_configured"] = True

            # Check if data is tracked
            data_dir = self.project_root / "data"
            if data_dir.exists() and any(data_dir.iterdir()):
                status_info["data_tracked"] = True

            return status_info

        except Exception as e:
            return {"dvc_initialized": False, "error": str(e)}
