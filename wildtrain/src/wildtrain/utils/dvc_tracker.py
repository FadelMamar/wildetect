"""
DVC integration utilities for dataset tracking in WildTrain.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DVCTracker:
    """Utility class for tracking datasets with DVC and linking to MLflow runs."""

    def __init__(self, dvc_root: Optional[Path] = None):
        """
        Initialize DVC tracker.

        Args:
            dvc_root: Root directory for DVC operations. Defaults to current directory.
        """
        self.dvc_root = dvc_root or Path.cwd()
        self._ensure_dvc_initialized()

    def _ensure_dvc_initialized(self):
        """Ensure DVC is initialized in the project."""
        try:
            subprocess.run(["dvc", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("DVC is not installed or not available in PATH")

        # Check if DVC is initialized
        if not (self.dvc_root / ".dvc").exists():
            logger.info("Initializing DVC in project...")
            subprocess.run(["dvc", "init"], cwd=self.dvc_root, check=True)

    def track_dataset(self, dataset_path: Path, dataset_name: str) -> str:
        """
        Track a dataset with DVC.

        Args:
            dataset_path: Path to the dataset directory
            dataset_name: Name for the dataset

        Returns:
            DVC hash of the tracked dataset
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        # Add dataset to DVC
        logger.info(f"Tracking dataset: {dataset_path}")
        result = subprocess.run(
            ["dvc", "add", str(dataset_path)],
            cwd=self.dvc_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Extract the hash from DVC output
        output_lines = result.stdout.split("\n")
        hash_line = [line for line in output_lines if "Hash" in line]
        if hash_line:
            dataset_hash = hash_line[0].split()[-1]
        else:
            # Fallback: get hash from .dvc file
            dvc_file = dataset_path.with_suffix(".dvc")
            if dvc_file.exists():
                with open(dvc_file, "r") as f:
                    content = f.read()
                    # Extract hash from the file content
                    for line in content.split("\n"):
                        if line.startswith("  hash:"):
                            dataset_hash = line.split(":")[1].strip()
                            break
            else:
                raise RuntimeError("Could not extract dataset hash from DVC")

        # Commit the .dvc file to git
        dvc_file = dataset_path.with_suffix(".dvc")
        if dvc_file.exists():
            subprocess.run(
                ["git", "add", str(dvc_file.relative_to(self.dvc_root))],
                cwd=self.dvc_root,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", f"Add dataset: {dataset_name}"],
                cwd=self.dvc_root,
                check=True,
            )

        logger.info(f"Dataset {dataset_name} tracked with hash: {dataset_hash}")
        return dataset_hash

    def get_dataset_info(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Get information about a tracked dataset.

        Args:
            dataset_path: Path to the dataset

        Returns:
            Dictionary containing dataset information
        """
        dataset_path = Path(dataset_path)
        metadata_file = dataset_path / "metadata.json"

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        else:
            # Fallback: basic info
            return {
                "dataset_name": dataset_path.name,
                "path": str(dataset_path),
                "exists": dataset_path.exists(),
            }

    def link_dataset_to_mlflow(
        self, dataset_path: Path, dataset_name: str, run_id: str
    ) -> None:
        """
        Link a tracked dataset to the current MLflow run.

        Args:
            dataset_path: Path to the dataset
            dataset_name: Name for the dataset
            run_id: MLflow run ID. If None, uses current active run.
        """
        try:
            # Get dataset info
            dataset_info = self.get_dataset_info(dataset_path)

            # Log dataset info to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param(f"dataset_{dataset_name}_path", str(dataset_path))
                mlflow.log_param(f"dataset_{dataset_name}_name", dataset_name)
                mlflow.log_dict(dataset_info, f"dataset_{dataset_name}_info.json")

            logger.info(f"Linked dataset {dataset_name} to MLflow run")

        except Exception as e:
            logger.error(f"Failed to link dataset to MLflow: {e}")

    def push_dataset(self, dataset_path: Path) -> None:
        """
        Push a tracked dataset to remote storage.

        Args:
            dataset_path: Path to the dataset
        """
        logger.info(f"Pushing dataset to remote storage: {dataset_path}")
        subprocess.run(
            ["dvc", "push", str(dataset_path)], cwd=self.dvc_root, check=True
        )
        logger.info("Dataset pushed successfully")

    def pull_dataset(self, dataset_path: Path) -> None:
        """
        Pull a tracked dataset from remote storage.

        Args:
            dataset_path: Path to the dataset
        """
        logger.info(f"Pulling dataset from remote storage: {dataset_path}")
        subprocess.run(
            ["dvc", "pull", str(dataset_path)], cwd=self.dvc_root, check=True
        )
        logger.info("Dataset pulled successfully")

    def run_pipeline(self, pipeline_name: str = "prepare_data") -> None:
        """
        Run a DVC pipeline stage.

        Args:
            pipeline_name: Name of the pipeline stage to run
        """
        logger.info(f"Running DVC pipeline: {pipeline_name}")
        subprocess.run(["dvc", "repro", pipeline_name], cwd=self.dvc_root, check=True)
        logger.info("Pipeline completed successfully")

    def track_dataset_for_training(
        self,
        dataset_path: Path,
        dataset_name: str,
        link_to_mlflow: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Convenience function to track a dataset for training.

        Args:
            dataset_path: Path to the dataset
            dataset_name: Name for the dataset
            link_to_mlflow: Whether to link the dataset to the current MLflow run

        Returns:
            DVCTracker instance
        """
        # Track the dataset
        self.track_dataset(dataset_path, dataset_name)

        # Link to MLflow if requested
        if link_to_mlflow and run_id is not None:
            self.link_dataset_to_mlflow(
                dataset_path=dataset_path, dataset_name=dataset_name, run_id=run_id
            )

        return None
