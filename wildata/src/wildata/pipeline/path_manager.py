"""
Centralized path management for the data pipeline.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ROOT
from ..logging_config import get_logger


class PathManager:
    """
    Centralized path management for the data pipeline.

    Provides consistent path resolution and eliminates hardcoded paths.
    Uses COCO-First design with split-based organization.
    """

    def __init__(self, root_data_directory: Path):
        """
        Initialize the path manager.
        """
        self.project_root = Path(root_data_directory)
        self.logger = get_logger(__name__)

        # Define standard directory structure
        self._setup_directory_structure()
        self._list_frameworks = ["coco", "yolo", "roi"]
        self._framework_formats_dir = {
            "coco": self.coco_formats_dir,
            "yolo": self.yolo_formats_dir,
            "roi": self.roi_formats_dir,
        }
        self.yolo_data_yaml_name = "data.yaml"

    def _setup_directory_structure(self):
        """Setup the standard directory structure."""
        # Data storage (COCO-First design)
        self.data_dir = self.project_root / "datasets"

        # Framework formats
        self.framework_formats_dir = self.project_root / "framework_formats"
        self.coco_formats_dir = self.framework_formats_dir / "coco"
        self.yolo_formats_dir = self.framework_formats_dir / "yolo"
        self.roi_formats_dir = self.framework_formats_dir / "roi"

        # DVC and configuration
        self.dvc_dir = ROOT / ".dvc"
        self.config_dir = ROOT / "config"

    def get_framework_dir(self, framework: str) -> Path:
        """Get the framework directory."""
        if framework.lower() in self._list_frameworks:
            return self._framework_formats_dir[framework.lower()]
        else:
            raise ValueError(
                f"Unsupported framework: {framework} in {self._list_frameworks}"
            )

    def get_dataset_dir(self, dataset_name: str) -> Path:
        """Get the data directory for a specific dataset."""
        return self.data_dir / dataset_name

    def get_dataset_annotations_dir(self, dataset_name: str) -> Path:
        """Get the annotations directory for a specific dataset."""
        return self.get_dataset_dir(dataset_name) / "annotations"

    def get_dataset_split_annotations_file(self, dataset_name: str, split: str) -> Path:
        """Get the annotations file path for a specific dataset and split."""
        return self.get_dataset_annotations_dir(dataset_name) / f"{split}.json"

    def get_dataset_split_images_dir(self, dataset_name: str, split: str) -> Path:
        """Get the images directory for a specific dataset and split."""
        return self.get_dataset_dir(dataset_name) / "images" / split

    def get_dataset_info_file(self, dataset_name: str) -> Path:
        """Get the dataset info file path for a specific dataset."""
        return self.get_dataset_dir(dataset_name) / "dataset_info.json"

    def get_framework_format_dir(self, dataset_name: str, framework: str) -> Path:
        """Get the framework format directory for a dataset."""
        if framework.lower() in self._list_frameworks:
            return self.framework_formats_dir / framework / dataset_name
        else:
            raise ValueError(
                f"Unsupported framework: {framework} in {self._list_frameworks}"
            )

    def get_framework_images_dir(self, dataset_name: str, framework: str) -> Path:
        """Get the images directory for a framework format."""
        framework_dir = self.get_framework_format_dir(dataset_name, framework)
        if framework.lower() in self._list_frameworks:
            return framework_dir / "images"
        else:
            raise ValueError(
                f"Unsupported framework: {framework} in {self._list_frameworks}"
            )

    def get_framework_annotations_dir(self, dataset_name: str, framework: str) -> Path:
        """Get the annotations directory for a framework format."""
        framework_dir = self.get_framework_format_dir(dataset_name, framework)
        if framework.lower() == "coco":
            return framework_dir / "annotations"
        elif framework.lower() == "yolo":
            return framework_dir / "labels"
        elif framework.lower() == "roi":
            return framework_dir / "labels"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def ensure_directories(self, dataset_name: str, frameworks: Optional[list] = None):
        """Ensure all necessary directories exist."""
        # Dataset directories
        self.get_dataset_dir(dataset_name).mkdir(parents=True, exist_ok=True)
        self.get_dataset_annotations_dir(dataset_name).mkdir(
            parents=True, exist_ok=True
        )

        # Framework directories
        if frameworks:
            for framework in frameworks:
                framework_dir = self.get_framework_format_dir(dataset_name, framework)
                framework_dir.mkdir(parents=True, exist_ok=True)

                # Create split directories only for existing splits
                # Get existing splits from data
                existing_splits = self.get_existing_splits(dataset_name)

                for split in existing_splits:
                    self.get_framework_split_image_dir(
                        dataset_name, framework, split
                    ).mkdir(parents=True, exist_ok=True)
                    self.get_framework_split_annotations_dir(
                        dataset_name, framework, split
                    ).mkdir(parents=True, exist_ok=True)

    def get_existing_splits(self, dataset_name: str) -> List[str]:
        """
        Get list of splits that actually exist in the data.
        """
        try:
            annotations_dir = self.get_dataset_annotations_dir(dataset_name)
            if not annotations_dir.exists():
                return []

            existing_splits = []
            for annotation_file in annotations_dir.glob("*.json"):
                if annotation_file.name != "dataset_info.json":
                    split_name = annotation_file.stem  # Remove .json extension
                    existing_splits.append(split_name)

            return sorted(existing_splits)
        except Exception as e:
            self.logger.warning(
                f"Error getting existing splits for dataset '{dataset_name}': {e}"
            )
            return []

    def get_framework_split_image_dir(
        self, dataset_name: str, framework: str, split: str
    ) -> Path:
        """Get the images directory for a specific split."""
        return self.get_framework_images_dir(dataset_name, framework) / split

    def get_framework_split_annotations_dir(
        self, dataset_name: str, framework: str, split: str
    ) -> Path:
        """Get the annotations directory for a specific split."""
        return self.get_framework_annotations_dir(dataset_name, framework) / split

    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists in data storage."""
        return self.get_dataset_info_file(dataset_name).exists()

    def framework_format_exists(self, dataset_name: str, framework: str) -> bool:
        """Check if a framework format exists for a dataset."""
        return self.get_framework_format_dir(dataset_name, framework).exists()

    def list_datasets(self) -> list:
        """List all available datasets."""
        datasets = []
        if self.data_dir.exists():
            for dataset_dir in self.data_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_info_file = dataset_dir / "dataset_info.json"
                    if dataset_info_file.exists():
                        datasets.append(dataset_dir.name)
        return datasets

    def list_framework_datasets(self, framework: str) -> list:
        """List all available datasets for a framework."""
        datasets = []
        framework_dir = self._framework_formats_dir.get(framework)
        if framework_dir:
            for dataset_dir in framework_dir.iterdir():
                if dataset_dir.is_dir():
                    datasets.append(dataset_dir.name)
        else:
            raise ValueError(f"Framework directory {framework_dir} does not exist")

        return datasets

    def list_framework_formats(self, dataset_name: str) -> Dict[str, bool]:
        """List available framework formats for a dataset."""
        formats = {}
        for framework in self._list_frameworks:
            formats[framework] = self.framework_format_exists(dataset_name, framework)
        return formats
