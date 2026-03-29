"""
Data pipeline for importing, transforming, and exporting datasets.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ROIConfig
from ..logging_config import get_logger
from ..transformations.transformation_pipeline import TransformationPipeline
from .data_manager import DataManager
from .framework_data_manager import FrameworkDataManager
from .loader import Loader
from .path_manager import PathManager


class DataPipeline:
    """
    High-level data pipeline for managing dataset operations.

    This class orchestrates the complete data workflow:
    - Import datasets from various formats (COCO, YOLO)
    - Apply transformations and augmentations
    - Store data in COCO format with split-based organization
    - Export to framework-specific formats
    - Manage DVC integration for version control
    """

    def __init__(
        self,
        root: str,
        split_name: str,
        transformation_pipeline: Optional[TransformationPipeline] = None,
        enable_dvc: bool = True,
    ):
        """
        Initialize the data pipeline.

        Args:
            root: Root directory for data storage
            transformation_pipeline: Optional transformation pipeline
            enable_dvc: Whether to enable DVC integration
        """
        self.root = Path(root)
        self.logger = get_logger(self.__class__.__name__)

        assert split_name in [
            "train",
            "val",
            "test",
        ], f"Invalid split name: {split_name}"

        self.split_name = split_name

        # Initialize path manager for consistent path resolution
        self.path_manager = PathManager(self.root)

        # Initialize transformation pipeline
        self.transformation_pipeline = (
            transformation_pipeline or TransformationPipeline()
        )

        # Initialize data manager with DVC support
        self.data_manager = DataManager(self.path_manager, enable_dvc=enable_dvc)

        # Initialize framework data manager
        self.framework_data_manager = FrameworkDataManager(self.path_manager)

    def import_dataset(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        processing_mode: str = "batch",  # "streaming", "batch"
        track_with_dvc: bool = False,
        bbox_tolerance: int = 5,
        roi_config: Optional[ROIConfig] = None,
        dotenv_path: Optional[str] = None,
        ls_xml_config: Optional[str] = None,
        ls_parse_config: bool = False,
    ) -> Dict[str, Any]:
        """
        Import dataset with enhanced transformation-storage integration.

        Args:
            source_path: Path to source dataset
            source_format: Format of source dataset ('coco' or 'yolo')
            dataset_name: Name for the dataset in COCO format
            processing_mode: Processing mode ('streaming', 'batch')
            apply_transformations: Whether to apply transformations during import
            track_with_dvc: Whether to track the dataset with DVC
            bbox_tolerance: Tolerance for bbox validation
            roi_config: ROI configuration
            dotenv_path: Path to .env file
            ls_xml_config: Path to Label Studio XML config file. If given, then parse_ls_config must be False
            ls_parse_config: Whether to parse Label Studio config

        Returns:
            Dictionary with import result information
        """
        self.logger.debug(
            f"Starting import_dataset_with_options: {source_path}, {source_format}, {dataset_name}, mode={processing_mode}"
        )

        assert processing_mode in [
            "streaming",
            "batch",
        ], f"Invalid processing mode: {processing_mode}"

        try:
            self.logger.info(
                f"Importing dataset from {source_path} ({source_format} format) with {processing_mode} mode"
            )

            # Load and validate dataset
            loader = Loader()
            dataset_info, split_data = loader.load(
                source_path=source_path,
                source_format=source_format,
                dataset_name=dataset_name,
                bbox_tolerance=bbox_tolerance,
                split_name=self.split_name,
                dotenv_path=dotenv_path,
                ls_xml_config=ls_xml_config,
                ls_parse_config=ls_parse_config,
            )

            if not dataset_info or not split_data:
                return {
                    "success": False,
                    "error": "Failed to load and validate dataset",
                    "validation_errors": [],
                    "hints": [],
                }

            if roi_config:
                self.framework_data_manager.create_roi_format(
                    dataset_name=dataset_name,
                    coco_data=split_data[self.split_name],
                    split=self.split_name,
                    roi_config=roi_config,
                )

            # Store dataset using data manager with transformation options
            self.logger.debug("Storing dataset with data manager")
            dataset_info_path = self.data_manager.store_dataset(
                dataset_name=dataset_name,
                dataset_info=dataset_info,
                split_data=split_data,
                track_with_dvc=track_with_dvc,
                transformation_pipeline=self.transformation_pipeline,
                processing_mode=processing_mode,
            )

            # Create framework formats
            self.logger.debug("Creating framework formats")
            framework_paths = self.framework_data_manager.create_framework_formats(
                dataset_name
            )

            self.logger.info(
                f"Successfully imported dataset '{dataset_name}' with {processing_mode} mode"
            )
            return {
                "success": True,
                "dataset_name": dataset_name,
                "dataset_info_path": dataset_info_path,
                "framework_paths": framework_paths,
                "processing_mode": processing_mode,
                "dvc_tracked": track_with_dvc
                and self.data_manager.dvc_manager is not None,
            }

        except Exception as e:
            self.logger.error(f"Error importing dataset: {str(e)}")
            self.logger.error(
                f"Exception in import_dataset_with_options: {traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": str(e),
                "validation_errors": [],
                "hints": [],
            }

    def export_dataset(
        self, dataset_name: str, target_format: str, target_path: str
    ) -> bool:
        """
        Export a dataset from COCO format to target format.

        Args:
            dataset_name: Name of the dataset in COCO format
            target_format: Target format ('coco' or 'yolo')
            target_path: Path where to export the dataset

        Returns:
            True if export successful, False otherwise
        """
        try:
            # Load dataset data
            dataset_data = self.data_manager.load_dataset_data(dataset_name)

            if target_format == "coco":
                # Export COCO format
                return self._export_coco_format(dataset_data, target_path)
            elif target_format == "yolo":
                # Export YOLO format
                return self._export_yolo_format(dataset_data, target_path)
            else:
                self.logger.error(f"Unsupported target format: {target_format}")
                return False

        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            return False

    def _export_coco_format(
        self, dataset_data: Dict[str, Any], target_path: str
    ) -> bool:
        """Export dataset to COCO format."""
        try:
            target_path_obj = Path(target_path)
            target_path_obj.mkdir(parents=True, exist_ok=True)

            # Export each split
            for split_name, split_data in dataset_data["split_data"].items():
                split_file = target_path_obj / f"{split_name}.json"
                with open(split_file, "w") as f:
                    json.dump(split_data, f, indent=2)

            # Export dataset info
            info_file = target_path_obj / "dataset_info.json"
            with open(info_file, "w") as f:
                json.dump(dataset_data["dataset_info"], f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Error exporting COCO format: {str(e)}")
            return False

    def _export_yolo_format(
        self, dataset_data: Dict[str, Any], target_path: str
    ) -> bool:
        """Export dataset to YOLO format."""
        try:
            # Use framework data manager to convert COCO to YOLO
            result = self.framework_data_manager.export_framework_format(
                dataset_data["dataset_info"]["name"], "yolo"
            )
            return "path" in result
        except Exception as e:
            self.logger.error(f"Error exporting YOLO format: {str(e)}")
            return False

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the data pipeline.

        Returns:
            Dictionary with pipeline status information
        """
        return {
            "root_directory": str(self.root),
            "dvc_enabled": self.data_manager.dvc_manager is not None,
            "transformation_pipeline": self.transformation_pipeline.get_pipeline_info()
            if self.transformation_pipeline
            else None,
            "datasets": self.list_datasets(),
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.

        Returns:
            List of dataset information dictionaries
        """
        return self.data_manager.list_datasets()

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        return self.data_manager.get_dataset_info(dataset_name)

    def delete_dataset(self, dataset_name: str, remove_from_dvc: bool = True) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_name: Name of the dataset to delete
            remove_from_dvc: Whether to remove from DVC tracking

        Returns:
            True if deletion successful, False otherwise
        """
        return self.data_manager.delete_dataset(dataset_name, remove_from_dvc)

    def export_framework_format(
        self, dataset_name: str, framework: str
    ) -> Dict[str, Any]:
        """
        Export a dataset to a specific framework format.

        Args:
            dataset_name: Name of the dataset
            framework: Framework name ('coco' or 'yolo')

        Returns:
            Dictionary with export information
        """
        return self.framework_data_manager.export_framework_format(
            dataset_name, framework
        )

    def list_framework_formats(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        List available framework formats for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of framework format information
        """
        return self.framework_data_manager.list_framework_formats(dataset_name)
