"""
Data manager for storing and managing datasets in COCO format.
"""

import json
import logging
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
from tqdm import tqdm

from ..logging_config import get_logger
from ..transformations.transformation_pipeline import TransformationPipeline
from .dvc_manager import DVCConfig, DVCManager, DVCStorageType
from .path_manager import PathManager


class DataManager:
    """
    Manages dataset storage in COCO format with split-based organization.

    This class is responsible for:
    - Storing datasets in COCO format with split-based organization
    - Managing DVC integration for version control
    - Providing dataset information and statistics
    - Supporting dataset operations (list, delete, pull)
    """

    def __init__(
        self,
        path_manager: PathManager,
        enable_dvc: bool = False,
        dvc_config: Optional[DVCConfig] = None,
    ):
        """
        Initialize the data manager.

        Args:
            path_manager: PathManager instance for consistent path resolution
            enable_dvc: Whether to enable DVC integration
            dvc_config: DVC configuration (optional)
        """
        self.path_manager = path_manager
        self.data_dir = path_manager.data_dir

        # Setup logging
        self.logger = get_logger(__name__)

        # Initialize DVC integration
        self.dvc_manager = None
        if enable_dvc:
            try:
                # Use project root from path manager
                project_root = path_manager.project_root
                self.dvc_manager = DVCManager(project_root, dvc_config)
                self.logger.info("DVC integration enabled")
            except Exception as e:
                self.logger.warning(
                    f"DVC integration failed: {e}. Continuing without DVC."
                )

    def store_dataset(
        self,
        dataset_name: str,
        dataset_info: Dict[str, Any],
        split_data: Dict[str, Dict[str, Any]],
        track_with_dvc: bool = False,
        transformation_pipeline: Optional[TransformationPipeline] = None,
        processing_mode: str = "standard",
    ) -> str:
        """
        Store a dataset in COCO format with split-based organization.

        This method can handle different processing modes:
        - "streaming": Process one image at a time, save immediately
        - "batch": Process images in small batches for memory efficiency

        Args:
            dataset_name: Name of the dataset
            dataset_info: Common dataset metadata (classes, version, etc.)
            split_data: Dictionary mapping split names to COCO format data
            track_with_dvc: Whether to track the dataset with DVC
            transformation_pipeline: Optional transformation pipeline to apply
            processing_mode: Processing mode ('standard', 'streaming', 'batch')
            save_transformation_metadata: Whether to save transformation metadata

        Returns:
            Path to the stored dataset info file
        """

        assert processing_mode in ["streaming", "batch"], f"Received: {processing_mode}"

        # If no transformation pipeline, use standard storage
        if transformation_pipeline is None:
            return self._store_dataset_standard(
                dataset_name, dataset_info, split_data, track_with_dvc
            )
        elif len(transformation_pipeline) == 0:
            return self._store_dataset_standard(
                dataset_name, dataset_info, split_data, track_with_dvc
            )

        # Apply transformations based on processing mode
        elif processing_mode == "streaming":
            return self.store_dataset_with_batch_transformation(
                dataset_name,
                dataset_info,
                split_data,
                transformation_pipeline,
                track_with_dvc,
                batch_size=1,
            )
        else:
            print("Applying batch transformation")
            return self.store_dataset_with_batch_transformation(
                dataset_name,
                dataset_info,
                split_data,
                transformation_pipeline,
                track_with_dvc,
                batch_size=10,
            )

    def _setup_dataset_storage(
        self, dataset_name: str, dataset_info: Dict[str, Any]
    ) -> str:
        """
        Common setup for dataset storage.

        Args:
            dataset_name: Name of the dataset
            dataset_info: Dataset metadata

        Returns:
            Path to the dataset info file
        """
        self.path_manager.ensure_directories(dataset_name)

        # Store dataset info
        dataset_info_file = self.path_manager.get_dataset_info_file(dataset_name)
        with open(dataset_info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        return str(dataset_info_file)

    def _track_with_dvc(self, dataset_name: str) -> None:
        """
        Common DVC tracking logic.

        Args:
            dataset_name: Name of the dataset to track
        """
        if self.dvc_manager:
            try:
                dataset_path = self.path_manager.get_dataset_dir(dataset_name)
                if self.dvc_manager.add_data_to_dvc(dataset_path, dataset_name):
                    self.logger.info(f"Dataset '{dataset_name}' tracked with DVC")
                else:
                    self.logger.warning(
                        f"Failed to track dataset '{dataset_name}' with DVC"
                    )
            except Exception as e:
                self.logger.error(f"Error tracking dataset with DVC: {e}")

    def _load_transform_one_image(
        self,
        image_info: Dict[str, Any],
        split_coco_data: Dict[str, Any],
        transformation_pipeline: Optional[TransformationPipeline] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image with optional transformations.

        Args:
            image_info: Image metadata
            split_coco_data: Split data containing annotations
            transformation_pipeline: Optional transformation pipeline

        Returns:
            Dictionary with processed data or None if processing failed
        """
        # Load image
        image_path = Path(image_info["file_name"])
        if not image_path.exists():
            print(f"Could not load image: {image_path}")
            return None

        # Get annotations for this image
        image_annotations = [
            ann
            for ann in split_coco_data["annotations"]
            if ann["image_id"] == image_info["id"]
        ]

        image = cv2.imread(str(image_path))
        # If no transformation pipeline, return original data
        if transformation_pipeline is None:
            return {
                "image": image,
                "annotations": image_annotations,
                "info": image_info,
                "transformed": False,
            }

        # Apply transformations
        try:
            inputs = {
                "image": image,
                "annotations": image_annotations,
                "info": image_info,
            }
            transformed_data = transformation_pipeline.transform(inputs)
            return {"transformed_data": transformed_data, "transformed": True}

        except Exception as e:
            self.logger.error(
                f"Error transforming image {image_info['file_name']}: {str(e)}"
            )
            return None

    def _save_coco_json(
        self, dataset_name: str, split_name: str, split_data_to_store: Dict[str, Any]
    ) -> str:
        """
        Save split data to file.

        Args:
            dataset_name: Name of the dataset
            split_name: Name of the split
            split_data_to_store: Data to store

        Returns:
            Path to the saved annotations file
        """
        split_annotations_file = self.path_manager.get_dataset_split_annotations_file(
            dataset_name, split_name
        )

        with open(split_annotations_file, "w") as f:
            json.dump(split_data_to_store, f, indent=2)

        return str(split_annotations_file)

    def _update_annotation_ids_and_clean(
        self, coco_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update annotation IDs to match new image IDs and discard empty annotations.

        Args:
            annotations: List of annotation dictionaries
            images: List of image info dictionaries

        Returns:
            Cleaned and updated annotations list
        """
        # Create mapping from old image IDs to new image IDs
        cleaned_coco_json = coco_json.copy()
        images = coco_json["images"]
        annotations = coco_json["annotations"]
        assert len(images) == len(
            annotations
        ), "Number of images and annotations must be the same"
        new_images = []
        new_annotations = []

        for new_id, (image_info, annotation) in enumerate(
            zip(images, annotations), start=1
        ):
            image_info["id"] = new_id
            new_images.append(image_info)
            if len(annotation) > 0:
                for ann in annotation:
                    ann["image_id"] = new_id
                    ann["id"] = str(new_id) + "_" + str(uuid4())
                new_annotations.extend(annotation)

        cleaned_coco_json["images"] = new_images
        cleaned_coco_json["annotations"] = new_annotations

        return cleaned_coco_json

    def _is_annotation_valid(self, annotation: Dict[str, Any]) -> bool:
        """
        Check if an annotation is valid (not empty).

        Args:
            annotation: Annotation dictionary

        Returns:
            True if annotation is valid, False if empty/invalid
        """
        # Check for bounding box
        if "bbox" in annotation and annotation["bbox"]:
            bbox = annotation["bbox"]
            if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                # Check if bbox has valid dimensions
                if bbox[2] > 0 and bbox[3] > 0:  # width and height > 0
                    return True

        # Check for segmentation
        if "segmentation" in annotation and annotation["segmentation"]:
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list) and len(segmentation) > 0:
                return True

        # Check for keypoints
        if "keypoints" in annotation and annotation["keypoints"]:
            keypoints = annotation["keypoints"]
            if isinstance(keypoints, list) and len(keypoints) > 0:
                return True

        # If none of the above, consider it empty
        return False

    def _get_transformation_metadata(
        self, transformation_pipeline: TransformationPipeline
    ) -> Dict[str, Any]:
        """
        Get transformation metadata.

        Args:
            transformation_pipeline: Transformation pipeline

        Returns:
            Transformation metadata dictionary
        """
        return {
            "pipeline_info": transformation_pipeline.get_pipeline_info(),
            "transformation_history": transformation_pipeline.get_transformation_history(),
        }

    def _store_dataset_standard(
        self,
        dataset_name: str,
        dataset_info: Dict[str, Any],
        split_data: Dict[str, Dict[str, Any]],
        track_with_dvc: bool = False,
    ) -> str:
        """
        Store dataset without transformations (original method).
        """
        dataset_info_file = self._setup_dataset_storage(dataset_name, dataset_info)

        # Store each split
        for split_name, split_coco_data in split_data.items():
            # Store split annotations
            split_annotations_file = (
                self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split_name
                )
            )

            # Copy images for this split & Update the file path in the annotations
            split_coco_data = self._copy_images_for_split(
                dataset_name, split_name, split_coco_data
            )

            with open(split_annotations_file, "w") as f:
                json.dump(split_coco_data, f, indent=2)

        self.logger.info(f"Stored dataset '{dataset_name}' in COCO format")

        # Track with DVC if enabled
        if track_with_dvc and self.dvc_manager:
            self._track_with_dvc(dataset_name)

        return str(dataset_info_file)

    def store_dataset_with_batch_transformation(
        self,
        dataset_name: str,
        dataset_info: Dict[str, Any],
        split_data: Dict[str, Dict[str, Any]],
        transformation_pipeline: TransformationPipeline,
        track_with_dvc: bool = False,
        batch_size: int = 10,
    ) -> str:
        """
        Store dataset with batch transformation processing.

        This method processes images in small batches to balance memory usage
        and performance for medium-sized datasets.

        Args:
            dataset_name: Name of the dataset
            dataset_info: Common dataset metadata (classes, version, etc.)
            split_data: Dictionary mapping split names to COCO format data
            transformation_pipeline: Transformation pipeline to apply
            track_with_dvc: Whether to track the dataset with DVC
            batch_size: Number of images to process in each batch

        Returns:
            Path to the stored dataset info file
        """

        dataset_info_file = self._setup_dataset_storage(dataset_name, dataset_info)

        # Process each split

        for split_name, split_coco_data in split_data.items():
            self.logger.info(
                f"Processing split: {split_name} with batch transformation"
            )

            # Initialize storage for this split
            split_images_dir = self.path_manager.get_dataset_split_images_dir(
                dataset_name, split_name
            )
            split_images_dir.mkdir(parents=True, exist_ok=True)

            # Process images in batches
            images_infos = split_coco_data["images"]
            batch_data_to_store = []
            for i in tqdm(
                range(0, len(images_infos), batch_size),
                desc=f"Processing batches for {split_name}",
            ):
                batch_images_infos = images_infos[i : i + batch_size]
                batch_data = self._process_single_batch(
                    batch_images_infos,
                    split_coco_data,
                    dataset_name,
                    split_images_dir,
                    transformation_pipeline,
                )
                batch_data_to_store.append(batch_data)

            # merge split data
            split_data_to_store = {
                "annotations": [],
                "categories": split_coco_data.get("categories", []),
                "images": [],
            }
            for data in batch_data_to_store:
                split_data_to_store["annotations"].extend(data["annotations"])
                split_data_to_store["images"].extend(data["images"])

            # Clean and update annotation IDs before saving
            split_data_to_store = self._update_annotation_ids_and_clean(
                split_data_to_store
            )

            # save_transformation_metadata
            transformation_metadata = {
                "transformation_history": transformation_pipeline.get_transformation_history(),
            }
            # split_data_to_store["transformation_metadata"] = transformation_metadata

            # Save split data using helper method
            self._save_coco_json(dataset_name, split_name, split_data_to_store)

            self.logger.info(
                f"Stored dataset {split_name} split of '{dataset_name}' with batch transformation"
            )

        # Track with DVC if enabled
        if track_with_dvc and self.dvc_manager:
            self._track_with_dvc(dataset_name)

        return str(dataset_info_file)

    def _process_single_batch(
        self,
        batch_images_infos: List[Dict[str, Any]],
        split_coco_data: Dict[str, Any],
        dataset_name: str,
        split_images_dir: Path,
        transformation_pipeline: Optional[TransformationPipeline] = None,
    ) -> Dict[str, Any]:
        """
        Process a single batch of images.
        """
        transformed_images = []
        transformed_annotations = []
        transformed_image_infos = []

        # process in batch
        for image_info in batch_images_infos:
            # Process image using helper method
            result = self._load_transform_one_image(
                image_info, split_coco_data, transformation_pipeline
            )

            if result is None:
                continue

            if result["transformed"]:
                # Handle transformed data
                for data in result["transformed_data"]:
                    transformed_images.append(data["image"])
                    transformed_annotations.append(data["annotations"])
                    transformed_image_infos.append(data["info"])
            else:
                # Handle original data (no transformations)
                transformed_images.append(result["image"])
                transformed_annotations.append(result["annotations"])
                transformed_image_infos.append([result["info"]])

        # Store split data
        split_data_to_store = {
            "annotations": transformed_annotations,
            "images": transformed_image_infos,
        }

        new_paths = self._copy_or_save_batch_images(
            transformed_images, transformed_image_infos, dataset_name, split_images_dir
        )

        for image_info, new_path in zip(split_data_to_store["images"], new_paths):
            image_info["file_name"] = new_path

        # split_data_to_store["images"] = transformed_image_infos

        return split_data_to_store

    def _copy_or_save_batch_images(
        self,
        transformed_images: List[np.ndarray],
        transformed_image_infos: List[Dict[str, Any]],
        dataset_name: str,
        split_images_dir: Path,
    ) -> List[str]:
        def func(image_info, image_data):
            return self._copy_or_save_one_image(
                image_info=image_info,
                split_images_dir=split_images_dir,
                dataset_name=dataset_name,
                image_data=image_data,
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            new_paths = [
                p
                for p in executor.map(func, transformed_image_infos, transformed_images)
            ]

        return new_paths

    def _copy_or_save_one_image(
        self,
        image_info: Dict[str, Any],
        split_images_dir: Path,
        dataset_name: str,
        image_data: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        # Extract original image path
        original_path = image_info.get("file_name", "")
        if not original_path:
            print(f"Image file_name is not set for image {image_info}")
            return None

        # Try to find the image file
        original_file = Path(original_path)
        if (not original_file.exists()) and (image_data is None):
            # Try relative to the annotation file directory
            annotation_dir = self.path_manager.get_dataset_annotations_dir(dataset_name)
            original_file = annotation_dir.parent / original_path
            if not original_file.exists():
                print(f"Image file not found: {original_path}")
                return None

        # Copy image to split directory
        filename = original_file.name
        new_path = split_images_dir / filename

        if image_data is None:
            if not new_path.exists():
                shutil.copy2(original_file, new_path)
        else:
            cv2.imwrite(str(new_path), image_data)

        # path to the new image relative to the data_dir
        relpath = os.path.relpath(new_path, start=self.path_manager.data_dir)
        return Path(relpath).as_posix()

    def _copy_images_for_split(
        self, dataset_name: str, split_name: str, split_coco_data: Dict[str, Any]
    ):
        """Copy images for a specific split to the dataset directory."""
        infos = split_coco_data.get("images", [])

        # Create split images directory
        split_images_dir = self.path_manager.get_dataset_split_images_dir(
            dataset_name, split_name
        )
        split_images_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=3) as executor:
            func = partial(
                self._copy_or_save_one_image,
                split_images_dir=split_images_dir,
                dataset_name=dataset_name,
            )
            for i, new_path in enumerate(executor.map(func, infos)):
                if new_path is not None:
                    infos[i]["file_name"] = new_path

        return split_coco_data

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        dataset_info_file = self.path_manager.get_dataset_info_file(dataset_name)

        if not dataset_info_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")

        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        # Get split information
        existing_splits = self.path_manager.get_existing_splits(dataset_name)
        images_by_split = {}
        annotations_by_split = {}

        for split in existing_splits:
            split_annotations_file = (
                self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split
                )
            )
            if split_annotations_file.exists():
                with open(split_annotations_file, "r") as f:
                    split_data = json.load(f)
                    images_by_split[split] = len(split_data.get("images", []))
                    annotations_by_split[split] = len(split_data.get("annotations", []))

        # Get DVC information if available
        dvc_info = {}
        if self.dvc_manager:
            try:
                dvc_status = self.dvc_manager.get_status()
                dvc_info = {
                    "dvc_enabled": True,
                    "dvc_status": dvc_status,
                }
            except Exception as e:
                dvc_info = {
                    "dvc_enabled": True,
                    "dvc_error": str(e),
                }
        else:
            dvc_info = {"dvc_enabled": False}

        return {
            "dataset_name": dataset_name,
            "dataset_info": dataset_info,
            "splits": existing_splits,
            "images_by_split": images_by_split,
            "annotations_by_split": annotations_by_split,
            "total_images": sum(images_by_split.values()),
            "total_annotations": sum(annotations_by_split.values()),
            "dvc_info": dvc_info,
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets in data storage.

        Returns:
            List of dataset information dictionaries
        """
        datasets = []

        # Use PathManager to list datasets
        for dataset_name in self.path_manager.list_datasets():
            try:
                dataset_info = self.get_dataset_info(dataset_name)
                datasets.append(dataset_info)
            except Exception as e:
                self.logger.warning(f"Error reading dataset {dataset_name}: {str(e)}")

        return datasets

    def load_dataset_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load all data for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset info and split data
        """
        dataset_info_file = self.path_manager.get_dataset_info_file(dataset_name)

        if not dataset_info_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")

        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        # Load all split data
        split_data = {}
        existing_splits = self.path_manager.get_existing_splits(dataset_name)

        for split in existing_splits:
            split_annotations_file = (
                self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split
                )
            )
            if split_annotations_file.exists():
                with open(split_annotations_file, "r") as f:
                    split_data[split] = json.load(f)

        return {"dataset_info": dataset_info, "split_data": split_data}

    def delete_dataset(self, dataset_name: str, remove_from_dvc: bool = True) -> bool:
        """
        Delete a dataset from data storage.

        Args:
            dataset_name: Name of the dataset to delete
            remove_from_dvc: Whether to remove from DVC tracking

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            dataset_dir = self.path_manager.get_dataset_dir(dataset_name)
            if not dataset_dir.exists():
                self.logger.warning(f"Dataset '{dataset_name}' not found")
                return False

            # Remove from DVC if enabled
            if remove_from_dvc and self.dvc_manager:
                try:
                    # Note: This method may not exist in DVCManager, handle gracefully
                    if hasattr(self.dvc_manager, "remove_data_from_dvc"):
                        self.dvc_manager.remove_data_from_dvc(dataset_name)
                        self.logger.info(f"Removed dataset '{dataset_name}' from DVC")
                    else:
                        self.logger.warning(
                            "DVC remove_data_from_dvc method not available"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to remove dataset from DVC: {e}")

            # Remove dataset directory
            shutil.rmtree(dataset_dir)
            self.logger.info(f"Deleted dataset '{dataset_name}'")

            return True

        except Exception as e:
            self.logger.error(f"Error deleting dataset '{dataset_name}': {e}")
            return False

    def pull_dataset(self, dataset_name: str) -> bool:
        """
        Pull a dataset from DVC remote storage.

        Args:
            dataset_name: Name of the dataset to pull

        Returns:
            True if pull successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.pull_data(dataset_name)
        except Exception as e:
            self.logger.error(f"Error pulling dataset '{dataset_name}': {e}")
            return False

    def setup_remote_storage(
        self, storage_type: DVCStorageType, storage_path: str, force: bool = False
    ) -> bool:
        """
        Setup remote storage for DVC.

        Args:
            storage_type: Type of storage (local, s3, gcs, etc.)
            storage_path: Path to remote storage
            force: Whether to force setup even if already configured

        Returns:
            True if setup successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.setup_remote_storage(
                storage_type, storage_path, force
            )
        except Exception as e:
            self.logger.error(f"Error setting up remote storage: {e}")
            return False

    def get_dvc_status(self) -> Dict[str, Any]:
        """
        Get DVC status information.

        Returns:
            Dictionary with DVC status information
        """
        if not self.dvc_manager:
            return {"dvc_enabled": False}

        try:
            return {
                "dvc_enabled": True,
                "status": self.dvc_manager.get_status(),
                "config": self.dvc_manager.get_config(),
            }
        except Exception:
            return {
                "dvc_enabled": True,
                "error": traceback.format_exc(),
            }

    def create_data_pipeline(
        self, pipeline_name: str, stages: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a data pipeline configuration.

        Args:
            pipeline_name: Name of the pipeline
            stages: List of pipeline stages

        Returns:
            True if creation successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.create_pipeline(pipeline_name, stages)
        except Exception as e:
            self.logger.error(f"Error creating data pipeline: {e}")
            return False

    def run_data_pipeline(self, pipeline_name: str) -> bool:
        """
        Run a data pipeline.

        Args:
            pipeline_name: Name of the pipeline to run

        Returns:
            True if pipeline execution successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.run_pipeline(pipeline_name)
        except Exception as e:
            self.logger.error(f"Error running data pipeline: {e}")
            return False
