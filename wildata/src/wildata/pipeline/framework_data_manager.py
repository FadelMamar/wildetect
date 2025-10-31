"""
Framework data manager for creating framework-specific formats.
"""

import json
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..adapters.roi_adapter import ROIAdapter
from ..adapters.yolo_adapter import YOLOAdapter
from ..config import ROIConfig
from .path_manager import PathManager

logger = logging.getLogger(__name__)


class FrameworkDataManager:
    """
    Manages creation of framework-specific formats from COCO storage.

    This class is responsible for:
    - Creating COCO format exports (symlinks/copies from COCO storage)
    - Creating YOLO format exports using YOLOAdapter
    - Managing framework format directories and files
    """

    def __init__(self, path_manager: PathManager):
        """
        Initialize the framework data manager.

        Args:
            path_manager: PathManager instance for consistent path resolution
        """
        self.path_manager = path_manager
        self.logger = logging.getLogger(__name__)

    def create_framework_formats(self, dataset_name: str) -> Dict[str, str]:
        """
        Create framework formats for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary mapping framework names to their paths
        """
        framework_paths = {}

        # Create YOLO format
        try:
            yolo_path = self._create_yolo_format(dataset_name)
            framework_paths["yolo"] = yolo_path
        except Exception as e:
            self.logger.error(f"Error creating YOLO format: {traceback.format_exc()}")
            raise e

        return framework_paths

    def _create_coco_format(self, dataset_name: str) -> str:
        """Create COCO format for a dataset (symlink/copy from COCO storage)."""
        # Ensure directories exist
        self.path_manager.ensure_directories(dataset_name, ["coco"])

        # Get paths using PathManager
        coco_dir = self.path_manager.get_framework_format_dir(dataset_name, "coco")
        coco_data_dir = self.path_manager.get_framework_images_dir(dataset_name, "coco")
        coco_annotations_dir = self.path_manager.get_framework_annotations_dir(
            dataset_name, "coco"
        )

        # Symlink/copy images and annotations for each split
        existing_splits = self.path_manager.get_existing_splits(dataset_name)
        for split in existing_splits:
            # Images
            src_images_dir = self.path_manager.get_dataset_split_images_dir(
                dataset_name, split
            )
            dst_images_dir = coco_data_dir / split
            dst_images_dir.mkdir(parents=True, exist_ok=True)
            self._symlink_or_copy_dir(src_images_dir, dst_images_dir)

            # Annotations
            src_ann_file = self.path_manager.get_dataset_split_annotations_file(
                dataset_name, split
            )
            dst_ann_file = coco_annotations_dir / f"{split}.json"
            dst_ann_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_ann_file.exists():
                dst_ann_file.unlink()
            shutil.copy2(src_ann_file, dst_ann_file)

        # Copy dataset_info.json
        src_info_file = self.path_manager.get_dataset_info_file(dataset_name)
        dst_info_file = coco_dir / "dataset_info.json"
        shutil.copy2(src_info_file, dst_info_file)

        self.logger.info(f"Created COCO format for dataset '{dataset_name}'")
        return str(coco_dir)

    def _create_yolo_format(self, dataset_name: str) -> str:
        """Create YOLO format for a dataset."""
        # Ensure directories exist
        self.path_manager.ensure_directories(dataset_name, ["yolo"])

        # Get paths using PathManager
        yolo_dir = self.path_manager.get_framework_format_dir(dataset_name, "yolo")

        # Create symlinks for images
        self._create_image_symlinks(dataset_name, "yolo")

        # Generate YOLO annotations using adapter for each split
        existing_splits = self.path_manager.get_existing_splits(dataset_name)
        dataset_info = self._load_dataset_info(dataset_name)

        all_yolo_data = {"annotations": {}, "names": {}}

        for split in existing_splits:
            try:
                # Load split COCO data
                split_ann_file = self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split
                )
                if not split_ann_file.exists():
                    continue

                with open(split_ann_file, "r") as f:
                    split_coco_data = json.load(f)

                # Create adapter for this split
                adapter = YOLOAdapter(coco_data=split_coco_data)

                # Convert to YOLO format
                split_yolo = adapter.convert()
                all_yolo_data["annotations"][split] = split_yolo

                # Get class names from first split
                if not all_yolo_data["names"]:
                    classes = dataset_info.get("classes", [])
                    all_yolo_data["names"] = {cat["id"]: cat["name"] for cat in classes}

            except Exception as e:
                self.logger.warning(
                    f"Could not convert split '{split}': {traceback.format_exc()}"
                )
                raise e

        # Save YOLO annotations and data.yaml
        self._save_yolo_annotations(dataset_name, all_yolo_data)
        self._save_yolo_data_yaml(dataset_name, all_yolo_data)

        self.logger.info(f"Created YOLO format for dataset '{dataset_name}'")
        return str(yolo_dir)

    def create_roi_format(
        self,
        dataset_name: str,
        roi_config: ROIConfig,
        coco_data: Dict[str, Any],
        split: str,
        draw_original_bboxes: bool = False,
    ) -> str:
        """Create ROI format for a dataset."""
        # Ensure directories exist
        self.path_manager.ensure_directories(dataset_name, ["roi"])

        try:
            # Create adapter for this split
            adapter = ROIAdapter.from_config(coco_data=coco_data, config=roi_config)
            roi_data = adapter.convert()

            images_dir = self.path_manager.get_framework_split_image_dir(
                dataset_name, "roi", split
            )
            images_dir.mkdir(parents=True, exist_ok=True)

            labels_dir = self.path_manager.get_framework_split_annotations_dir(
                dataset_name, "roi", split
            )
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Save ROI data
            adapter.save(
                roi_data,
                output_labels_dir=labels_dir,
                output_images_dir=images_dir,
                draw_original_bboxes=draw_original_bboxes,
            )

        except Exception as e:
            self.logger.warning(
                f"Could not convert split '{split}' to ROI format: {traceback.format_exc()}"
            )
            raise e

        self.logger.info(f"Created ROI format for dataset '{dataset_name}'")
        return str(self.path_manager.get_framework_format_dir(dataset_name, "roi"))

    def _create_image_symlinks(self, dataset_name: str, source_format: str):
        """Create symlinks for images in the target directory."""
        logger.debug(f"Creating symlinks for images in {source_format} format")

        # Get existing splits from PathManager
        existing_splits = self.path_manager.get_existing_splits(dataset_name)
        if not existing_splits:
            self.logger.warning(
                f"No existing splits found for dataset '{dataset_name}'"
            )
            return

        for split in existing_splits:
            split_dir = self.path_manager.get_framework_split_image_dir(
                dataset_name, source_format, split
            )
            split_dir.mkdir(exist_ok=True)
            src_images_dir = self.path_manager.get_dataset_split_images_dir(
                dataset_name, split
            )
            self._symlink_or_copy_dir(src_images_dir, split_dir)

    def _symlink_or_copy_dir(self, src_dir: Path, dst_dir: Path):
        """Symlink all files from src_dir to dst_dir, or copy if symlinks are not supported."""
        src_dir = src_dir.resolve()
        dst_dir = dst_dir.resolve()
        for item in src_dir.iterdir():
            dst_item = dst_dir / item.name
            if dst_item.exists():
                continue
            try:
                relative_path = os.path.relpath(item, dst_dir)
                os.symlink(relative_path, dst_item)
                logger.debug(f"Created relative symlink: {dst_item} -> {relative_path}")
            except Exception:
                shutil.copy2(item, dst_item)

    def _load_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset_info.json for a dataset."""
        info_file = self.path_manager.get_dataset_info_file(dataset_name)
        if info_file.exists():
            with open(info_file, "r") as f:
                return json.load(f)
        return {}

    def _save_yolo_annotations(self, dataset_name: str, yolo_data: Dict[str, Any]):
        """Save YOLO annotations for all splits."""
        labels_dir = self.path_manager.get_framework_annotations_dir(
            dataset_name, "yolo"
        )
        labels_dir.mkdir(parents=True, exist_ok=True)
        for split, split_ann in yolo_data["annotations"].items():
            split_dir = labels_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for image_name, ann_list in split_ann.items():
                label_file = split_dir / f"{Path(image_name).stem}.txt"
                if len(ann_list) == 0:
                    continue
                with open(label_file, "w") as f:
                    for ann in ann_list:
                        f.write(ann + "\n")

    def _save_yolo_data_yaml(self, dataset_name: str, yolo_data: Dict[str, Any]):
        """Save YOLO data.yaml file."""
        yolo_dir = self.path_manager.get_framework_format_dir(dataset_name, "yolo")
        train_dir = self.path_manager.get_framework_split_image_dir(
            dataset_name, "yolo", "train"
        )
        val_dir = self.path_manager.get_framework_split_image_dir(
            dataset_name, "yolo", "val"
        )
        test_dir = self.path_manager.get_framework_split_image_dir(
            dataset_name, "yolo", "test"
        )
        data_yaml = {
            "path": Path(yolo_dir).as_posix(),
            "train": Path(os.path.relpath(train_dir, yolo_dir)).as_posix(),
            "val": Path(os.path.relpath(val_dir, yolo_dir)).as_posix(),
            "test": Path(os.path.relpath(test_dir, yolo_dir)).as_posix(),
            "names": yolo_data["names"],
            "nc": len(yolo_data["names"]),
        }
        with open(
            yolo_dir / self.path_manager.yolo_data_yaml_name, "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(data_yaml, f)

    def export_framework_format(
        self, dataset_name: str, framework: str
    ) -> Dict[str, Any]:
        """
        Export a dataset to a specific framework format.

        Args:
            dataset_name: Name of the dataset
            framework: Framework name ('coco', 'yolo', or 'roi')

        Returns:
            Dictionary with export information
        """
        if framework.lower() == "coco":
            path = self._create_coco_format(dataset_name)
        elif framework.lower() == "yolo":
            path = self._create_yolo_format(dataset_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        return {"framework": framework, "path": path}

    def list_framework_formats(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        List available framework formats for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of framework format information
        """
        formats = []
        for framework in ["coco", "yolo", "roi"]:
            exists = self.path_manager.framework_format_exists(dataset_name, framework)
            formats.append({"framework": framework, "exists": exists})
        return formats
