import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

from ..adapters.utils import read_image
from ..validators.yolo_validator import YOLOValidator
from .base_converter import BaseConverter


class YOLOToMasterConverter(BaseConverter):
    """
    Converter from YOLO format to master annotation format.
    """

    def __init__(self, yolo_data_yaml_path: str):
        """
        Initialize the converter with the path to the YOLO data.yaml file.
        """
        super().__init__()
        self.yolo_data_yaml_path = yolo_data_yaml_path
        self.yolo_data: Dict[str, Any] = {}
        self.base_path = None

    def _load_yolo_data(self, filter_invalid_annotations: bool = False) -> None:
        """
        Load the YOLO data.yaml file and parse the dataset structure.
        Args:
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        """
        # Validate before loading
        validator = YOLOValidator(
            self.yolo_data_yaml_path,
            filter_invalid_annotations=filter_invalid_annotations,
        )
        is_valid, errors, warnings = validator.validate()

        if not is_valid:
            error_msg = f"YOLO validation failed for {self.yolo_data_yaml_path}:\n"
            error_msg += "\n".join(errors)
            if warnings:
                error_msg += f"\nWarnings:\n" + "\n".join(warnings)
            raise ValueError(error_msg)

        # Load the data
        with open(self.yolo_data_yaml_path, "r", encoding="utf-8") as f:
            self.yolo_data = yaml.safe_load(f)
        self.base_path = self.yolo_data.get("path")
        if self.base_path and not os.path.isabs(self.base_path):
            self.base_path = os.path.abspath(os.path.dirname(self.yolo_data_yaml_path))
        if filter_invalid_annotations:
            skipped_count = validator.get_skipped_count()
            if skipped_count > 0:
                print(
                    f"Warning: Skipped {skipped_count} invalid YOLO annotation lines during validation"
                )

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        if self.base_path:
            return os.path.abspath(os.path.join(self.base_path, p))
        return os.path.abspath(
            os.path.join(os.path.dirname(self.yolo_data_yaml_path), p)
        )

    def convert(
        self,
        dataset_name: str,
        task_type: str = "detection",
        filter_invalid_annotations: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Convert the loaded YOLO data to COCO format with split-based organization.
        Args:
            dataset_name (str): Name of the dataset.
            version (str): Version of the dataset.
            task_type (str): Type of task (detection, segmentation).
            validate_output (bool): Whether to validate the output.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        Returns:
            Tuple of (dataset_info, split_data) where split_data maps split names to COCO format data.
        """

        self._load_yolo_data(filter_invalid_annotations=filter_invalid_annotations)

        # Extract class names
        class_names_dict = self.yolo_data.get("names", {})
        classes = [
            {
                "id": int(class_id),
                "name": name,
                "supercategory": "",  # YOLO doesn't have supercategories
            }
            for class_id, name in class_names_dict.items()
        ]

        # Create dataset info
        dataset_info = {
            "name": dataset_name,
            "task_type": task_type,
            "classes": classes,
        }

        # Process each split
        split_data = {}

        for split in ["train", "val", "test"]:
            split_paths = self.yolo_data.get(split, [])
            if not split_paths:
                continue

            split_images = []
            split_annotations = []
            annotation_id = 1

            if isinstance(split_paths, list):
                # Handle list of paths
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if not split_path or not os.path.exists(resolved):
                        continue
                    self._process_split_directory_for_coco(
                        resolved, split_images, split_annotations, annotation_id
                    )
                    annotation_id = len(split_annotations) + 1
            elif isinstance(split_paths, str):
                resolved = self._resolve_path(split_paths)
                if resolved and os.path.exists(resolved):
                    self._process_split_directory_for_coco(
                        resolved, split_images, split_annotations, annotation_id
                    )
                    annotation_id = len(split_annotations) + 1

            # Only add split if it has data
            if split_images:
                split_data[split] = {
                    "images": split_images,
                    "annotations": split_annotations,
                    "categories": classes,
                }

            try:
                self._validate_coco_annotation(
                    split_data[split], filter_invalid_annotations
                )
            except Exception as e:
                print(
                    f"Error validating COCO annotation for split {split}: {traceback.format_exc()}"
                )

        return dataset_info, split_data

    def _process_split_directory_for_coco(
        self,
        split_path: str,
        split_images: List,
        split_annotations: List,
        annotation_id: int,
    ):
        """Process a single split directory for COCO format."""
        # Get image files
        image_files = self._get_image_files(split_path)

        # Process each image
        for img_idx, img_file in enumerate(
            tqdm(image_files, desc=f"Converting YOLO {split_path} to COCO")
        ):
            img_id = len(split_images) + 1

            # Get image dimensions
            width, height = self._get_image_dimensions(img_file)

            # Create COCO image entry
            coco_image = {
                "id": img_id,
                "file_name": img_file,
                "width": width,
                "height": height,
            }
            split_images.append(coco_image)

            # Process corresponding label file
            label_file = self._get_label_file_path(img_file)
            if os.path.exists(label_file):
                annotations = self._parse_yolo_label_file(
                    label_file, img_id, width, height
                )
                for ann in annotations:
                    ann["id"] = annotation_id
                    annotation_id += 1
                    split_annotations.append(ann)

    def _get_image_files(self, images_dir: str) -> List[str]:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                p = (Path(images_dir) / file).resolve().as_posix()
                image_files.append(p)
        return sorted(image_files)

    def _get_label_file_path(self, image_file: str) -> str:
        path = Path(image_file).with_suffix(".txt")
        return str(path).replace("images", "labels")

    def _get_image_dimensions(self, image_file: str) -> Tuple[int, int]:
        assert os.path.exists(image_file), f"Image file does not exist: {image_file}"
        image = read_image(image_file)
        width, height = image.size
        return width, height

    def _parse_yolo_label_file(
        self, label_file: str, image_id: int, width: int, height: int
    ) -> List[Dict[str, Any]]:
        annotations = []
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    # Try to parse class_id, skip if invalid
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue  # Skip invalid class_id

                    # Try to parse coordinates, skip if invalid
                    try:
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])

                        # Validate coordinate ranges
                        if not (
                            0 <= x_center_norm <= 1
                            and 0 <= y_center_norm <= 1
                            and 0 <= w_norm <= 1
                            and 0 <= h_norm <= 1
                        ):
                            continue  # Skip invalid coordinates
                    except ValueError:
                        continue  # Skip invalid coordinates

                    x_center = x_center_norm * width
                    y_center = y_center_norm * height
                    w = w_norm * width
                    h = h_norm * height
                    x = x_center - w / 2
                    y = y_center - h / 2
                    area = w * h
                    segmentation = []

                    # Parse segmentation points if present
                    if len(parts) > 5:
                        try:
                            seg_points = []
                            for i in range(5, len(parts), 2):
                                if i + 1 < len(parts):
                                    x_norm = float(parts[i])
                                    y_norm = float(parts[i + 1])
                                    # Validate segmentation coordinate ranges
                                    if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
                                        continue  # Skip invalid segmentation points
                                    x_abs = x_norm * width
                                    y_abs = y_norm * height
                                    seg_points.extend([x_abs, y_abs])
                            if seg_points:
                                segmentation = [seg_points]
                        except (ValueError, IndexError):
                            # Skip invalid segmentation points
                            pass

                    annotation = {
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                        "keypoints": [],
                        "attributes": {},
                    }
                    annotations.append(annotation)
        except FileNotFoundError:
            pass  # Label file doesn't exist
        return annotations
