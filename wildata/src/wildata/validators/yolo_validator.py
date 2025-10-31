import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class YOLOValidator:
    """
    Validator for YOLO format datasets.
    """

    def __init__(
        self, yolo_data_yaml_path: str, filter_invalid_annotations: bool = False
    ):
        """
        Initialize the validator with the path to the YOLO data.yaml file.
        Args:
            yolo_data_yaml_path (str): Path to data.yaml
            filter_invalid_annotations (bool): If True, skip invalid annotations instead of erroring.
        """
        self.yolo_data_yaml_path = yolo_data_yaml_path
        self.yolo_data: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.base_path: Optional[str] = None
        self.filter_invalid_annotations = filter_invalid_annotations
        self.skipped_annotations: List[Dict[str, Any]] = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Perform comprehensive validation of the YOLO dataset.
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Load and validate data.yaml
        if not self._load_data_yaml():
            return False, self.errors, self.warnings

        # Set base path for relative resolution
        path_value = self.yolo_data.get("path")
        if path_value is None:
            self.errors.append("'path' field is required in data.yaml")
            return False, self.errors, self.warnings
        self.base_path = str(Path(path_value).resolve())

        # Validate data.yaml structure
        self._validate_data_yaml_structure()

        # Validate directories and files
        self._validate_directories()

        # Validate label files
        self._validate_label_files()

        return len(self.errors) == 0, self.errors, self.warnings

    def _load_data_yaml(self) -> bool:
        """Load the data.yaml file and validate basic structure."""
        try:
            if not os.path.exists(self.yolo_data_yaml_path):
                self.errors.append(
                    f"data.yaml file does not exist: {self.yolo_data_yaml_path}"
                )
                return False

            with open(self.yolo_data_yaml_path, "r", encoding="utf-8") as f:
                self.yolo_data = yaml.safe_load(f)

            if not isinstance(self.yolo_data, dict):
                self.errors.append("data.yaml root element must be a dictionary")
                return False

            path_value = self.yolo_data.get("path")
            if path_value is None:
                self.errors.append("'path' field is required in data.yaml")
                return False
            if not os.path.exists(path_value):
                self.errors.append(
                    f"'path' field in data.yaml must be an existing directory: {path_value}"
                )
                return False

            names = self.yolo_data.get("names")
            if names is None:
                self.errors.append("'names' field is required in data.yaml")
                return False
            elif len(names) == 0:
                self.errors.append(
                    "'names' field in data.yaml must be a non-empty dictionary"
                )
                return False

            if self.yolo_data.get("train") is None:
                self.errors.append("'train' field is required in data.yaml")
                return False

            return True

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in data.yaml: {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"Error loading data.yaml: {str(e)}")
            return False

    def _resolve_path(self, p: str) -> str:
        if not self.base_path or not p:
            return ""
        return os.path.join(self.base_path, p)

    def _validate_data_yaml_structure(self):
        """Validate the structure of data.yaml file."""
        required_fields = ["train", "names"]
        for field in required_fields:
            if field not in self.yolo_data:
                self.errors.append(f"Missing required field: {field}")

        # Validate class names
        if "names" in self.yolo_data:
            names = self.yolo_data["names"]
            if not names:
                self.errors.append("data.yaml 'names' cannot be empty")
            else:
                # Handle both list and dictionary formats
                if isinstance(names, list):
                    # List format: names are in order, class_id is the index
                    for class_id, name in enumerate(names):
                        if not isinstance(name, str):
                            self.errors.append(
                                f"Class name for ID {class_id} must be a string"
                            )
                        elif not name.strip():
                            self.errors.append(
                                f"Class name for ID {class_id} cannot be empty"
                            )
                elif isinstance(names, dict):
                    # Dictionary format: {class_id: name}
                    for class_id, name in names.items():
                        if not isinstance(name, str):
                            self.errors.append(
                                f"Class name for ID {class_id} must be a string"
                            )
                        elif not name.strip():
                            self.errors.append(
                                f"Class name for ID {class_id} cannot be empty"
                            )
                        if not isinstance(class_id, int):
                            self.errors.append(
                                f"Class ID {class_id} must be an integer"
                            )
                else:
                    self.errors.append("data.yaml 'names' must be a list or dictionary")

        # Validate split directories
        splits = ["train", "val", "test"]
        for split in splits:
            if split in self.yolo_data:
                split_paths = self.yolo_data[split]
                if isinstance(split_paths, list) or isinstance(split_paths, str):
                    pass
                else:
                    self.errors.append(
                        f"data.yaml '{split}' field must be a string or list. Got: {type(split_paths)}"
                    )
                    continue

                # Handle single string path
                if isinstance(split_paths, str):
                    split_paths = [split_paths]
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if split_path and not os.path.exists(resolved):
                        self.warnings.append(f"Split directory does not exist: {resolved}")

    def _validate_directories(self):
        """Validate image and label directories."""
        splits = ["train", "val", "test"]

        for split in splits:
            if split not in self.yolo_data:
                continue

            split_paths = self.yolo_data[split]
            if isinstance(split_paths, list):
                # Handle list of paths
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if not split_path or not os.path.exists(resolved):
                        self.errors.append(
                            f"Split directory does not exist: {resolved}"
                        )
                        continue
                    self._validate_single_split_directory(resolved, split)
            elif isinstance(split_paths, str):
                # Handle single string path
                resolved = self._resolve_path(split_paths)
                if not resolved or not os.path.exists(resolved):
                    self.errors.append(f"Split directory does not exist: {resolved}")
                    continue
                self._validate_single_split_directory(resolved, split)

    def _validate_single_split_directory(self, split_path: str, split_name: str):
        """Validate a single split directory."""
        # The split_path is the images directory specified in data.yaml (e.g., 'train/images')
        images_dir = split_path
        if not os.path.exists(images_dir):
            self.errors.append(
                f"Missing 'images' directory in {split_name} split: {images_dir}"
            )
            return

        # Labels directory should be at the same level as images, replacing 'images' with 'labels'
        labels_dir = images_dir.replace("images", "labels")
        if not os.path.exists(labels_dir):
            self.errors.append(
                f"Missing 'labels' directory in {split_name} split: {labels_dir}"
            )
            return

        # Check for image files
        image_files = self._get_image_files(images_dir)
        if not image_files:
            self.warnings.append(
                f"No image files found in {split_name}/images directory: {images_dir}"
            )
            return

        # Check for corresponding label files
        missing_labels = []
        for img_file in image_files:
            label_file = self._get_label_file_path(img_file, images_dir, labels_dir)
            if not os.path.exists(label_file):
                missing_labels.append(os.path.basename(img_file))

        if missing_labels:
            self.warnings.append(
                f"Missing label files for {len(missing_labels)} images in {split_name} split"
            )

        # Validate directory structure for each file
        self._validate_file_directory_structure(image_files, labels_dir, split_name)

    def _validate_file_directory_structure(
        self, image_files: List[str], labels_dir: str, split_name: str
    ):
        """Validate that files are in the correct directories (images/ and labels/)."""
        for img_file in image_files:
            # Check that image file is in an 'images' directory
            img_path_parts = Path(img_file).parts
            if "images" not in img_path_parts:
                self.errors.append(
                    f"Image file must be in 'images' directory: {img_file}"
                )
            elif img_path_parts.count("images") > 1:
                self.errors.append(
                    f"Image file path contains multiple 'images' directories: {img_file}"
                )

            # Check that corresponding label file is in a 'labels' directory
            label_file = self._get_label_file_path(
                img_file, os.path.dirname(img_file), labels_dir
            )
            label_path_parts = Path(label_file).parts
            if "labels" not in label_path_parts:
                self.errors.append(
                    f"Label file must be in 'labels' directory: {label_file}"
                )
            elif label_path_parts.count("labels") > 1:
                self.errors.append(
                    f"Label file path contains multiple 'labels' directories: {label_file}"
                )

    def _validate_label_files(self):
        """Validate YOLO label files."""
        splits = ["train", "val", "test"]

        for split in splits:
            if split not in self.yolo_data:
                continue

            split_paths = self.yolo_data[split]
            if isinstance(split_paths, list):
                # Handle list of paths
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if split_path and os.path.exists(resolved):
                        self._validate_label_files_in_directory(resolved)
            elif isinstance(split_paths, str):
                # Handle single string path
                resolved = self._resolve_path(split_paths)
                if split_paths and os.path.exists(resolved):
                    self._validate_label_files_in_directory(resolved)

    def _validate_label_files_in_directory(
        self,
        split_path: str,
    ):
        """Validate label files in a specific directory."""
        # Get class names for validation
        class_names = self.yolo_data.get("names", {})
        # Handle both list and dictionary formats for names
        if isinstance(class_names, list):
            num_classes = len(class_names)
        elif isinstance(class_names, dict):
            num_classes = len(class_names)
        else:
            num_classes = 0

        # Check each label file
        labels_dir = str(split_path).replace("images", "labels")
        label_files = self._get_label_files(labels_dir)
        for label_file in label_files:
            self._validate_single_label_file(label_file, num_classes)

    def _validate_single_label_file(self, label_file: str, num_classes: int):
        """Validate a single YOLO label file."""
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    msg = f"{label_file}:{line_num} - Invalid format, need at least 5 values"
                    if self.filter_invalid_annotations:
                        self.skipped_annotations.append(
                            {
                                "file": label_file,
                                "line": line_num,
                                "reason": "too few values",
                                "content": line,
                            }
                        )
                    else:
                        self.errors.append(msg)
                    continue

                # Validate class ID
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        msg = f"{label_file}:{line_num} - Invalid class ID: {class_id}"
                        if self.filter_invalid_annotations:
                            self.skipped_annotations.append(
                                {
                                    "file": label_file,
                                    "line": line_num,
                                    "reason": "invalid class id",
                                    "content": line,
                                }
                            )
                        else:
                            self.errors.append(msg)
                        continue
                except ValueError:
                    msg = f"{label_file}:{line_num} - Invalid class ID: {parts[0]}"
                    if self.filter_invalid_annotations:
                        self.skipped_annotations.append(
                            {
                                "file": label_file,
                                "line": line_num,
                                "reason": "invalid class id",
                                "content": line,
                            }
                        )
                    else:
                        self.errors.append(msg)
                    continue

                # Validate normalized coordinates
                coord_error = False
                for i in range(1, 5):
                    try:
                        coord = float(parts[i])
                        if coord < 0 or coord > 1:
                            msg = f"{label_file}:{line_num} - Coordinate {i} out of range [0,1]: {coord}"
                            if self.filter_invalid_annotations:
                                self.skipped_annotations.append(
                                    {
                                        "file": label_file,
                                        "line": line_num,
                                        "reason": f"coordinate {i} out of range",
                                        "content": line,
                                    }
                                )
                                coord_error = True
                                break
                            else:
                                self.errors.append(msg)
                                coord_error = True
                                break
                    except ValueError:
                        msg = f"{label_file}:{line_num} - Invalid coordinate {i}: {parts[i]}"
                        if self.filter_invalid_annotations:
                            self.skipped_annotations.append(
                                {
                                    "file": label_file,
                                    "line": line_num,
                                    "reason": f"invalid coordinate {i}",
                                    "content": line,
                                }
                            )
                            coord_error = True
                            break
                        else:
                            self.errors.append(msg)
                            coord_error = True
                            break
                if coord_error:
                    continue

                # Validate segmentation points if present
                if len(parts) > 5:
                    if (
                        len(parts) % 2 != 1
                    ):  # Must be odd (class + pairs of coordinates)
                        msg = f"{label_file}:{line_num} - Invalid number of segmentation points"
                        if self.filter_invalid_annotations:
                            self.skipped_annotations.append(
                                {
                                    "file": label_file,
                                    "line": line_num,
                                    "reason": "invalid segmentation count",
                                    "content": line,
                                }
                            )
                        else:
                            self.errors.append(msg)
                        continue
                    else:
                        for i in range(5, len(parts), 2):
                            try:
                                x = float(parts[i])
                                y = float(parts[i + 1])
                                if x < 0 or x > 1 or y < 0 or y > 1:
                                    msg = f"{label_file}:{line_num} - Segmentation point out of range: ({x}, {y})"
                                    if self.filter_invalid_annotations:
                                        self.skipped_annotations.append(
                                            {
                                                "file": label_file,
                                                "line": line_num,
                                                "reason": "segmentation point out of range",
                                                "content": line,
                                            }
                                        )
                                    else:
                                        self.errors.append(msg)
                                    break
                            except (ValueError, IndexError):
                                msg = f"{label_file}:{line_num} - Invalid segmentation point"
                                if self.filter_invalid_annotations:
                                    self.skipped_annotations.append(
                                        {
                                            "file": label_file,
                                            "line": line_num,
                                            "reason": "invalid segmentation point",
                                            "content": line,
                                        }
                                    )
                                else:
                                    self.errors.append(msg)
                                break

        except FileNotFoundError:
            self.errors.append(f"Label file not found: {label_file}")
        except Exception as e:
            self.errors.append(f"Error reading label file {label_file}: {str(e)}")

    def _get_image_files(self, directory: str) -> List[str]:
        """Get list of image files in the directory."""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        try:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(directory, file))
        except OSError:
            self.errors.append(f"Cannot read directory: {directory}")

        return sorted(image_files)

    def _get_label_files(self, directory: str) -> List[str]:
        """Get list of label files in the directory."""
        label_files = []

        try:
            for file in os.listdir(directory):
                if file.endswith(".txt"):
                    label_files.append(os.path.join(directory, file))
        except OSError:
            self.errors.append(f"Cannot read directory: {directory}")

        return sorted(label_files)

    def _get_label_file_path(
        self, image_file: str, images_dir: str, labels_dir: str
    ) -> str:
        """Get the corresponding label file path for an image, given images_dir and labels_dir."""
        relative_image_path = os.path.relpath(image_file, images_dir)
        label_file = str(Path(relative_image_path).with_suffix(".txt")).replace(
            "images", "labels"
        )
        return os.path.join(labels_dir, label_file)

    def get_skipped_annotations(self) -> List[Dict[str, Any]]:
        """Get information about skipped annotations when filter_invalid_annotations=True."""
        return self.skipped_annotations

    def get_skipped_count(self) -> int:
        """Get the number of skipped annotations."""
        return len(self.skipped_annotations)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        # Count files
        total_images = 0
        total_labels = 0
        for split in ["train", "val", "test"]:
            if split in self.yolo_data:
                split_paths = self.yolo_data[split]
                if isinstance(split_paths, list):
                    for split_path in split_paths:
                        resolved = self._resolve_path(split_path)
                        if split_path and os.path.exists(resolved):
                            images_dir = resolved
                            labels_dir = images_dir.replace("images", "labels")
                            if os.path.exists(images_dir):
                                total_images += len(self._get_image_files(images_dir))
                            if os.path.exists(labels_dir):
                                total_labels += len(self._get_label_files(labels_dir))
                elif isinstance(split_paths, str):
                    resolved = self._resolve_path(split_paths)
                    if split_paths and os.path.exists(resolved):
                        images_dir = resolved
                        labels_dir = images_dir.replace("images", "labels")
                        if os.path.exists(images_dir):
                            total_images += len(self._get_image_files(images_dir))
                        if os.path.exists(labels_dir):
                            total_labels += len(self._get_label_files(labels_dir))
        return {
            "data_yaml_path": self.yolo_data_yaml_path,
            "is_valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "class_count": len(self.yolo_data.get("names", {})),
            "total_images": total_images,
            "total_labels": total_labels,
            "skipped_annotation_count": len(self.skipped_annotations),
            "errors": self.errors,
            "warnings": self.warnings,
        }
