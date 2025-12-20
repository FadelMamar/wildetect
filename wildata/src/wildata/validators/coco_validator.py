import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class COCOValidator:
    """
    Validator for COCO format annotation files.
    """

    def __init__(
        self,
        coco_file_path: str,
        filter_invalid_annotations: bool = False,
        coco_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the validator with the path to the COCO annotation file.
        Args:
            coco_file_path (str): Path to the COCO annotation file.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        """
        self.coco_file_path = coco_file_path
        self.filter_invalid_annotations = filter_invalid_annotations
        self.coco_data: Dict[str, Any] = coco_data or {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.skipped_annotations = []

    def validate(self, bbox_tolerance: int = 5) -> Tuple[bool, List[str], List[str]]:
        """
        Perform comprehensive validation of the COCO file.
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        self.bbox_tolerance = bbox_tolerance

        # Load and validate file
        if not self._load_file():
            return False, self.errors, self.warnings

        # Validate structure
        self._validate_structure()

        # Validate data integrity
        self._validate_data_integrity()

        # Validate relationships
        self._validate_relationships()

        # Validate annotations
        self._validate_annotations()

        return len(self.errors) == 0, self.errors, self.warnings

    def _load_file(self) -> bool:
        """Load the COCO JSON file and validate basic structure."""
        if self.coco_data:
            return True

        try:
            assert str(self.coco_file_path).endswith(
                ".json"
            ), "Provide the path to the annotations"
            if not os.path.exists(self.coco_file_path):
                self.errors.append(f"File does not exist: {self.coco_file_path}")
                return False

            with open(self.coco_file_path, "r", encoding="utf-8") as f:
                self.coco_data = json.load(f)

            if not isinstance(self.coco_data, dict):
                self.errors.append("Root element must be a dictionary")
                return False

            return True

        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"Error loading file: {str(e)}")
            return False

    def _validate_structure(self):
        """Validate the basic structure of the COCO file."""
        required_sections = ["images", "annotations", "categories"]

        for section in required_sections:
            if section not in self.coco_data:
                self.errors.append(f"Missing required section: {section}")
            elif not isinstance(self.coco_data[section], list):
                self.errors.append(f"Section '{section}' must be a list")

    def _validate_data_integrity(self):
        """Validate data integrity within each section."""
        # Validate images
        if "images" in self.coco_data:
            self._validate_images()

        # Validate categories
        if "categories" in self.coco_data:
            self._validate_categories()

        # Validate annotations
        if "annotations" in self.coco_data:
            self._validate_annotation_structure()

    def _validate_images(self):
        """Validate image entries."""
        required_image_fields = ["id", "file_name", "width", "height"]
        image_ids = set()

        for i, img in enumerate(self.coco_data["images"]):
            if not isinstance(img, dict):
                self.errors.append(f"Image {i} must be a dictionary")
                continue

            # Check required fields
            for field in required_image_fields:
                if field not in img:
                    self.errors.append(f"Image {i} missing required field: {field}")
                elif field in ["width", "height"] and not isinstance(img[field], int):
                    self.errors.append(f"Image {i} field '{field}' must be an integer")

            # Check for duplicate IDs
            if "id" in img:
                if img["id"] in image_ids:
                    self.errors.append(f"Duplicate image ID: {img['id']}")
                else:
                    image_ids.add(img["id"])

            # Validate dimensions
            if "width" in img and "height" in img:
                if img["width"] <= 0 or img["height"] <= 0:
                    self.errors.append(
                        f"Image {i} has invalid dimensions: {img['width']}x{img['height']}"
                    )

    def _validate_categories(self):
        """Validate category entries."""
        required_category_fields = ["id", "name"]
        category_ids = set()

        for i, cat in enumerate(self.coco_data["categories"]):
            if not isinstance(cat, dict):
                self.errors.append(f"Category {i} must be a dictionary")
                continue

            # Check required fields
            for field in required_category_fields:
                if field not in cat:
                    self.errors.append(f"Category {i} missing required field: {field}")

            # Check for duplicate IDs
            if "id" in cat:
                if cat["id"] in category_ids:
                    self.errors.append(f"Duplicate category ID: {cat['id']}")
                else:
                    category_ids.add(cat["id"])

    def _validate_bbox(
        self, bbox: List[float], image_width: int, image_height: int, tolerance: int = 5
    ) -> bool:
        """Validate bbox."""
        x, y, w, h = bbox
        is_valid = (
            x >= -tolerance
            and y >= -tolerance
            and x + w <= image_width + tolerance
            and y + h <= image_height + tolerance
        )
        return is_valid

    def _validate_annotation_structure(self):
        """Validate annotation structure."""
        required_annotation_fields = ["id", "image_id", "category_id", "bbox"]
        valid_annotations = []
        skipped_count = 0

        for i, ann in enumerate(self.coco_data["annotations"]):
            is_valid = True
            skip_reason = None

            if not isinstance(ann, dict):
                if self.filter_invalid_annotations:
                    skip_reason = "not a dictionary"
                    is_valid = False
                else:
                    self.errors.append(f"Annotation {i} must be a dictionary")
                    continue

            # Check required fields
            for field in required_annotation_fields:
                if field not in ann:
                    if self.filter_invalid_annotations:
                        skip_reason = f"missing required field: {field}"
                        is_valid = False
                        break
                    else:
                        self.errors.append(
                            f"Annotation {i} missing required field: {field}"
                        )
                        continue

            # Validate bbox format
            if "bbox" in ann:
                bbox = ann["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    if self.filter_invalid_annotations:
                        skip_reason = "bbox must be a list of 4 numbers"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} bbox must be a list of 4 numbers"
                        )
                        continue
                elif not all(isinstance(x, (int, float)) for x in bbox):
                    if self.filter_invalid_annotations:
                        skip_reason = "bbox must contain only numbers"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} bbox must contain only numbers"
                        )
                        continue

            # Add to valid annotations or skip
            if is_valid:
                valid_annotations.append(ann)
            else:
                skipped_count += 1
                self.skipped_annotations.append(
                    {"index": i, "annotation": ann, "reason": skip_reason}
                )

        # Update the COCO data with filtered annotations
        if self.filter_invalid_annotations and skipped_count > 0:
            self.coco_data["annotations"] = valid_annotations
            self.warnings.append(
                f"Skipped {skipped_count} invalid annotations during structure validation"
            )

    def _validate_relationships(self):
        """Validate relationships between images, annotations, and categories."""
        if (
            "images" not in self.coco_data
            or "annotations" not in self.coco_data
            or "categories" not in self.coco_data
        ):
            return

        image_ids = {img["id"] for img in self.coco_data["images"]}
        category_ids = {cat["id"] for cat in self.coco_data["categories"]}
        valid_annotations = []
        skipped_count = 0

        # Check annotation references
        for i, ann in enumerate(self.coco_data["annotations"]):
            is_valid = True
            skip_reason = None

            if "image_id" in ann and ann["image_id"] not in image_ids:
                if self.filter_invalid_annotations:
                    skip_reason = f"references non-existent image_id: {ann['image_id']}"
                    is_valid = False
                else:
                    self.errors.append(
                        f"Annotation {i} references non-existent image_id: {ann['image_id']}"
                    )
                    continue

            if "category_id" in ann and ann["category_id"] not in category_ids:
                if self.filter_invalid_annotations:
                    skip_reason = (
                        f"references non-existent category_id: {ann['category_id']}"
                    )
                    is_valid = False
                else:
                    self.errors.append(
                        f"Annotation {i} references non-existent category_id: {ann['category_id']}"
                    )
                    continue

            # Add to valid annotations or skip
            if is_valid:
                valid_annotations.append(ann)
            else:
                skipped_count += 1
                self.skipped_annotations.append(
                    {"index": i, "annotation": ann, "reason": skip_reason}
                )

        # Update the COCO data with filtered annotations
        if self.filter_invalid_annotations and skipped_count > 0:
            self.coco_data["annotations"] = valid_annotations
            self.warnings.append(
                f"Skipped {skipped_count} invalid annotations during relationship validation"
            )

    def _validate_annotations(self):
        """Validate annotation-specific content."""
        valid_annotations = []
        skipped_count = 0

        for i, ann in enumerate(self.coco_data["annotations"]):
            is_valid = True
            skip_reason = None

            # Validate bbox values
            if (
                "bbox" in ann
                and isinstance(ann["bbox"], list)
                and len(ann["bbox"]) == 4
            ):
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    if self.filter_invalid_annotations:
                        skip_reason = f"invalid bbox dimensions: {w}x{h}"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} has invalid bbox dimensions: {w}x{h}"
                        )
                        continue

            # Validate area
            if "area" in ann:
                if not isinstance(ann["area"], (int, float)) or ann["area"] <= 0:
                    if self.filter_invalid_annotations:
                        skip_reason = f"invalid area: {ann['area']}"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} has invalid area: {ann['area']}"
                        )
                        continue

            # Validate iscrowd
            if "iscrowd" in ann:
                if ann["iscrowd"] not in [0, 1]:
                    if self.filter_invalid_annotations:
                        skip_reason = f"invalid iscrowd value: {ann['iscrowd']}"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} has invalid iscrowd value: {ann['iscrowd']}"
                        )
                        continue

            # Validate segmentation
            if "segmentation" in ann:
                seg = ann["segmentation"]
                if not isinstance(seg, list):
                    if self.filter_invalid_annotations:
                        skip_reason = "segmentation must be a list"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} segmentation must be a list"
                        )
                        continue
                elif seg and not isinstance(seg[0], list):
                    if self.filter_invalid_annotations:
                        skip_reason = "segmentation must be a list of polygons"
                        is_valid = False
                    else:
                        self.errors.append(
                            f"Annotation {i} segmentation must be a list of polygons"
                        )
                        continue

            # Add to valid annotations or skip
            if is_valid:
                valid_annotations.append(ann)
            else:
                skipped_count += 1
                self.skipped_annotations.append(
                    {"index": i, "annotation": ann, "reason": skip_reason}
                )

        # Update the COCO data with filtered annotations
        if self.filter_invalid_annotations and skipped_count > 0:
            self.coco_data["annotations"] = valid_annotations
            self.warnings.append(
                f"Skipped {skipped_count} invalid annotations during content validation"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "file_path": self.coco_file_path,
            "is_valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "image_count": len(self.coco_data.get("images", [])),
            "annotation_count": len(self.coco_data.get("annotations", [])),
            "category_count": len(self.coco_data.get("categories", [])),
            "skipped_annotation_count": len(self.skipped_annotations),
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def get_skipped_annotations(self) -> List[Dict[str, Any]]:
        """
        Get information about skipped annotations when filter_invalid_annotations=True.
        Returns:
            List[Dict[str, Any]]: List of skipped annotation information.
        """
        return self.skipped_annotations

    def get_skipped_count(self) -> int:
        """
        Get the number of skipped annotations.
        Returns:
            int: Number of skipped annotations.
        """
        return len(self.skipped_annotations)
