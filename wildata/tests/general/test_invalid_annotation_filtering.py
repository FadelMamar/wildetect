import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from wildata.validators.coco_validator import COCOValidator
from wildata.validators.yolo_validator import YOLOValidator


def create_invalid_coco_data():
    """Create COCO data with invalid annotations."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            # Valid annotation
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
            },
            # Invalid: missing required field
            {
                "id": 2,
                "image_id": 1,
                "bbox": [100, 100, 200, 150],
            },  # missing category_id
            # Invalid: references non-existent image
            {
                "id": 3,
                "image_id": 999,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
            },
            # Invalid: negative area
            {
                "id": 4,
                "image_id": 2,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": -1000,
            },
            # Invalid: invalid bbox dimensions
            {
                "id": 5,
                "image_id": 2,
                "category_id": 1,
                "bbox": [100, 100, 0, 150],
                "area": 0,
            },  # width = 0
        ],
        "categories": [{"id": 1, "name": "test_category", "supercategory": "test"}],
    }


class TestInvalidAnnotationFiltering:
    """Test the invalid annotation filtering functionality."""

    def test_coco_validator_with_invalid_data(self):
        """Test that COCOValidator can handle invalid annotations."""
        coco_data = create_invalid_coco_data()

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        try:
            # Create images directory
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Create dummy image files
            for img in coco_data["images"]:
                img_path = os.path.join(images_dir, img["file_name"])
                # Create a dummy image file
                with open(img_path, "w") as f:
                    f.write("dummy image data")

            # Save annotation file
            annotation_file = os.path.join(temp_dir, "annotations.json")
            with open(annotation_file, "w") as f:
                json.dump(coco_data, f)

            # Test validation
            validator = COCOValidator(annotation_file)
            is_valid, errors, warnings = validator.validate()

            # Should detect invalid annotations
            assert not is_valid
            assert len(errors) > 0

            # Check that specific errors are detected
            error_messages = " ".join(errors).lower()
            assert "missing" in error_messages or "invalid" in error_messages

        finally:
            shutil.rmtree(temp_dir)

    def test_yolo_validator_with_invalid_data(self):
        """Test that YOLOValidator can handle invalid annotations."""
        # Create temporary YOLO structure with invalid data
        temp_dir = tempfile.mkdtemp()
        try:
            # Create data.yaml with invalid class mapping
            data_yaml = {
                "path": temp_dir,
                "train": "train/images",
                "val": "val/images",
                "nc": 1,
                "names": ["test_class"],
            }

            yaml_path = os.path.join(temp_dir, "data.yaml")
            with open(yaml_path, "w") as f:
                import yaml

                yaml.dump(data_yaml, f)

            # Create invalid label file
            labels_dir = os.path.join(temp_dir, "train", "labels")
            os.makedirs(labels_dir, exist_ok=True)

            # Create label file with invalid format
            label_file = os.path.join(labels_dir, "image1.txt")
            with open(label_file, "w") as f:
                f.write("1 0.5 0.5 0.1 0.1\n")  # Valid
                f.write("999 0.5 0.5 0.1 0.1\n")  # Invalid class ID
                f.write("1 1.5 0.5 0.1 0.1\n")  # Invalid coordinates

            # Test validation
            validator = YOLOValidator(yaml_path)
            is_valid, errors, warnings = validator.validate()

            # Should detect invalid annotations
            assert not is_valid
            assert len(errors) > 0

        finally:
            shutil.rmtree(temp_dir)

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        # Test empty COCO data
        empty_coco = {"images": [], "annotations": [], "categories": []}

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        try:
            # Create images directory
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Save annotation file
            annotation_file = os.path.join(temp_dir, "annotations.json")
            with open(annotation_file, "w") as f:
                json.dump(empty_coco, f)

            validator = COCOValidator(annotation_file)
            is_valid, errors, warnings = validator.validate()

            # Empty dataset should be valid but may or may not have warnings
            assert is_valid
            # Note: Empty datasets might not generate warnings in all cases

        finally:
            shutil.rmtree(temp_dir)
