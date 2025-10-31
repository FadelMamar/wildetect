import os

import pytest
from wildata.validators.coco_validator import COCOValidator
from wildata.validators.yolo_validator import YOLOValidator

COCO_DATA_DIR = os.getenv("COCO_DATA_DIR", "data/savmap/coco")
YOLO_DATA_DIR = os.getenv("YOLO_DATA_DIR", "data/savmap/yolo")


def test_coco_validator_with_real_data():
    """Test COCO validator with real data."""
    if not os.path.exists(COCO_DATA_DIR):
        pytest.skip(f"COCO data directory not found: {COCO_DATA_DIR}")

    coco_files = [f for f in os.listdir(COCO_DATA_DIR) if f.endswith(".json")]
    if not coco_files:
        pytest.skip("No COCO annotation files found")

    coco_file = os.path.join(COCO_DATA_DIR, coco_files[0])
    validator = COCOValidator(coco_file)
    is_valid, errors, warnings = validator.validate()

    # Should be valid if it's a real dataset
    assert is_valid, f"COCO validation failed: {errors}"

    # Get summary
    summary = validator.get_summary()
    assert summary["is_valid"] == is_valid
    assert summary["image_count"] > 0
    assert summary["annotation_count"] > 0
    assert summary["category_count"] > 0


def test_yolo_validator_with_real_data():
    """Test YOLO validator with real data."""
    if not os.path.exists(YOLO_DATA_DIR):
        pytest.skip(f"YOLO data directory not found: {YOLO_DATA_DIR}")

    data_yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    if not os.path.exists(data_yaml_path):
        pytest.skip("data.yaml not found in YOLO directory")

    validator = YOLOValidator(data_yaml_path)
    is_valid, errors, warnings = validator.validate()

    # Should be valid if it's a real dataset
    assert is_valid, f"YOLO validation failed: {errors}"

    # Get summary
    summary = validator.get_summary()
    assert summary["is_valid"] == is_valid
    assert summary["class_count"] > 0
    assert summary["total_images"] > 0


def test_coco_validator_invalid_file():
    """Test COCO validator with invalid file."""
    # Test with non-existent file
    validator = COCOValidator("non_existent_file.json")
    is_valid, errors, warnings = validator.validate()

    assert not is_valid
    assert len(errors) > 0
    assert "does not exist" in errors[0]


def test_yolo_validator_invalid_file():
    """Test YOLO validator with invalid file."""
    # Test with non-existent file
    validator = YOLOValidator("non_existent_data.yaml")
    is_valid, errors, warnings = validator.validate()

    assert not is_valid
    assert len(errors) > 0
    assert "data.yaml file does not exist" in errors[0]


def test_yolo_validator_directory_structure():
    """Test YOLO validator with proper images/ and labels/ directory structure."""
    import tempfile
    from pathlib import Path

    import yaml

    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create the expected directory structure
        train_dir = temp_path / "train"
        train_images_dir = train_dir / "images"
        train_labels_dir = train_dir / "labels"
        val_dir = temp_path / "val"
        val_images_dir = val_dir / "images"
        val_labels_dir = val_dir / "labels"
        test_dir = temp_path / "test"
        test_images_dir = test_dir / "images"
        test_labels_dir = test_dir / "labels"

        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)
        test_images_dir.mkdir(parents=True, exist_ok=True)
        test_labels_dir.mkdir(parents=True, exist_ok=True)

        # Create some test files
        (train_images_dir / "image1.jpg").touch()
        (train_images_dir / "image2.png").touch()
        (train_labels_dir / "image1.txt").touch()
        (train_labels_dir / "image2.txt").touch()
        (val_images_dir / "val_image1.jpg").touch()
        (val_labels_dir / "val_image1.txt").touch()
        (test_images_dir / "test_image1.jpg").touch()
        (test_labels_dir / "test_image1.txt").touch()

        # Create data.yaml
        data_yaml = {
            "path": str(temp_path),
            "train": r"train/images",
            "val": r"val/images",
            "test": r"test/images",
            "names": {0: "class1", 1: "class2"},
        }

        data_yaml_path = temp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        # Test validator
        validator = YOLOValidator(str(data_yaml_path))
        is_valid, errors, warnings = validator.validate()

        # Should be valid with proper structure
        assert is_valid, f"Validation failed: {errors}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Test summary
        summary = validator.get_summary()
        assert summary["is_valid"] == True
        assert summary["total_images"] == 4  # 2 train + 1 val + 1 test
        assert summary["total_labels"] == 4  # 2 train + 1 val + 1 test


def test_yolo_validator_missing_directories():
    """Test YOLO validator with missing images/ and labels/ directories."""
    import tempfile
    from pathlib import Path

    import yaml

    # Create a temporary directory structure without proper subdirectories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create train directory but no images/ or labels/ subdirectories
        train_dir = temp_path / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        # Create data.yaml
        data_yaml = {
            "path": str(temp_path),
            "train": "train/images",
            "names": {0: "class1", 1: "class2"},
        }

        data_yaml_path = temp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        # Test validator
        validator = YOLOValidator(str(data_yaml_path))
        is_valid, errors, warnings = validator.validate()

        # Should be invalid due to missing directories
        print(f"is_valid: {is_valid}")
        print(f"errors: {errors}")
        print(f"warnings: {warnings}")
        assert not is_valid
        assert any("Split directory does not exist" in error for error in errors)
