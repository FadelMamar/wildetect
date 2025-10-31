import json
import os

import pytest
from wildata.converters.yolo_to_master import YOLOToMasterConverter

COCO_DATA_DIR = os.getenv(
    "COCO_DATA_DIR", r"D:\workspace\repos\wildtrain\data\savmap\coco"
)
YOLO_DATA_DIR = os.getenv(
    "YOLO_DATA_DIR", r"D:\workspace\repos\wildtrain\data\savmap\yolo"
)


def test_yolo_to_coco_conversion(tmp_path):
    if not os.path.exists(YOLO_DATA_DIR):
        pytest.skip(f"YOLO data directory not found: {YOLO_DATA_DIR}")
    data_yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    if not os.path.exists(data_yaml_path):
        pytest.skip("data.yaml not found in YOLO directory")
    converter = YOLOToMasterConverter(data_yaml_path)
    converter.load_yolo_data()
    dataset_info, split_data = converter.convert_to_coco_format("test_dataset")

    # Verify structure
    assert "name" in dataset_info
    assert "version" in dataset_info
    assert len(split_data) > 0

    # Check that we have COCO format data for each split
    for split_name, coco_data in split_data.items():
        assert "images" in coco_data
        assert "annotations" in coco_data
        assert "categories" in coco_data
        assert len(coco_data["images"]) > 0


def test_converter_validation_disabled():
    """Test that converters work when validation is disabled."""
    if not os.path.exists(YOLO_DATA_DIR):
        pytest.skip(f"YOLO data directory not found: {YOLO_DATA_DIR}")
    data_yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    if not os.path.exists(data_yaml_path):
        pytest.skip("data.yaml not found in YOLO directory")
    converter = YOLOToMasterConverter(data_yaml_path)
    converter.load_yolo_data()
    # This should work without validation
    dataset_info, split_data = converter.convert_to_coco_format("test_dataset")
    assert "name" in dataset_info
    assert len(split_data) > 0
