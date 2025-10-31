import json
import os
import tempfile

import pytest
from wildata.adapters.yolo_adapter import YOLOAdapter

# Real data paths (can be overridden with environment variables)
COCO_DATA_DIR = os.getenv("COCO_DATA_DIR", "D:/workspace/savmap/coco")
YOLO_DATA_DIR = os.getenv("YOLO_DATA_DIR", "D:/workspace/savmap/yolo")


def test_yolo_adapter_with_coco_data(tmp_path):
    """Test YOLO adapter using COCO data."""
    # Create sample COCO data
    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": "test_image.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "test_category",
                "supercategory": "test",
            }
        ],
    }

    # Test YOLO adapter with COCO data
    adapter = YOLOAdapter(coco_data=coco_data)
    adapter.load_coco_annotation()

    # Convert to YOLO format
    yolo_data = adapter.convert("train")

    # Debug output
    print(f"YOLO data keys: {list(yolo_data.keys())}")
    print(f"Number of images with labels: {len(yolo_data)}")
    if yolo_data:
        first_key = list(yolo_data.keys())[0]
        print(f"First image: {first_key}")
        print(f"Labels for first image: {yolo_data[first_key]}")

    # Verify YOLO structure
    assert len(yolo_data) > 0
    assert "test_image.jpg" in yolo_data
    assert len(yolo_data["test_image.jpg"]) > 0

    # Save YOLO output
    adapter.save(yolo_data)
