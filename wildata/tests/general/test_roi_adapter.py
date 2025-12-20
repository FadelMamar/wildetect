import json
import os
import tempfile
from pathlib import Path

import pytest
from wildata.adapters.roi_adapter import ROIAdapter


def test_roi_adapter_basic():
    """Test basic ROI adapter functionality."""
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

    # Create ROI adapter
    adapter = ROIAdapter(coco_data=coco_data)

    # Test conversion
    roi_data = adapter.convert("train")

    # Verify structure
    assert "roi_images" in roi_data
    assert "roi_labels" in roi_data
    assert "class_mapping" in roi_data
    assert "statistics" in roi_data

    # Verify we have one ROI from the annotation
    assert len(roi_data["roi_images"]) == 1
    assert len(roi_data["roi_labels"]) == 1

    # Verify ROI image info
    roi_image = roi_data["roi_images"][0]
    assert "roi_id" in roi_image
    assert "roi_filename" in roi_image
    assert "original_image_path" in roi_image
    assert "bbox" in roi_image

    # Verify ROI label info
    roi_label = roi_data["roi_labels"][0]
    assert "roi_id" in roi_label
    assert "class_id" in roi_label
    assert "class_name" in roi_label
    assert roi_label["class_name"] == "test_category"


def test_roi_adapter_with_callback():
    """Test ROI adapter with custom callback."""

    def mock_roi_callback(image_path, image_data):
        """Mock callback that returns predefined ROIs."""
        return [
            {"bbox": [50, 50, 100, 100], "class": "callback_object"},
            {"bbox": [200, 200, 150, 150], "class": "callback_object"},
        ]

    # Create sample COCO data with no annotations
    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": "test_image.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [],  # No annotations
        "categories": [
            {
                "id": 1,
                "name": "test_category",
                "supercategory": "test",
            }
        ],
    }

    # Create ROI adapter with callback
    adapter = ROIAdapter(
        coco_data=coco_data, roi_callback=mock_roi_callback, random_roi_count=3
    )

    # Test conversion
    roi_data = adapter.convert("train")

    # Verify we have ROIs from callback
    assert len(roi_data["roi_images"]) >= 2  # At least 2 from callback
    assert len(roi_data["roi_labels"]) >= 2

    # Verify callback ROIs have correct class
    callback_rois = [
        label
        for label in roi_data["roi_labels"]
        if label["class_name"] == "callback_object"
    ]
    assert len(callback_rois) >= 2


def test_roi_adapter_save_json():
    """Test ROI adapter save functionality with JSON format."""
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

    # Create ROI adapter
    adapter = ROIAdapter(coco_data=coco_data)

    # Convert to ROI format
    roi_data = adapter.convert("train")

    # Test saving to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter.save(roi_data, temp_dir)

        # Verify files were created
        temp_path = Path(temp_dir)

        # Check that roi_labels.json exists
        labels_file = temp_path / "roi_labels.json"
        assert labels_file.exists()

        # Check that class_mapping.json exists
        mapping_file = temp_path / "class_mapping.json"
        assert mapping_file.exists()

        # Check that statistics.json exists
        stats_file = temp_path / "statistics.json"
        assert stats_file.exists()

        # Verify JSON content
        with open(labels_file, "r") as f:
            saved_data = json.load(f)
            assert "roi_labels" in saved_data
            assert "class_mapping" in saved_data
            assert "statistics" in saved_data
            assert len(saved_data["roi_labels"]) == 1


def test_roi_adapter_configuration():
    """Test ROI adapter with different configuration options."""
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

    # Test with different configurations
    adapter = ROIAdapter(
        coco_data=coco_data,
        roi_padding=0.2,  # 20% padding
        min_roi_size=64,  # Larger minimum size
        background_class="empty",
        save_format="png",
        quality=90,
    )

    roi_data = adapter.convert("train")

    # Verify configuration was applied
    assert len(roi_data["roi_images"]) == 1

    # Check that padding was applied (bbox should be larger than original)
    roi_image = roi_data["roi_images"][0]
    original_bbox = roi_image["original_bbox"]
    padded_bbox = roi_image["bbox"]

    # Padded bbox should be larger than original
    original_w = original_bbox[2]
    original_h = original_bbox[3]
    padded_w = padded_bbox[2] - padded_bbox[0]
    padded_h = padded_bbox[3] - padded_bbox[1]

    assert padded_w > original_w
    assert padded_h > original_h


if __name__ == "__main__":
    pytest.main([__file__])
