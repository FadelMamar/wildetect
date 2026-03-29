"""
Tests for ROI filtering functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from wildata.filters import (
    FilterConfig,
    FilterPipeline,
    ROIFilterPipeline,
    ROIToCOCOConverter,
)


def create_sample_roi_data() -> Dict[str, Any]:
    """Create sample ROI data for testing."""
    roi_images = [
        {
            "roi_id": 1,
            "roi_filename": "sample_roi_000001.jpg",
            "original_image_path": "/path/to/original/image1.jpg",
            "original_image_id": 1,
            "bbox": [10, 10, 138, 138],  # [x1, y1, x2, y2]
            "original_bbox": [50, 50, 100, 100],  # [x, y, w, h]
            "width": 128,
            "height": 128,
        },
        {
            "roi_id": 2,
            "roi_filename": "sample_roi_000002.jpg",
            "original_image_path": "/path/to/original/image2.jpg",
            "original_image_id": 2,
            "bbox": [20, 20, 148, 148],
            "original_bbox": [60, 60, 120, 120],
            "width": 128,
            "height": 128,
        },
    ]

    roi_labels = [
        {
            "roi_id": 1,
            "class_id": 1,
            "class_name": "person",
            "original_annotation_id": 101,
            "file_name": "sample_roi_000001.jpg",
        },
        {
            "roi_id": 2,
            "class_id": 2,
            "class_name": "car",
            "original_annotation_id": 102,
            "file_name": "sample_roi_000002.jpg",
        },
    ]

    class_mapping = {
        1: "person",
        2: "car",
    }

    return {
        "roi_images": roi_images,
        "roi_labels": roi_labels,
        "class_mapping": class_mapping,
        "statistics": {
            "total_images": 2,
            "annotated_images": 2,
            "unannotated_images": 0,
            "total_rois": 2,
            "rois_from_annotations": 2,
            "rois_from_callback": 0,
            "rois_from_random": 0,
        },
    }


def test_roi_to_coco_converter():
    """Test ROI to COCO conversion."""
    roi_data = create_sample_roi_data()
    converter = ROIToCOCOConverter(roi_data)

    # Convert to COCO-like format
    coco_data = converter.convert_to_coco_like()

    # Check structure
    assert "images" in coco_data
    assert "annotations" in coco_data
    assert "categories" in coco_data

    # Check images
    assert len(coco_data["images"]) == 2
    assert coco_data["images"][0]["id"] == 1
    assert coco_data["images"][0]["file_name"] == "sample_roi_000001.jpg"
    assert coco_data["images"][0]["width"] == 128
    assert coco_data["images"][0]["height"] == 128

    # Check annotations
    assert len(coco_data["annotations"]) == 2
    assert coco_data["annotations"][0]["image_id"] == 1
    assert coco_data["annotations"][0]["category_id"] == 1
    assert len(coco_data["annotations"][0]["bbox"]) == 4  # [x, y, w, h]

    # Check categories
    assert len(coco_data["categories"]) == 2
    assert coco_data["categories"][0]["id"] == 1
    assert coco_data["categories"][0]["name"] == "person"


def test_roi_to_coco_converter_roundtrip():
    """Test converting ROI to COCO and back."""
    roi_data = create_sample_roi_data()
    converter = ROIToCOCOConverter(roi_data)

    # Convert to COCO-like format
    coco_data = converter.convert_to_coco_like()

    # Convert back to ROI format
    filtered_roi_data = converter.convert_filtered_coco_to_roi(coco_data)

    # Check that we get back the same number of items
    assert len(filtered_roi_data["roi_images"]) == len(roi_data["roi_images"])
    assert len(filtered_roi_data["roi_labels"]) == len(roi_data["roi_labels"])


def test_roi_filter_pipeline_creation():
    """Test creating ROI filter pipeline."""
    # Create a simple filter config
    config = FilterConfig()
    config.clustering.enabled = True
    config.clustering.x_percent = 0.5

    # Create filter pipeline
    filter_pipeline = FilterPipeline.from_config(config)

    # Create ROI filter pipeline
    roi_filter_pipeline = ROIFilterPipeline(filter_pipeline)

    assert roi_filter_pipeline is not None
    assert roi_filter_pipeline.filter_pipeline is not None


def test_roi_filter_pipeline_filtering():
    """Test ROI filtering with a simple pass-through filter."""
    # Create sample ROI data
    roi_data = create_sample_roi_data()

    # Create a filter config that doesn't actually filter (keep all data)
    config = FilterConfig()
    config.clustering.enabled = False
    config.quality.size_filter_enabled = False

    # Create filter pipeline
    filter_pipeline = FilterPipeline.from_config(config)
    roi_filter_pipeline = ROIFilterPipeline(filter_pipeline)

    # Filter ROI data
    filtered_roi_data = roi_filter_pipeline.filter_roi_data(roi_data)

    # Should keep all data since no filters are enabled
    assert len(filtered_roi_data["roi_images"]) == len(roi_data["roi_images"])
    assert len(filtered_roi_data["roi_labels"]) == len(roi_data["roi_labels"])


def test_roi_filter_pipeline_save():
    """Test saving filtered ROI data."""
    roi_data = create_sample_roi_data()

    # Create filter pipeline
    config = FilterConfig()
    filter_pipeline = FilterPipeline.from_config(config)
    roi_filter_pipeline = ROIFilterPipeline(filter_pipeline)

    # Filter ROI data
    filtered_roi_data = roi_filter_pipeline.filter_roi_data(roi_data)

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "filtered"

        roi_filter_pipeline.save_filtered_roi_data(
            filtered_roi_data=filtered_roi_data,
            output_labels_dir=output_dir / "labels",
            output_images_dir=output_dir / "images",
            copy_images=False,  # Don't copy images in test
        )

        # Check that files were created
        assert (output_dir / "labels" / "filtered_roi_labels.json").exists()
        assert (output_dir / "labels" / "filtered_class_mapping.json").exists()
        assert (output_dir / "labels" / "filter_statistics.json").exists()


def test_roi_to_coco_converter_from_files():
    """Test creating converter from files."""
    roi_data = create_sample_roi_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save ROI data to files
        labels_file = temp_path / "roi_labels.json"
        class_mapping_file = temp_path / "class_mapping.json"
        statistics_file = temp_path / "statistics.json"

        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(roi_data["roi_labels"], f, indent=2)

        with open(class_mapping_file, "w", encoding="utf-8") as f:
            json.dump(roi_data["class_mapping"], f, indent=2)

        with open(statistics_file, "w", encoding="utf-8") as f:
            json.dump(roi_data["statistics"], f, indent=2)

        # Create converter from files
        converter = ROIToCOCOConverter.from_roi_files(labels_file)

        # Test conversion
        coco_data = converter.convert_to_coco_like()
        assert len(coco_data["images"]) == 2
        assert len(coco_data["annotations"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])
