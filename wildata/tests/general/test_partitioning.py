"""
Tests for the partitioning system.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
from wildata.partitioning.partitioning_pipeline import (
    PartitioningPipeline,
    PartitioningStrategy,
)


def create_test_coco_data() -> Dict[str, Any]:
    """Create test COCO data with spatial metadata."""
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "test_category", "supercategory": "test"},
    ]

    # Create test images with GPS coordinates
    for i in range(20):
        image = {
            "id": i + 1,
            "file_name": f"test_image_{i+1}.jpg",
            "width": 640,
            "height": 480,
            "gps_lat": 1.0 + (i % 4) * 0.01,  # Create spatial groups
            "gps_lon": 1.0 + (i // 4) * 0.01,
            "camp_id": f"camp_{i % 3}",  # 3 different camps
            "dataset_id": f"dataset_{i % 2}",  # 2 different datasets
        }
        images.append(image)

        # Add annotation
        annotation = {
            "id": i + 1,
            "image_id": i + 1,
            "category_id": 1,
            "bbox": [100, 100, 200, 150],
            "area": 30000,
            "iscrowd": 0,
        }
        annotations.append(annotation)

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def test_spatial_partitioning():
    """Test spatial partitioning strategy."""
    coco_data = create_test_coco_data()

    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.SPATIAL,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
        spatial_threshold=0.02,
    )

    # Test partitioning
    split_data = pipeline.apply_partitioning_to_coco_data(coco_data)

    # Verify splits exist
    assert "train" in split_data
    assert "val" in split_data
    assert "test" in split_data

    # Verify all images are included
    total_images = (
        len(split_data["train"]["images"])
        + len(split_data["val"]["images"])
        + len(split_data["test"]["images"])
    )
    assert total_images == len(coco_data["images"])

    # Verify no overlap between splits
    train_ids = {img["id"] for img in split_data["train"]["images"]}
    val_ids = {img["id"] for img in split_data["val"]["images"]}
    test_ids = {img["id"] for img in split_data["test"]["images"]}

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_camp_based_partitioning():
    """Test camp-based partitioning strategy."""
    coco_data = create_test_coco_data()

    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.CAMP_BASED,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
        camp_metadata_key="camp_id",
    )

    # Test partitioning
    split_data = pipeline.apply_partitioning_to_coco_data(coco_data)

    # Verify splits exist
    assert "train" in split_data
    assert "val" in split_data
    assert "test" in split_data

    # Verify all images are included
    total_images = (
        len(split_data["train"]["images"])
        + len(split_data["val"]["images"])
        + len(split_data["test"]["images"])
    )
    assert total_images == len(coco_data["images"])


def test_hybrid_partitioning():
    """Test hybrid partitioning strategy."""
    coco_data = create_test_coco_data()

    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.HYBRID,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
        spatial_threshold=0.02,
        camp_metadata_key="camp_id",
        metadata_keys=["dataset_id", "camp_id"],
    )

    # Test partitioning
    split_data = pipeline.apply_partitioning_to_coco_data(coco_data)

    # Verify splits exist
    assert "train" in split_data
    assert "val" in split_data
    assert "test" in split_data

    # Verify all images are included
    total_images = (
        len(split_data["train"]["images"])
        + len(split_data["val"]["images"])
        + len(split_data["test"]["images"])
    )
    assert total_images == len(coco_data["images"])


def test_statistics():
    """Test statistics generation."""
    coco_data = create_test_coco_data()

    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.SPATIAL,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
    )

    # Get statistics
    stats = pipeline.get_statistics(coco_data["images"])

    # Verify basic statistics
    assert "strategy" in stats
    assert "test_size" in stats
    assert "val_size" in stats
    assert "total_images" in stats
    assert stats["total_images"] == 20


def test_configuration_save_load():
    """Test configuration save and load."""
    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.SPATIAL,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        spatial_threshold=0.01,
    )

    # Save configuration
    config_path = "test_partitioning_config.json"
    pipeline.save_partitioning_config(config_path)

    # Load configuration
    loaded_pipeline = PartitioningPipeline.from_config(config_path)

    # Verify loaded configuration
    assert loaded_pipeline.strategy == PartitioningStrategy.SPATIAL
    assert loaded_pipeline.test_size == 0.2
    assert loaded_pipeline.val_size == 0.2
    assert loaded_pipeline.random_state == 42

    # Clean up
    import os

    if os.path.exists(config_path):
        os.remove(config_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
