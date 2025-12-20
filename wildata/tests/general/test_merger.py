"""
Tests for COCODatasetMerger in wildtrain.partitioning.merger
"""

import copy
from typing import Any, Dict, List

import pytest
from wildata.partitioning.merger import COCODatasetMerger


def make_synthetic_coco_dataset(
    dataset_id: int,
    num_images: int,
    num_categories: int,
    category_names: List[str],
    image_offset: int = 0,
    annotation_offset: int = 0,
    category_offset: int = 0,
    duplicate_images: bool = False,
    supercategory: str = "animal",
) -> Dict[str, Any]:
    """
    Generate a synthetic COCO dataset with specified parameters.
    """
    categories = [
        {
            "id": i + 1 + category_offset,
            "name": category_names[i % len(category_names)],
            "supercategory": supercategory,
        }
        for i in range(num_categories)
    ]
    images = []
    annotations = []
    for i in range(num_images):
        img_id = i + 1 + image_offset
        file_name = (
            f"/abs/path/to/image_{dataset_id}_{i if not duplicate_images else 0}.jpg"
        )
        images.append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": 640,
                "height": 480,
                "camp_id": f"camp_{dataset_id}",
            }
        )
        # Each image gets one annotation per category
        for j, cat in enumerate(categories):
            ann_id = annotation_offset + i * num_categories + j + 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat["id"],
                    "bbox": [10, 10, 100, 100],
                    "area": 10000,
                    "iscrowd": 0,
                }
            )
    return {
        "images": images,
        "annotations": annotations,
        "categories": copy.deepcopy(categories),
        "info": {"description": f"Synthetic dataset {dataset_id}"},
        "licenses": [{"id": 1, "name": "CC-BY"}],
    }


def test_merger_basic():
    """Test merging two simple datasets with unique categories and images."""
    ds1 = make_synthetic_coco_dataset(1, 3, 2, ["lion", "zebra"])
    ds2 = make_synthetic_coco_dataset(
        2,
        2,
        2,
        ["elephant", "giraffe"],
        image_offset=100,
        annotation_offset=1000,
        category_offset=10,
    )
    merger = COCODatasetMerger([ds1, ds2])
    merged = merger.merge()

    # Check total images and annotations
    assert len(merged["images"]) == 5
    assert len(merged["annotations"]) == 10
    # Check all categories are present
    cat_names = {cat["name"] for cat in merged["categories"]}
    assert cat_names == {"lion", "zebra", "elephant", "giraffe"}
    # Check IDs are unique
    image_ids = [img["id"] for img in merged["images"]]
    assert len(image_ids) == len(set(image_ids))
    ann_ids = [ann["id"] for ann in merged["annotations"]]
    assert len(ann_ids) == len(set(ann_ids))


def test_merger_category_harmonization():
    """Test merging datasets with overlapping category names (case-insensitive)."""
    ds1 = make_synthetic_coco_dataset(1, 2, 2, ["Lion", "Zebra"])
    ds2 = make_synthetic_coco_dataset(
        2,
        2,
        2,
        ["lion", "zebra"],
        image_offset=100,
        annotation_offset=1000,
        category_offset=10,
    )
    merger = COCODatasetMerger([ds1, ds2])
    merged = merger.merge()
    # Should only have 2 categories (lion, zebra)
    cat_names = sorted([cat["name"].lower() for cat in merged["categories"]])
    assert cat_names == ["lion", "zebra"]
    # All annotations should map to the correct unified category IDs
    cat_id_map = {cat["name"].lower(): cat["id"] for cat in merged["categories"]}
    for ann in merged["annotations"]:
        cat_id = ann["category_id"]
        assert cat_id in cat_id_map.values()


def test_merger_duplicate_images():
    """Test merging datasets with duplicate images (same file path)."""
    ds1 = make_synthetic_coco_dataset(1, 2, 1, ["lion"])
    ds2 = make_synthetic_coco_dataset(2, 2, 1, ["zebra"], duplicate_images=True)
    # Both datasets will have image_0.jpg as a file path
    merger = COCODatasetMerger([ds1, ds2])
    merged = merger.merge()
    # Only one image with file_name ...image_0.jpg should be present
    file_names = [img["file_name"] for img in merged["images"]]
    assert len(file_names) == len(set(file_names))


def test_merger_annotation_remapping():
    """Test that annotation image_id and category_id are remapped correctly."""
    ds1 = make_synthetic_coco_dataset(1, 1, 1, ["lion"])
    ds2 = make_synthetic_coco_dataset(
        2, 1, 1, ["lion"], image_offset=100, annotation_offset=1000, category_offset=10
    )
    merger = COCODatasetMerger([ds1, ds2])
    merged = merger.merge()
    # All annotation image_ids and category_ids should exist in merged images/categories
    image_ids = {img["id"] for img in merged["images"]}
    category_ids = {cat["id"] for cat in merged["categories"]}
    for ann in merged["annotations"]:
        assert ann["image_id"] in image_ids
        assert ann["category_id"] in category_ids


def test_merger_large():
    """Test merging many datasets with many categories and images."""
    datasets = []
    for i in range(5):
        ds = make_synthetic_coco_dataset(
            dataset_id=i,
            num_images=10,
            num_categories=3,
            category_names=["lion", "zebra", f"species_{i}"],
            image_offset=i * 1000,
            annotation_offset=i * 10000,
            category_offset=i * 100,
        )
        datasets.append(ds)
    merger = COCODatasetMerger(datasets)
    merged = merger.merge()
    # Check total images
    assert len(merged["images"]) == 50
    # Check all unique categories are present
    cat_names = {cat["name"] for cat in merged["categories"]}
    for i in range(5):
        assert f"species_{i}" in cat_names
    assert "lion" in cat_names and "zebra" in cat_names
    # Check annotation count
    assert len(merged["annotations"]) == 50 * 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
