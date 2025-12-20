"""
Test helper utilities for WildTrain integration tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import numpy as np
import torch
from PIL import Image


def create_temp_dataset(num_images: int = 10, image_size: Tuple[int, int] = (224, 224)) -> str:
    """Create a temporary dataset with mock images and annotations."""
    temp_dir = tempfile.mkdtemp(prefix="wildtrain_test_dataset_")
    dataset_path = Path(temp_dir)
    
    # Create images directory
    images_dir = dataset_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create mock images
    for i in range(num_images):
        img = Image.new('RGB', image_size, color=(i * 25, i * 25, i * 25))
        img.save(images_dir / f"image_{i:03d}.jpg")
    
    # Create annotations directory
    annotations_dir = dataset_path / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    # Create COCO format annotations
    annotations = {
        "images": [
            {
                "id": i,
                "file_name": f"image_{i:03d}.jpg",
                "width": image_size[0],
                "height": image_size[1]
            }
            for i in range(num_images)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": i % 3,  # 3 classes
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0
            }
            for i in range(num_images)
        ],
        "categories": [
            {"id": 0, "name": "class_0"},
            {"id": 1, "name": "class_1"},
            {"id": 2, "name": "class_2"}
        ]
    }
    
    # Save annotations
    with open(annotations_dir / "train.json", 'w') as f:
        json.dump(annotations, f)
    
    return str(dataset_path)


def cleanup_temp_dataset(dataset_path: str):
    """Clean up temporary dataset directory."""
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path, ignore_errors=True)


def create_mock_classification_annotations(num_samples: int = 20, num_classes: int = 3) -> List[Dict]:
    """Create mock classification annotations."""
    return [
        {
            "id": i,
            "file_name": f"image_{i:03d}.jpg",
            "class_id": i % num_classes,
            "class_name": f"class_{i % num_classes}"
        }
        for i in range(num_samples)
    ]


def create_mock_detection_annotations(num_samples: int = 10, num_classes: int = 3) -> Dict:
    """Create mock detection annotations in COCO format."""
    return {
        "images": [
            {
                "id": i,
                "file_name": f"image_{i:03d}.jpg",
                "width": 224,
                "height": 224
            }
            for i in range(num_samples)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": i % num_classes,
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0
            }
            for i in range(num_samples)
        ],
        "categories": [
            {"id": i, "name": f"class_{i}"}
            for i in range(num_classes)
        ]
    }


def create_mock_model_checkpoint(num_classes: int = 3, model_path: str = None) -> str:
    """Create a mock model checkpoint."""
    if model_path is None:
        model_path = tempfile.mktemp(suffix=".ckpt", prefix="wildtrain_test_model_")
    
    # Create a simple mock checkpoint
    checkpoint = {
        "state_dict": {
            "backbone.weight": torch.randn(64, 3, 7, 7),
            "classifier.weight": torch.randn(num_classes, 64),
            "classifier.bias": torch.randn(num_classes)
        },
        "hyper_parameters": {
            "num_classes": num_classes,
            "learning_rate": 0.001,
            "backbone": "resnet18"
        }
    }
    
    torch.save(checkpoint, model_path)
    return model_path


def cleanup_mock_model_checkpoint(model_path: str):
    """Clean up mock model checkpoint."""
    if os.path.exists(model_path):
        os.remove(model_path)


def validate_class_distribution(annotations: List[Dict], expected_classes: int = None) -> Dict[int, int]:
    """Validate and return class distribution from annotations."""
    class_counts = {}
    for ann in annotations:
        class_id = ann.get('class_id', ann.get('category_id'))
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    if expected_classes is not None:
        assert len(class_counts) == expected_classes, f"Expected {expected_classes} classes, got {len(class_counts)}"
    
    return class_counts


def validate_crop_dataset_structure(crop_dataset, expected_crops: int = None):
    """Validate PatchDataset structure and properties."""
    assert hasattr(crop_dataset, '__len__'), "PatchDataset must have __len__ method"
    assert hasattr(crop_dataset, '__getitem__'), "PatchDataset must have __getitem__ method"
    
    if expected_crops is not None:
        assert len(crop_dataset) == expected_crops, f"Expected {expected_crops} crops, got {len(crop_dataset)}"
    
    # Test getting a sample
    if len(crop_dataset) > 0:
        sample = crop_dataset[0]
        assert isinstance(sample, (tuple, list)), "PatchDataset should return tuple/list"
        assert len(sample) == 2, "PatchDataset should return (crop, label)"


def validate_dataloader_compatibility(dataset, batch_size: int = 4):
    """Validate that dataset is compatible with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Test loading a batch
    batch = next(iter(dataloader))
    assert isinstance(batch, (tuple, list)), "DataLoader should return tuple/list"
    assert len(batch) == 2, "DataLoader should return (data, labels)"
    
    return dataloader


def calculate_class_distribution_change(before_annotations: List[Dict], after_annotations: List[Dict]) -> Dict:
    """Calculate the change in class distribution after filtering."""
    before_dist = validate_class_distribution(before_annotations)
    after_dist = validate_class_distribution(after_annotations)
    
    changes = {}
    all_classes = set(before_dist.keys()) | set(after_dist.keys())
    
    for class_id in all_classes:
        before_count = before_dist.get(class_id, 0)
        after_count = after_dist.get(class_id, 0)
        change = after_count - before_count
        change_percent = (change / before_count * 100) if before_count > 0 else 0
        changes[class_id] = {
            "before": before_count,
            "after": after_count,
            "change": change,
            "change_percent": change_percent
        }
    
    return changes


def create_test_config(config_type: str = "classification") -> Dict[str, Any]:
    """Create test configuration based on type."""
    base_config = {
        "data": {
            "batch_size": 4,
            "num_workers": 0,
            "debug": True
        },
        "training": {
            "max_epochs": 2,
            "debug": True
        }
    }
    
    if config_type == "classification":
        base_config.update({
            "model": {
                "backbone": "resnet18",
                "num_classes": 3
            }
        })
    elif config_type == "detection":
        base_config.update({
            "model": {
                "weights": "yolov8n.pt",
                "imgsz": 224,
                "device": "cpu"
            }
        })
    
    return base_config


def mock_external_dependencies():
    """Mock external dependencies for testing."""
    # This would be implemented based on specific external dependencies
    # For now, we'll just set environment variables
    os.environ["WILDTRAIN_TEST_MODE"] = "true"
    os.environ["WILDTRAIN_DEBUG_MODE"] = "true"


def cleanup_external_dependencies():
    """Clean up mocked external dependencies."""
    if "WILDTRAIN_TEST_MODE" in os.environ:
        del os.environ["WILDTRAIN_TEST_MODE"]
    if "WILDTRAIN_DEBUG_MODE" in os.environ:
        del os.environ["WILDTRAIN_DEBUG_MODE"]


def assert_approximately_equal(actual: float, expected: float, tolerance: float = 0.01):
    """Assert that two values are approximately equal within tolerance."""
    assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual}"


def assert_list_length(actual_list: List, expected_length: int, list_name: str = "list"):
    """Assert that a list has the expected length."""
    assert len(actual_list) == expected_length, f"Expected {list_name} to have length {expected_length}, got {len(actual_list)}"


def assert_dict_keys(actual_dict: Dict, expected_keys: List[str], dict_name: str = "dictionary"):
    """Assert that a dictionary has the expected keys."""
    missing_keys = set(expected_keys) - set(actual_dict.keys())
    extra_keys = set(actual_dict.keys()) - set(expected_keys)
    
    assert not missing_keys, f"Missing keys in {dict_name}: {missing_keys}"
    assert not extra_keys, f"Unexpected keys in {dict_name}: {extra_keys}" 