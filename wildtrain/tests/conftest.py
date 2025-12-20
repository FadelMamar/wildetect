"""
Pytest configuration and fixtures for WildTrain integration tests.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
from PIL import Image
import torch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="wildtrain_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def real_dataset_path():
    """Provide path to real dataset for testing."""
    # Use the actual demo dataset path
    dataset_path = Path(r"D:\workspace\data\demo-dataset")
    
    if not dataset_path.exists():
        pytest.skip(f"Real dataset not found at {dataset_path}")
    
    return str(dataset_path)


@pytest.fixture(scope="session")
def real_dataset_datasets_path():
    """Provide path to the datasets subdirectory."""
    # Use the actual datasets subdirectory
    datasets_path = Path(r"D:\workspace\data\demo-dataset\datasets")
    
    if not datasets_path.exists():
        pytest.skip(f"Real datasets directory not found at {datasets_path}")
    
    return str(datasets_path)


@pytest.fixture(scope="session")
def mock_model_checkpoint(test_data_dir):
    """Provide path to mock model checkpoint."""
    checkpoint_path = Path(test_data_dir) / "mock_model.ckpt"
    
    # Create a simple mock checkpoint
    checkpoint = {
        "state_dict": {
            "backbone.weight": torch.randn(64, 3, 7, 7),
            "classifier.weight": torch.randn(3, 64),
            "classifier.bias": torch.randn(3)
        },
        "hyper_parameters": {
            "num_classes": 3,
            "learning_rate": 0.001
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture(scope="session")
def test_config():
    """Provide basic test configuration."""
    return {
        "data": {
            "root_data_directory": r"D:\workspace\data\demo-dataset",
            "batch_size": 4,
            "num_workers": 0
        },
        "model": {
            "backbone": "resnet18",
            "num_classes": 3
        },
        "training": {
            "max_epochs": 2,
            "debug": True
        }
    }


@pytest.fixture(scope="session")
def mock_fiftyone_session():
    """Mock FiftyOne session for testing."""
    # This would be mocked in actual implementation
    return None


@pytest.fixture(scope="function")
def temp_dir():
    """Provide a temporary directory for each test."""
    temp_dir = tempfile.mkdtemp(prefix="wildtrain_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_classification_annotations():
    """Provide mock classification annotations."""
    return [
        {"id": i, "file_name": f"image_{i:03d}.jpg", "class_id": i % 3, "class_name": f"class_{i % 3}"}
        for i in range(20)
    ]


@pytest.fixture(scope="session")
def mock_detection_annotations():
    """Provide mock detection annotations in COCO format."""
    return {
        "images": [
            {
                "id": i,
                "file_name": f"image_{i:03d}.jpg",
                "width": 224,
                "height": 224
            }
            for i in range(10)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": i % 3,
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0
            }
            for i in range(10)
        ],
        "categories": [
            {"id": 0, "name": "class_0"},
            {"id": 1, "name": "class_1"},
            {"id": 2, "name": "class_2"}
        ]
    }


@pytest.fixture(scope="session")
def mock_yolo_dataset(test_data_dir):
    """Provide mock YOLO format dataset."""
    dataset_path = Path(test_data_dir) / "mock_yolo_dataset"
    dataset_path.mkdir(exist_ok=True)
    
    # Create images
    images_dir = dataset_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    for i in range(10):
        img = Image.new('RGB', (224, 224), color=(i * 25, i * 25, i * 25))
        img.save(images_dir / f"image_{i:03d}.jpg")
    
    # Create labels
    labels_dir = dataset_path / "labels"
    labels_dir.mkdir(exist_ok=True)
    
    for i in range(10):
        # YOLO format: class_id center_x center_y width height
        label_content = f"{i % 3} 0.5 0.5 0.4 0.4\n"
        with open(labels_dir / f"image_{i:03d}.txt", 'w') as f:
            f.write(label_content)
    
    # Create data.yaml
    data_yaml = {
        "train": str(images_dir),
        "val": str(images_dir),
        "nc": 3,
        "names": ["class_0", "class_1", "class_2"]
    }
    
    with open(dataset_path / "data.yaml", 'w') as f:
        import yaml
        yaml.dump(data_yaml, f)
    
    return str(dataset_path)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test mode
    os.environ["WILDTRAIN_TEST_MODE"] = "true"
    os.environ["WILDTRAIN_DEBUG_MODE"] = "true"
    
    # Set PYTHONPATH if not already set
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent.parent / "src")
    
    yield
    
    # Cleanup
    if "WILDTRAIN_TEST_MODE" in os.environ:
        del os.environ["WILDTRAIN_TEST_MODE"]
    if "WILDTRAIN_DEBUG_MODE" in os.environ:
        del os.environ["WILDTRAIN_DEBUG_MODE"] 