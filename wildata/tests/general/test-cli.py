#!/usr/bin/env python3
"""
Test script for the wildata CLI.
This script tests the CLI functionality without actually importing datasets.
"""

import sys
from pathlib import Path

from wildata.cli import ImportDatasetConfig, ROIConfigCLI


def test_config_validation():
    """Test the Pydantic configuration validation."""
    print("🧪 Testing configuration validation...")

    # Test valid configuration
    try:
        config = ImportDatasetConfig(
            source_path="examples/example.py",  # Use existing file for testing
            source_format="coco",
            dataset_name="test_dataset",
        )
        print("✅ Valid configuration created successfully")
        print(f"   Dataset name: {config.dataset_name}")
        print(f"   Source format: {config.source_format}")
        print(f"   Processing mode: {config.processing_mode}")
    except Exception as e:
        print(f"❌ Failed to create valid configuration: {e}")
        return False

    # Test invalid source format
    try:
        config = ImportDatasetConfig(
            source_path="examples/example.py",
            source_format="invalid_format",
            dataset_name="test_dataset",
        )
        print("❌ Should have failed with invalid format")
        return False
    except Exception as e:
        print("✅ Correctly rejected invalid source format")

    # Test invalid split name
    try:
        config = ImportDatasetConfig(
            source_path="examples/example.py",
            source_format="coco",
            dataset_name="test_dataset",
            split_name="invalid_split",
        )
        print("❌ Should have failed with invalid split")
        return False
    except Exception as e:
        print("✅ Correctly rejected invalid split name")

    # Test ROI configuration
    try:
        roi_config = ROIConfigCLI(
            random_roi_count=5,
            roi_box_size=64,
            min_roi_size=16,
            dark_threshold=0.7,
            background_class="background",
            save_format="jpg",
            quality=90,
        )
        print("✅ ROI configuration created successfully")
    except Exception as e:
        print(f"❌ Failed to create ROI configuration: {e}")
        return False

    return True


def test_yaml_config():
    """Test YAML configuration loading."""
    print("\n🧪 Testing YAML configuration...")

    # Create a test config
    config = ImportDatasetConfig(
        source_path="examples/example.py",
        source_format="coco",
        dataset_name="yaml_test_dataset",
        processing_mode="batch",
        track_with_dvc=True,
    )

    # Save to YAML
    yaml_path = "test_config.yaml"
    try:
        config.to_yaml(yaml_path)
        print("✅ Configuration saved to YAML")

        # Load from YAML
        loaded_config = ImportDatasetConfig.from_yaml(yaml_path)
        print("✅ Configuration loaded from YAML")
        print(f"   Dataset name: {loaded_config.dataset_name}")
        print(f"   Processing mode: {loaded_config.processing_mode}")

        # Clean up
        Path(yaml_path).unlink()
        print("✅ Test file cleaned up")

    except Exception as e:
        print(f"❌ YAML test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("🚀 Starting wildata CLI tests...\n")

    success = True
    success &= test_config_validation()
    success &= test_yaml_config()

    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
