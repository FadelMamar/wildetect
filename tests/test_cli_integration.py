#!/usr/bin/env python3
"""
Test script to verify CLI integration with CampaignManager.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test that all necessary imports work."""
    try:
        from src.wildetect.core.campaign_manager import CampaignConfig, CampaignManager
        from src.wildetect.core.config import (
            FlightSpecs,
            LoaderConfig,
            PredictionConfig,
        )
        from src.wildetect.core.data.census import CensusDataManager, DetectionResults
        from src.wildetect.core.detection_pipeline import DetectionPipeline

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test that configurations can be created correctly."""
    try:
        from src.wildetect.core.campaign_manager import CampaignConfig
        from src.wildetect.core.config import (
            FlightSpecs,
            LoaderConfig,
            PredictionConfig,
        )

        # Test LoaderConfig
        loader_config = LoaderConfig(
            tile_size=640,
            batch_size=4,
            flight_specs=FlightSpecs(
                sensor_height=24.0, focal_length=35.0, flight_height=180.0
            ),
        )
        print("✓ LoaderConfig created successfully")

        # Test PredictionConfig
        pred_config = PredictionConfig(
            model_path=r"D:\workspace\repos\wildetect\weights\best.onnx",
            model_type="yolo",
            confidence_threshold=0.25,
            device="cpu",
            batch_size=4,
            tilesize=640,
            cls_imgsz=224,
            verbose=True,
        )
        print("✓ PredictionConfig created successfully")

        # Test CampaignConfig
        campaign_config = CampaignConfig(
            campaign_id="test_campaign",
            loader_config=loader_config,
            prediction_config=pred_config,
            metadata={"test": "data"},
            fiftyone_dataset_name="test_dataset",
        )
        print("✓ CampaignConfig created successfully")

        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False


def test_campaign_manager():
    """Test that CampaignManager can be instantiated."""
    try:
        from src.wildetect.core.campaign_manager import CampaignConfig, CampaignManager
        from src.wildetect.core.config import (
            FlightSpecs,
            LoaderConfig,
            PredictionConfig,
        )

        # Create configurations
        loader_config = LoaderConfig(
            tile_size=640,
            batch_size=4,
            flight_specs=FlightSpecs(
                sensor_height=24.0, focal_length=35.0, flight_height=180.0
            ),
        )

        pred_config = PredictionConfig(
            model_path=r"D:\workspace\repos\wildetect\weights\best.onnx",
            model_type="yolo",
            confidence_threshold=0.25,
            roi_weights=r"D:\workspace\repos\wildetect\weights\roi_classifier.torchscript",
            device="cpu",
            batch_size=4,
            tilesize=640,
            cls_imgsz=96,
            verbose=True,
        )

        campaign_config = CampaignConfig(
            campaign_id="test_campaign",
            loader_config=loader_config,
            prediction_config=pred_config,
            metadata={"test": "data"},
            fiftyone_dataset_name="test_dataset",
        )

        # Create campaign manager
        campaign_manager = CampaignManager(campaign_config)
        print("✓ CampaignManager created successfully")

        return True
    except Exception as e:
        print(f"✗ CampaignManager creation failed: {e}")
        return False


def test_detection_pipeline():
    """Test that DetectionPipeline can be created."""
    try:
        from src.wildetect.core.config import LoaderConfig, PredictionConfig
        from src.wildetect.core.detection_pipeline import DetectionPipeline

        # Create configurations
        loader_config = LoaderConfig(tile_size=640, batch_size=4)

        pred_config = PredictionConfig(
            model_path=r"D:\workspace\repos\wildetect\weights\best.onnx",
            model_type="yolo",
            confidence_threshold=0.25,
            device="cpu",
            batch_size=4,
            tilesize=640,
        )

        # Create detection pipeline
        pipeline = DetectionPipeline(
            config=pred_config, loader_config=loader_config, device="cpu"
        )
        print("✓ DetectionPipeline created successfully")

        return True
    except Exception as e:
        print(f"✗ DetectionPipeline creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing CLI Integration...")
    print("=" * 50)

    tests = [
        test_imports,
        test_config_creation,
        test_campaign_manager,
        test_detection_pipeline,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! CLI integration looks good.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
