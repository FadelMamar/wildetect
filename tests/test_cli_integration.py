#!/usr/bin/env python3
"""
Test script to verify CLI integration with CampaignManager.
"""

MODEL_PATH = r"D:\workspace\repos\wildetect\weights\best.pt"
ROI_WEIGHTS_PATH = r"D:\workspace\repos\wildetect\weights\roi_classifier.torchscript"


def test_imports():
    """Test that all necessary imports work."""
    try:
        from wildetect.core.campaign_manager import CampaignConfig, CampaignManager
        from wildetect.core.config import (
            FlightSpecs,
            LoaderConfig,
            PredictionConfig,
        )
        from wildetect.core.data.census import CensusDataManager, DetectionResults
        from wildetect.core.detection_pipeline import DetectionPipeline

        print("✓ All imports successful")
        assert True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        assert False, f"Import failed: {e}"


def test_config_creation():
    """Test that configurations can be created correctly."""
    try:
        from wildetect.core.campaign_manager import CampaignConfig
        from wildetect.core.config import (
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
            model_path=MODEL_PATH,
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

        assert True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        assert False, f"Config creation failed: {e}"


def test_campaign_manager():
    """Test that CampaignManager can be instantiated."""
    try:
        from wildetect.core.campaign_manager import CampaignConfig, CampaignManager
        from wildetect.core.config import (
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
            model_path=MODEL_PATH,
            model_type="yolo",
            confidence_threshold=0.25,
            roi_weights=ROI_WEIGHTS_PATH,
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

        assert True
    except Exception as e:
        print(f"✗ CampaignManager creation failed: {e}")
        assert False, f"CampaignManager creation failed: {e}"


def test_detection_pipeline():
    """Test that DetectionPipeline can be created."""
    try:
        from wildetect.core.config import LoaderConfig, PredictionConfig
        from wildetect.core.detection_pipeline import DetectionPipeline

        # Create configurations
        loader_config = LoaderConfig(tile_size=640, batch_size=4)

        pred_config = PredictionConfig(
            model_path=MODEL_PATH,
            model_type="yolo",
            confidence_threshold=0.25,
            device="cpu",
            batch_size=4,
            tilesize=640,
        )

        # Create detection pipeline
        pipeline = DetectionPipeline(config=pred_config, loader_config=loader_config)
        print("✓ DetectionPipeline created successfully")

        assert True
    except Exception as e:
        print(f"✗ DetectionPipeline creation failed: {e}")
        assert False, f"DetectionPipeline creation failed: {e}"


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
        try:
            test()
            passed += 1
        except AssertionError:
            pass
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
    import sys

    sys.exit(main())
