#!/usr/bin/env python3
"""
Test script to verify curriculum learning integration with classification trainer.
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wildtrain.trainers.classification_trainer import ClassifierTrainer
from wildtrain.data import ClassificationDataModule
from wildtrain.shared.models import CurriculumConfig


def test_curriculum_config():
    """Test curriculum configuration creation."""
    print("Testing curriculum configuration...")
    
    # Test basic curriculum config
    config = CurriculumConfig(
        enabled=True,
        type="difficulty",
        difficulty_strategy="linear",
        start_difficulty=0.0,
        end_difficulty=1.0,
        warmup_epochs=2,
        log_frequency=1
    )
    
    print(f"✓ Curriculum config created: {config}")
    return config


def test_datamodule_curriculum():
    """Test datamodule with curriculum integration."""
    print("\nTesting datamodule curriculum integration...")
    
    # Create curriculum config
    curriculum_config = test_curriculum_config()
    
    # Test datamodule creation with curriculum
    try:
        datamodule = ClassificationDataModule(
            root_data_directory="D:/workspace/data/demo-dataset",
            batch_size=8,
            dataset_type="crop",
            curriculum_config=curriculum_config,
            compute_difficulties=True,
            preserve_aspect_ratio=True
        )
        
        print("✓ ClassificationDataModule created with curriculum")
        print(f"✓ Curriculum enabled: {datamodule.is_curriculum_enabled()}")
        
        # Test curriculum state
        if datamodule.is_curriculum_enabled():
            datamodule.setup_curriculum(10)  # 10 epochs
            state = datamodule.get_curriculum_state()
            print(f"✓ Curriculum state: {state}")
        
        return datamodule
        
    except Exception as e:
        print(f"✗ Error creating datamodule with curriculum: {e}")
        return None


def test_curriculum_callback():
    """Test curriculum callback creation."""
    print("\nTesting curriculum callback...")
    
    datamodule = test_datamodule_curriculum()
    if datamodule is None:
        return None
    
    try:
        callback = CurriculumCallback(datamodule)
        print("✓ CurriculumCallback created successfully")
        return callback
    except Exception as e:
        print(f"✗ Error creating curriculum callback: {e}")
        return None


def test_trainer_integration():
    """Test trainer integration with curriculum."""
    print("\nTesting trainer integration...")
    
    try:
        # Create a simple config for testing
        config_dict = {
            "dataset": {
                "root_data_directory": "D:/workspace/data/demo-dataset",
                "batch_size": 8,
                "dataset_type": "crop",
                "curriculum_config": {
                    "enabled": True,
                    "type": "difficulty",
                    "difficulty_strategy": "linear",
                    "start_difficulty": 0.0,
                    "end_difficulty": 1.0,
                    "warmup_epochs": 2,
                    "log_frequency": 1
                }
            },
            "model": {
                "backbone": "resnet18",
                "backbone_source": "timm",
                "pretrained": True,
                "dropout": 0.2,
                "freeze_backbone": True,
                "input_size": 224,
                "mean": [0.554, 0.469, 0.348],
                "std": [0.203, 0.173, 0.144]
            },
            "train": {
                "epochs": 5,
                "lr": 1e-3,
                "lrf": 1e-2,
                "weight_decay": 5e-3,
                "label_smoothing": 0.0,
                "accelerator": "auto",
                "precision": "16-mixed"
            },
            "checkpoint": {
                "monitor": "val_loss",
                "save_top_k": 1,
                "save_last": True,
                "mode": "min",
                "dirpath": "checkpoints",
                "filename": "best_classifier",
                "save_weights_only": False,
                "patience": 10,
                "min_delta": 0.001
            },
            "mlflow": {
                "experiment_name": "curriculum_test",
                "run_name": "curriculum_classification_test",
                "log_model": False
            }
        }
        
        from omegaconf import OmegaConf
        config = OmegaConf.create(config_dict)
        
        # Test trainer creation
        trainer = ClassifierTrainer(config)
        print("✓ ClassifierTrainer created with curriculum config")
        
        # Test callback creation
        callbacks, mlflow_logger = trainer.get_callbacks()
        print(f"✓ Base callbacks created: {len(callbacks)} callbacks")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Error in trainer integration: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Curriculum Learning Integration")
    print("=" * 60)
    
    # Run tests
    test_curriculum_config()
    test_datamodule_curriculum()
    test_curriculum_callback()
    test_trainer_integration()
    
    print("\n" + "=" * 60)
    print("Curriculum integration test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()