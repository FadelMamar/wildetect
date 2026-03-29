#!/usr/bin/env python3
"""
Example script for training a classifier using WildTrain framework.

This script demonstrates how to:
1. Load training configurations from YAML files
2. Run classification training with different configurations
3. Use different backbones and configurations

Usage:
    python examples/train_classifier.py
"""

import traceback

from omegaconf import DictConfig, OmegaConf

from wildtrain.trainers import ClassifierTrainer


def main():
    """
    Main function to run classification training with different configurations.
    """
    print("🚀 Starting WildTrain Classification Training Example")
    print("=" * 60)

    # Create output directories
    print("-" * 40)

    config = OmegaConf.load(
        r"D:\workspace\repos\wildtrain\configs\classification\classification_train.yaml"
    )

    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    try:
        trainer = ClassifierTrainer(DictConfig(config))
        trainer.run(debug=True)
        print("✅ Training completed successfully!")
    except Exception:
        print(f"❌ Training failed: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
