# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:46:54 2025

@author: FADELCO
"""

from omegaconf import OmegaConf, DictConfig
from wildtrain.trainers import UltralyticsDetectionTrainer
import traceback


def main():
    """
    Main function to run classification training with different configurations.
    """
    print("🚀 Starting WildTrain YOLO Training Example")
    print("=" * 60)

    # Create output directories
    print("-" * 40)

    config = OmegaConf.load(r"D:/workspace/repos/wildtrain/configs/detection/yolo_configs/yolo.yaml")

    # print("Configuration:")
    # print(OmegaConf.to_yaml(config))

    try:
        trainer = UltralyticsDetectionTrainer(DictConfig(config))
        trainer.run()
        print("✅ Training completed successfully!")
    except Exception:
        print(f"❌ Training failed: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
