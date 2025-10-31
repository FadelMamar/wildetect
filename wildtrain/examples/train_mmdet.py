#!/usr/bin/env python3

from omegaconf import OmegaConf, DictConfig
from wildtrain.trainers import MMDetectionTrainer
import traceback


def main():
    """
    Main function to run classification training with different configurations.
    """
    print("🚀 Starting WildTrain Classification Training Example")
    print("=" * 60)

    # Create output directories
    print("-" * 40)

    config = OmegaConf.load(
        r"D:\workspace\repos\wildtrain\configs\detection\mmdet.yaml"
    )

    # print("Configuration:")
    # print(OmegaConf.to_yaml(config))

    try:
        trainer = MMDetectionTrainer(DictConfig(config))
        trainer.run(debug=True)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
