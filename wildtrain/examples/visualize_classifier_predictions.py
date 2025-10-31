import fiftyone as fo
import sys
import os
from wildtrain.visualization import add_predictions_from_classifier
# Example configuration
DATASET_NAME = "wildtrain_classification_example"
DATA_ROOT = "D:/workspace/data/demo-dataset"  # Update this to your data root
CHECKPOINT_PATH = "D:/workspace/repos/wildtrain/checkpoints/classification/best-v5.ckpt"  # Update this to your checkpoint
CONFIG_PATH = None  # Or path to your OmegaConf YAML config
SPLIT = "val"  # or "train"/"test"


if __name__ == "__main__":
    

    # 3. Add predictions from classifier checkpoint
    add_predictions_from_classifier(
        dataset_name=DATASET_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        prediction_field="predictions",
        batch_size=32,
        device="cpu",
        debug=True
    )

    # 4. Launch FiftyOne app for visualization
    session = fo.launch_app()
    print(f"FiftyOne app launched for dataset: {DATASET_NAME}")
    session.wait() 