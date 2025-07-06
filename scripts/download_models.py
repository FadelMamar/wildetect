#!/usr/bin/env python3
"""
Download pre-trained models for WildDetect.

This script downloads pre-trained YOLO models that can be used
for wildlife detection from aerial images.
"""

import logging
import os
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.utils.config import create_directories, get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: str, description: str = "Downloading"):
    """Download a file with progress bar.

    Args:
        url: URL to download from
        destination: Local file path
        description: Description for progress bar
    """
    try:
        with tqdm(
            unit="B", unit_scale=True, unit_divisor=1024, desc=description
        ) as pbar:

            def progress_hook(block_num, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(url, destination, progress_hook)

        logger.info(f"Downloaded {destination}")

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def download_yolo_models():
    """Download YOLO models for wildlife detection."""
    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model URLs and destinations
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
    }

    logger.info("Downloading YOLO models...")

    for model_name, url in models.items():
        model_path = models_dir / model_name

        if model_path.exists():
            logger.info(f"Model {model_name} already exists, skipping...")
            continue

        try:
            download_file(url, str(model_path), f"Downloading {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")


def download_wildlife_models():
    """Download pre-trained wildlife detection models."""
    config = get_config()
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Note: These are placeholder URLs. In a real implementation,
    # you would have actual wildlife detection models trained on
    # aerial imagery datasets.
    wildlife_models = {
        "yolo_wildlife.pt": "https://example.com/wildlife_model.pt",  # Placeholder
        "wildlife_detector_v1.pt": "https://example.com/wildlife_v1.pt",  # Placeholder
    }

    logger.info("Wildlife detection models not available for download.")
    logger.info("Please train your own models or use the default YOLO models.")

    # For now, just copy a default YOLO model
    default_model = models_dir / "yolov8n.pt"
    wildlife_model = models_dir / "yolo_wildlife.pt"

    if default_model.exists() and not wildlife_model.exists():
        import shutil

        shutil.copy2(default_model, wildlife_model)
        logger.info(f"Copied {default_model} to {wildlife_model}")
        logger.info("You can now use this as a starting point for wildlife detection.")


def main():
    """Main function to download models."""
    logger.info("Starting model download...")

    try:
        # Create necessary directories
        create_directories()

        # Download YOLO models
        download_yolo_models()

        # Download wildlife models (placeholder)
        download_wildlife_models()

        logger.info("Model download completed!")
        logger.info("You can now use the models for wildlife detection.")

    except Exception as e:
        logger.error(f"Error during model download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
