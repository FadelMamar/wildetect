#!/usr/bin/env python3
"""
WildDetect Initialization Script

Set up directories and download models for WildDetect.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.utils.config import create_directories

from scripts.download_models import download_yolo_models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize WildDetect."""
    logger.info("Initializing WildDetect...")

    try:
        # Create directories
        logger.info("Creating directories...")
        create_directories()
        logger.info("✓ Directories created")

        # Download models
        logger.info("Downloading models...")
        download_yolo_models()
        logger.info("✓ Models downloaded")

        logger.info("WildDetect initialization completed!")
        logger.info("\nYou can now use:")
        logger.info("  python scripts/detect.py --images data/images/*.jpg")
        logger.info("  python scripts/fiftyone.py add --images data/images/*.jpg")
        logger.info(
            "  python scripts/train.py prepare-data --dataset wildlife_detection --output data/training"
        )

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
