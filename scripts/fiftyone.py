#!/usr/bin/env python3
"""
WildDetect FiftyOne Management Script

Manage FiftyOne datasets for wildlife detection with command-line interface.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.core.fiftyone_manager import FiftyOneManager
from app.utils.config import create_directories, get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main FiftyOne management script."""
    parser = argparse.ArgumentParser(description="WildDetect FiftyOne Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add images command
    add_parser = subparsers.add_parser("add", help="Add images to FiftyOne dataset")
    add_parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    add_parser.add_argument("--detections", help="Detection results file (JSON)")
    add_parser.add_argument(
        "--dataset", help="Dataset name (default: wildlife_detection)"
    )

    # Launch command
    subparsers.add_parser("launch", help="Launch FiftyOne app")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export dataset annotations")
    export_parser.add_argument("--output", required=True, help="Output directory")
    export_parser.add_argument(
        "--format",
        default="coco",
        choices=["coco", "yolo", "pascal"],
        help="Export format",
    )
    export_parser.add_argument(
        "--dataset", help="Dataset name (default: wildlife_detection)"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument(
        "--dataset", help="Dataset name (default: wildlife_detection)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create directories
    create_directories()

    # Initialize FiftyOne manager
    dataset_name = getattr(args, "dataset", None)
    fo_manager = FiftyOneManager(dataset_name)

    try:
        if args.command == "add":
            # Load detections if provided
            detections = None
            if args.detections:
                with open(args.detections, "r") as f:
                    detections = json.load(f)

            # Add images to dataset
            logger.info(f"Adding {len(args.images)} images to FiftyOne dataset...")
            if detections:
                fo_manager.add_images(args.images, detections)
            else:
                fo_manager.add_images(args.images)
            logger.info("✓ Images added to FiftyOne dataset")

            # Show dataset info
            info = fo_manager.get_dataset_info()
            print(f"Dataset: {info['name']}")
            print(f"Total samples: {info['num_samples']}")

        elif args.command == "launch":
            # Launch FiftyOne app
            logger.info("Launching FiftyOne app...")
            fo_manager.launch_app()

        elif args.command == "export":
            # Export dataset
            logger.info(f"Exporting dataset in {args.format} format...")
            fo_manager.export_annotations(args.output, args.format)
            logger.info(f"✓ Dataset exported to {args.output}")

        elif args.command == "stats":
            # Show statistics
            stats = fo_manager.get_annotation_stats()
            print("Dataset Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Annotated samples: {stats['annotated_samples']}")
            print(f"  Total detections: {stats['total_detections']}")

            if stats["species_counts"]:
                print(f"  Species distribution:")
                for species, count in sorted(
                    stats["species_counts"].items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"    {species}: {count}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
