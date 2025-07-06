#!/usr/bin/env python3
"""
WildDetect Training Script

Train and fine-tune wildlife detection models with command-line interface.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.core.trainer import ModelTrainer
from app.utils.config import create_directories, get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="WildDetect Training")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare data command
    prepare_parser = subparsers.add_parser(
        "prepare-data", help="Prepare training data from FiftyOne"
    )
    prepare_parser.add_argument(
        "--dataset", required=True, help="FiftyOne dataset name"
    )
    prepare_parser.add_argument("--output", required=True, help="Output directory")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train detection model")
    train_parser.add_argument(
        "--data", required=True, help="Dataset configuration path"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    train_parser.add_argument("--output", default="models", help="Output directory")
    train_parser.add_argument("--model", help="Base model path (optional)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model", required=True, help="Model path")
    eval_parser.add_argument("--data", required=True, help="Dataset configuration path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create directories
    create_directories()

    # Initialize trainer
    trainer = ModelTrainer(model_path=getattr(args, "model", None))

    try:
        if args.command == "prepare-data":
            # Prepare training data
            logger.info(f"Preparing training data from dataset: {args.dataset}")
            training_info = trainer.prepare_training_data(args.dataset, args.output)

            print("Training Data Prepared:")
            print(f"  Total samples: {training_info['total_samples']}")
            print(f"  Train samples: {training_info['train_samples']}")
            print(f"  Val samples: {training_info['val_samples']}")
            print(f"  Total annotations: {training_info['total_annotations']}")

            if training_info["species_counts"]:
                print(f"  Species distribution:")
                for species, count in sorted(
                    training_info["species_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"    {species}: {count}")

            print(f"\nDataset configuration saved to: {args.output}/dataset.yaml")

        elif args.command == "train":
            # Train model
            logger.info("Starting model training...")
            results = trainer.train_model(
                data_path=args.data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                save_dir=args.output,
            )

            print("Training Completed:")
            print(f"  Best model: {results['best_model_path']}")
            print(f"  mAP50: {results['best_map']:.3f}")
            print(f"  Epochs trained: {results['epochs_trained']}")
            print(f"  Save directory: {results['save_dir']}")

        elif args.command == "evaluate":
            # Evaluate model
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate_model(args.model, args.data)

            print("Evaluation Results:")
            print(f"  mAP50: {eval_results['mAP50']:.3f}")
            print(f"  mAP50-95: {eval_results['mAP50-95']:.3f}")
            print(f"  Precision: {eval_results['precision']:.3f}")
            print(f"  Recall: {eval_results['recall']:.3f}")
            print(f"  F1 Score: {eval_results['f1']:.3f}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
