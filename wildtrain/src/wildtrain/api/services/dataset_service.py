"""Dataset service for integrating CLI functionality with the API."""

import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ...cli.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for handling dataset operations."""

    @staticmethod
    def get_dataset_stats(data_dir: Path, split: str = "train", output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Get dataset statistics using the CLI."""
        try:
            logger.info(f"Getting dataset statistics for {data_dir}, split: {split}")

            # Import and run the CLI command
            from ...cli.commands.dataset import stats
            results = stats(data_dir=data_dir, split=split, output_file=output_file)

            # Parse results (this would need to be enhanced based on actual CLI output)
            formatted_results = {
                "stats": {
                    "mean": results["mean"],
                    "std": results["std"]
                },
                "split_info": {
                    "split": split,
                    "data_dir": str(data_dir)
                }
            }

            logger.info("Dataset statistics computed successfully")
            return formatted_results

        except Exception as e:
            logger.error(f"Dataset statistics computation failed: {e}")
            raise

    @staticmethod
    def get_dataset_splits(data_dir: Path) -> Dict[str, Any]:
        """Get available dataset splits."""
        try:
            from ...data.classification_datamodule import ClassificationDataModule
            
            # Create the data module to check available splits
            datamodule = ClassificationDataModule(
                root_data_directory=str(data_dir), 
                batch_size=32, 
                transforms=None, 
                load_as_single_class=True
            )
            
            splits = {}
            
            # Check which splits exist
            try:
                datamodule.setup(stage="fit")
                splits["train"] = {
                    "exists": True,
                    "count": len(datamodule.train_dataset) if hasattr(datamodule, 'train_dataset') else 0
                }
            except Exception:
                splits["train"] = {"exists": False, "count": 0}
            
            try:
                datamodule.setup(stage="validate")
                splits["val"] = {
                    "exists": True,
                    "count": len(datamodule.val_dataset) if hasattr(datamodule, 'val_dataset') else 0
                }
            except Exception:
                splits["val"] = {"exists": False, "count": 0}
            
            try:
                datamodule.setup(stage="test")
                splits["test"] = {
                    "exists": True,
                    "count": len(datamodule.test_dataset) if hasattr(datamodule, 'test_dataset') else 0
                }
            except Exception:
                splits["test"] = {"exists": False, "count": 0}
            
            return {
                "splits": splits,
                "total_splits": len([s for s in splits.values() if s["exists"]])
            }

        except Exception as e:
            logger.error(f"Failed to get dataset splits for {data_dir}: {e}")
            raise
