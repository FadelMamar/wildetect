"""Visualization service for FiftyOne dataset lookups."""

import logging
from typing import Any, Dict

from ...cli.config_loader import ConfigLoader
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class VisualizationService:
    """FiftyOne dataset lookups. Prediction upload goes through the harness."""

    @staticmethod
    def generate_classification_visualization_template() -> str:
        """Generate a classification visualization template."""
        try:
            template = ConfigLoader.generate_default_config(
                ConfigType.CLASSIFICATION_VISUALIZATION
            )
            return template
        except Exception as e:
            raise Exception(
                f"Failed to generate classification visualization template: {e}"
            )

    @staticmethod
    def generate_detection_visualization_template() -> str:
        """Generate a detection visualization template."""
        try:
            template = ConfigLoader.generate_default_config(
                ConfigType.DETECTION_VISUALIZATION
            )
            return template
        except Exception as e:
            raise Exception(f"Failed to generate detection visualization template: {e}")

    @staticmethod
    def get_fiftyone_datasets() -> Dict[str, Any]:
        """Get list of available FiftyOne datasets."""
        try:
            import fiftyone as fo

            datasets = fo.list_datasets()
            return {"datasets": datasets, "total_count": len(datasets)}
        except Exception as e:
            raise Exception(f"Failed to get FiftyOne datasets: {e}")

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific FiftyOne dataset."""
        try:
            import fiftyone as fo

            if dataset_name not in fo.list_datasets():
                raise ValueError(f"Dataset {dataset_name} not found")

            dataset = fo.load_dataset(dataset_name)
            return {
                "name": dataset_name,
                "sample_count": len(dataset),
                "tags": list(dataset.tags),
                "fields": list(dataset.get_field_schema().keys()),
                "info": dict(dataset.info),
            }
        except Exception as e:
            raise Exception(f"Failed to get dataset info for {dataset_name}: {e}")
