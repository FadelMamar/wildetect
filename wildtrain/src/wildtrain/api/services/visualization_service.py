"""Visualization service for integrating CLI functionality with the API."""

import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from omegaconf import OmegaConf
from ...cli.config_loader import ConfigLoader
from ...cli.models import ClassificationVisualizationConfig, DetectionVisualizationConfig
from ...shared.config_types import ConfigType

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for handling visualization operations."""

    @staticmethod
    def visualize_classifier_predictions(config: ClassificationVisualizationConfig) -> None:
        """Visualize classifier predictions using FiftyOne."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.debug(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.visualize import classifier_predictions
            classifier_predictions(config=Path(temp_config_path), template=False)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Classifier visualization completed successfully")
            return 

        except Exception as e:
            raise Exception(f"Classifier visualization failed: {e}")

    @staticmethod
    def visualize_detector_predictions(config: DetectionVisualizationConfig) -> None:
        """Visualize detector predictions using FiftyOne."""
        try:
            # Create a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_dict = config.model_dump()
                yaml_content = OmegaConf.to_yaml(OmegaConf.create(config_dict))
                f.write(yaml_content)
                temp_config_path = f.name

            logger.info(f"Created temporary config file: {temp_config_path}")

            # Import and run the CLI command
            from ...cli.commands.visualize import detector_predictions
            detector_predictions(config=Path(temp_config_path), template=False)

            # Clean up temporary file
            Path(temp_config_path).unlink(missing_ok=True)

            logger.info("Detector visualization completed successfully")
            return 

        except Exception as e:
            raise Exception(f"Detector visualization failed: {e}")

    @staticmethod
    def generate_classification_visualization_template() -> str:
        """Generate a classification visualization template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.CLASSIFICATION_VISUALIZATION)
            return template
        except Exception as e:
            raise Exception(f"Failed to generate classification visualization template: {e}")

    @staticmethod
    def generate_detection_visualization_template() -> str:
        """Generate a detection visualization template."""
        try:
            template = ConfigLoader.generate_default_config(ConfigType.DETECTION_VISUALIZATION)
            return template
        except Exception as e:
            raise Exception(f"Failed to generate detection visualization template: {e}")

    @staticmethod
    def get_fiftyone_datasets() -> Dict[str, Any]:
        """Get list of available FiftyOne datasets."""
        try:
            import fiftyone as fo
            datasets = fo.list_datasets()
            return {
                "datasets": datasets,
                "total_count": len(datasets)
            }
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
                "info": dict(dataset.info)
            }
        except Exception as e:
            raise Exception(f"Failed to get dataset info for {dataset_name}: {e}")
