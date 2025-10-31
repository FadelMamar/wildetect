"""
Transformation pipeline for orchestrating multiple data transformations.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from .base_transformer import BaseTransformer


class TransformationPipeline:
    """
    Pipeline for orchestrating multiple data transformations.

    This class manages a sequence of transformers and applies them
    in order to images and their annotations.
    """

    def __init__(self, transformers: Optional[List[BaseTransformer]] = None):
        """
        Initialize the transformation pipeline.

        Args:
            transformers: List of transformers to apply in sequence
        """
        self.transformers = transformers or []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_transformer(self, transformer: BaseTransformer) -> None:
        """
        Add a transformer to the pipeline.

        Args:
            transformer: Transformer to add
        """
        self.transformers.append(transformer)
        self.logger.info(f"Added transformer: {transformer.__class__.__name__}")

    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """
        Get the transformation history.

        Returns:
            List of transformation history
        """

        # Record transformation info
        transformation_history = []
        for i, transformer in enumerate(self.transformers):
            transformer_info = transformer.get_transformation_info()
            transformer_info["pipeline_index"] = i
            transformation_history.append(transformer_info)
        return transformation_history

    def remove_transformer(self, index: int) -> None:
        """
        Remove a transformer from the pipeline.

        Args:
            index: Index of transformer to remove
        """
        if 0 <= index < len(self.transformers):
            removed = self.transformers.pop(index)
            self.logger.info(f"Removed transformer: {removed.__class__.__name__}")

    def clear_transformers(self) -> None:
        """Clear all transformers from the pipeline."""
        self.transformers.clear()
        self.logger.info("Cleared all transformers")

    def transform(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply all transformers in sequence.
        Returns:
            List of transformed data dictionaries
        """
        data = [inputs]
        outputs = None

        if len(self.transformers) == 0:
            return [inputs]

        for i, transformer in enumerate(self.transformers):
            try:
                self.logger.debug(
                    f"Applying transformer {i + 1}/{len(self.transformers)}: {transformer.__class__.__name__}"
                )

                outputs = transformer.transform(data)
                data = outputs  # saving for next transform

                self.logger.debug(
                    f"Transformer {transformer.__class__.__name__} completed successfully"
                )

            except Exception as e:
                self.logger.error(
                    f"Error in transformer {transformer.__class__.__name__}: {traceback.format_exc()}"
                )
                raise

        return outputs

    def transform_batch(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Transform a batch of images and annotations.
        """
        results = []

        for i, data in enumerate(inputs):
            try:
                result = self.transform(data)
                results.append(result)
                self.logger.debug(f"Transformed image {i + 1}/{len(inputs)}")
            except Exception:
                self.logger.error(
                    f"Error transforming image {i + 1}: {traceback.format_exc()}"
                )
                raise

        return results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the transformation pipeline.

        Returns:
            Dictionary with pipeline information
        """
        return {
            "num_transformers": len(self.transformers),
            "transformer_types": [t.__class__.__name__ for t in self.transformers],
            "transformer_configs": [vars(t.config) for t in self.transformers],
        }

    def save_pipeline_config(self, filepath: str) -> None:
        """
        Save pipeline configuration to file.

        Args:
            filepath: Path to save configuration
        """
        config = self.get_pipeline_info()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Saved pipeline configuration to: {filepath}")

    def __len__(self) -> int:
        """Return the number of transformers in the pipeline."""
        return len(self.transformers)

    def __getitem__(self, index: int) -> BaseTransformer:
        """Get transformer at index."""
        return self.transformers[index]

    def __iter__(self) -> Generator[BaseTransformer, None, None]:
        """Iterate over transformers."""
        for transformer in self.transformers:
            yield transformer
