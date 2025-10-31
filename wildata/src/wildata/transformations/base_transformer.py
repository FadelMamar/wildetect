"""
Base transformer class for data transformations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BaseTransformer(ABC):
    """
    Abstract base class for all data transformations.

    This class defines the interface that all transformers must implement.
    Transformers can be applied to both images and their corresponding annotations.
    """

    def __init__(
        self,
    ):
        """
        Initialize the transformer.

        Args:
            config: Configuration dictionary or dataclass for the transformer
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.config = None

    def _validate_inputs(self, inputs: List[Dict[str, Any]]) -> None:
        """
        Validate the inputs.
        """
        for data in inputs:
            assert isinstance(data, dict), "inputs must be a dictionary"
            assert data.get("image") is not None, "inputs must contain an image"
            assert (
                data.get("annotations") is not None
            ), "inputs must contain annotations"
            assert (
                data["image"].shape[2] == 3
            ), "image must be a 3 channel image arranged like HWC"

            for annotation in data["annotations"]:
                assert "bbox" in annotation, "annotations must contain a bbox"
                assert len(annotation["bbox"]) == 4, "bbox must be a list of 4 elements"

    @abstractmethod
    def transform(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform both image and annotations together.

        Args:
            image: Input image as numpy array
            annotations: List of annotation dictionaries
            image_info: Metadata about the image

        Returns:
            Tuple of (transformed_image, transformed_annotations, updated_image_info)
        """
        pass

    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about the transformation that was applied.

        Returns:
            Dictionary with transformation metadata
        """
        return {"transformer_type": str(self.__class__.__name__)}
