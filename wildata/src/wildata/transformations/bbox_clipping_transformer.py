"""
Bounding box clipping transformer.

This transformer clips bounding boxes to ensure they stay within image boundaries
given a specified tolerance.
"""

from typing import Any, Dict, List

import numpy as np

from ..converters.utils import clip_bbox
from .base_transformer import BaseTransformer


class BoundingBoxClippingTransformer(BaseTransformer):
    """
    Transformer that clips bounding boxes to image boundaries.

    This transformer ensures that all bounding boxes in annotations stay within
    the image boundaries, with a specified tolerance for edge cases.
    """

    def __init__(self, tolerance: int = 5, skip_invalid: bool = False):
        """
        Initialize the bounding box clipping transformer.

        Args:
            tolerance: Number of pixels to allow outside boundaries before clipping
            skip_invalid: If True, skip invalid annotations (outside tolerance)
                         If False, raise ValueError for invalid annotations
        """
        super().__init__()
        self.tolerance = tolerance
        self.skip_invalid = skip_invalid
        self.clipped_count = 0
        self.skipped_count = 0
        self.valid_count = 0
        self.config = {"tolerance": tolerance, "skip_invalid": skip_invalid}

    def transform(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform annotations by clipping bounding boxes to image boundaries.

        Args:
            inputs: List of dictionaries containing:
                - 'image': numpy array of the image
                - 'annotations': list of annotation dictionaries
                - 'image_info': metadata about the image (optional)

        Returns:
            List of transformed dictionaries with clipped bounding boxes
        """
        self._validate_inputs(inputs)
        self.reset_stats()

        transformed_inputs = []

        for input_data in inputs:
            image = input_data["image"]
            annotations = input_data.get("annotations", [])
            height, width = image.shape[:2]

            # Transform annotations
            transformed_annotations = []

            for annotation in annotations:
                if "bbox" not in annotation:
                    transformed_annotations.append(annotation)
                    continue

                bbox = annotation["bbox"]
                clipped_bbox, was_clipped, is_valid = clip_bbox(
                    bbox, width, height, self.tolerance, verbose=False
                )
                if not is_valid:
                    if self.skip_invalid:
                        # Skip invalid annotations
                        self.skipped_count += 1
                        self.logger.debug(
                            f"Skipped invalid bbox {bbox} for image {width}x{height}"
                        )
                        continue
                    else:
                        # Raise error for invalid annotations
                        raise ValueError(
                            f"Invalid bbox {bbox} for image {width}x{height}, "
                            f"with tolerance {self.tolerance}"
                        )

                if was_clipped:
                    self.clipped_count += 1
                    self.logger.debug(f"Clipped bbox from {bbox} to {clipped_bbox}")
                else:
                    self.valid_count += 1

                # Create transformed annotation
                transformed_annotation = annotation.copy()
                transformed_annotation["bbox"] = clipped_bbox

                # Recalculate area if it exists
                if "area" in transformed_annotation:
                    w, h = clipped_bbox[2], clipped_bbox[3]
                    transformed_annotation["area"] = w * h

                transformed_annotations.append(transformed_annotation)

            # Create transformed input
            transformed_input = input_data.copy()
            transformed_input["annotations"] = transformed_annotations

            transformed_inputs.append(transformed_input)

        return transformed_inputs

    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about the transformation that was applied.

        Returns:
            Dictionary with transformation metadata
        """
        return {
            "transformer_type": self.__class__.__name__,
            "tolerance": self.tolerance,
            "skip_invalid": self.skip_invalid,
            "clipped_count": self.clipped_count,
            "skipped_count": self.skipped_count,
            "valid_count": self.valid_count,
            "total_processed": self.clipped_count
            + self.skipped_count
            + self.valid_count,
        }

    def reset_stats(self):
        """
        Reset the transformation statistics.
        """
        self.clipped_count = 0
        self.skipped_count = 0
        self.valid_count = 0
