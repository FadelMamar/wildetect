"""
Data augmentation transformer using Albumentations library.
"""

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

from ..config import AugmentationConfig
from .base_transformer import BaseTransformer


class AugmentationTransformer(BaseTransformer):
    """
    Transformer for data augmentation operations using Albumentations.

    Supports common augmentation techniques like:
    - Random rotation and affine transformations
    - Random flip (horizontal/vertical)
    - Random brightness/contrast adjustment
    - Random noise and blur
    - Random crop and resize
    - Color jittering
    - Dropout and regularization techniques
    """

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
    ):
        """
        Initialize the augmentation transformer.

        Args:
            config: AugmentationConfig dataclass with augmentation parameters
        """
        super().__init__()
        self.config = config or AugmentationConfig()
        self._create_albumentations_pipeline()

    def _create_albumentations_pipeline(self):
        """Create Albumentations pipeline based on configuration."""
        transforms = []

        # Basic geometric transforms
        transforms.append(
            A.Affine(
                scale=self.config.scale,
                translate_percent=self.config.translate,
                # Tuple for rotation range
                rotate=self.config.rotation_range,
                shear=self.config.shear,
                # Interpolation methods
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                # Other parameters
                fit_output=False,
                keep_ratio=True,
                rotate_method="largest_box",
                balanced_scale=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=self.config.probability,
            )
        )

        # Horizontal flip
        transforms.append(A.HorizontalFlip(p=self.config.probability))

        # Brightness and contrast adjustments
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_range,
                contrast_limit=self.config.contrast_range,
                p=self.config.probability,
            )
        )

        # Noise addition
        transforms.append(
            A.GaussNoise(p=self.config.probability, std_range=self.config.noise_std)
        )

        # Create the pipeline - only include bbox_params if we have bboxes
        self.pipeline = A.Compose(transforms, seed=self.config.seed)
        self.pipeline_with_bboxes = A.Compose(
            transforms,
            seed=self.config.seed,
            bbox_params=A.BboxParams(
                format="coco", label_fields=["class_labels"], min_visibility=0.0
            ),
        )

    def transform(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._validate_inputs(inputs)
        outputs = []
        for data in inputs:
            for i in range(self.config.num_transforms):
                outputs.append(self._transform_once(data, i))

        # add untransformed
        outputs.extend(inputs.copy())
        return outputs

    def _transform_once(self, inputs: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Apply augmentation transformations to both image and annotations.

        Args:
            image: Input image as numpy array (HWC format)
            annotations: List of annotation dictionaries
            image_info: Metadata about the image

        Returns:
            Tuple of (augmented_image, transformed_annotations, updated_image_info)
        """
        image = inputs["image"]
        annotations = inputs["annotations"]
        image_info = inputs["info"]

        # Prepare data for Albumentations
        albumentations_data = self._prepare_albumentations_data(image, annotations)

        # Apply augmentation
        try:
            # Use appropriate pipeline based on whether we have bboxes
            if (
                "bboxes" in albumentations_data
                and len(albumentations_data["bboxes"]) > 0
            ):
                augmented_data = self.pipeline_with_bboxes(**albumentations_data)
            else:
                augmented_data = self.pipeline(**albumentations_data)
            (
                augmented_image,
                transformed_annotations,
            ) = self._process_augmented_annotations(augmented_data, annotations)
            # Update image info
            updated_image_info = image_info.copy()
            updated_image_info[
                "file_name"
            ] = f"{Path(updated_image_info['file_name']).stem}_augmented_{index}.jpg"
            updated_image_info["augmentation_applied"] = True

            output = {
                "image": augmented_image,
                "annotations": transformed_annotations,
                "info": updated_image_info,
            }
            return output
        except Exception:
            self.logger.warning(
                f"Augmentation failed, returning original data: {traceback.format_exc()}"
            )
            raise ValueError(f"Augmentation failed: {traceback.format_exc()}")

    def _prepare_albumentations_data(
        self, image: np.ndarray, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data in the format expected by Albumentations."""
        data = {"image": image}

        # Extract bounding boxes
        bboxes = []
        class_labels = []
        for annotation in annotations:
            if "bbox" in annotation:
                bbox = annotation["bbox"]
                bboxes.append(bbox)
                class_labels.append(annotation.get("category_id", 0))

        if bboxes:
            data["bboxes"] = np.array(bboxes)
            data["class_labels"] = np.array(class_labels)

        return data

    def _process_augmented_annotations(
        self, augmented_data: Dict[str, Any], original_annotations: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process augmented data back to annotation format."""
        transformed_annotations = []

        # Process bounding boxes
        augmented_image = augmented_data["image"]
        bboxes = augmented_data.get("bboxes", [])
        class_labels = augmented_data.get("class_labels", [])
        transformed_annotations = []

        if len(bboxes) != len(class_labels):
            raise ValueError("Number of bboxes and class labels must be the same")

        if len(bboxes) == 0:
            return augmented_image, original_annotations

        for i, bbox in enumerate(bboxes):
            transformed_annotation = original_annotations[i].copy()
            transformed_annotation["bbox"] = list(bbox)
            transformed_annotations.append(transformed_annotation)

        return augmented_image, transformed_annotations

    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about the transformation pipeline."""
        return {
            "transformer_type": self.__class__.__name__,
            "config": self.config,
            "pipeline_info": {
                "num_transforms": len(self.pipeline.transforms),
                "transform_types": [type(t).__name__ for t in self.pipeline.transforms],
            },
        }
