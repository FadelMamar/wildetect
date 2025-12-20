"""
Tests for the transformation system.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from wildata.config import AugmentationConfig, TilingConfig
from wildata.transformations import (
    AugmentationTransformer,
    TilingTransformer,
    TransformationPipeline,
)
from wildata.transformations.base_transformer import BaseTransformer


def generate_synthetic_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate synthetic test data for transformations.

    Returns:
        Dictionary containing different test scenarios:
        - 'normal_images': Images with bounding box annotations
        - 'empty_images': Images without annotations
        - 'small_images': Images smaller than tile size
        - 'large_images': Images larger than tile size
        - 'mixed_annotations': Images with multiple bounding boxes
    """

    def create_image(height: int, width: int, channels: int = 3) -> np.ndarray:
        """Create a synthetic image with some patterns."""
        image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

        # Add some patterns to make it more realistic
        # Add a gradient
        for i in range(height):
            for j in range(width):
                image[i, j] = [
                    int(255 * i / height),
                    int(255 * j / width),
                    int(255 * (i + j) / (height + width)),
                ]

        return image

    def create_bbox_annotation(
        category_id: int, x: float, y: float, w: float, h: float
    ) -> Dict[str, Any]:
        """Create a bounding box annotation."""
        return {
            "id": category_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        }

    # Scenario 1: Normal images with bounding box annotations
    normal_images = []
    for i in range(3):
        image = create_image(512, 512)
        image_info = {
            "id": i,
            "file_name": f"normal_image_{i}.jpg",
            "width": 512,
            "height": 512,
        }

        # Add multiple bounding box annotations
        annotations = []

        # Bbox annotations
        for j in range(3):
            x = np.random.randint(50, 400)
            y = np.random.randint(50, 400)
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)
            annotations.append(create_bbox_annotation(j, x, y, w, h))

        normal_images.append(
            {"image": image, "annotations": annotations, "info": image_info}
        )

    # Scenario 2: Empty images (no annotations)
    empty_images = []
    for i in range(2):
        image = create_image(512, 512)
        image_info = {
            "id": i + 100,
            "file_name": f"empty_image_{i}.jpg",
            "width": 512,
            "height": 512,
        }

        empty_images.append({"image": image, "annotations": [], "info": image_info})

    # Scenario 3: Small images (smaller than tile size)
    small_images = []
    for i in range(2):
        image = create_image(256, 256)  # Smaller than typical tile size
        image_info = {
            "id": i + 200,
            "file_name": f"small_image_{i}.jpg",
            "width": 256,
            "height": 256,
        }

        # Add one small annotation
        annotations = []
        if i == 0:  # Add annotation to first small image
            annotations.append(create_bbox_annotation(0, 50, 50, 30, 30))

        small_images.append(
            {"image": image, "annotations": annotations, "info": image_info}
        )

    # Scenario 4: Large images (larger than tile size)
    large_images = []
    for i in range(2):
        image = create_image(1024, 1024)  # Larger than typical tile size
        image_info = {
            "id": i + 300,
            "file_name": f"large_image_{i}.jpg",
            "width": 1024,
            "height": 1024,
        }

        # Add multiple annotations spread across the image
        annotations = []
        for j in range(5):
            x = np.random.randint(100, 900)
            y = np.random.randint(100, 900)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            annotations.append(create_bbox_annotation(j, x, y, w, h))

        large_images.append(
            {"image": image, "annotations": annotations, "info": image_info}
        )

    # Scenario 5: Multiple bounding boxes
    mixed_images = []
    for i in range(2):
        image = create_image(512, 512)
        image_info = {
            "id": i + 400,
            "file_name": f"mixed_image_{i}.jpg",
            "width": 512,
            "height": 512,
        }

        annotations = []

        # Multiple bounding boxes
        annotations.append(create_bbox_annotation(0, 100, 100, 80, 80))
        annotations.append(create_bbox_annotation(1, 200, 200, 60, 60))
        annotations.append(create_bbox_annotation(2, 300, 300, 40, 40))

        mixed_images.append(
            {"image": image, "annotations": annotations, "info": image_info}
        )

    return {
        "normal_images": normal_images,
        "empty_images": empty_images,
        "small_images": small_images,
        "large_images": large_images,
        "mixed_images": mixed_images,
    }


class TestAugmentationTransformer:
    """Test the augmentation transformer."""

    def test_augmentation_transformer_initialization(self):
        """Test augmentation transformer initialization with dataclass config."""
        config = AugmentationConfig(rotation_range=(-10, 10), probability=0.3)
        transformer = AugmentationTransformer(config)
        assert transformer.config.rotation_range == (-10, 10)
        assert transformer.config.probability == 0.3

    def test_augmentation_transformer_default_config(self):
        """Test augmentation transformer with default config."""
        transformer = AugmentationTransformer()
        assert hasattr(transformer.config, "rotation_range")
        assert hasattr(transformer.config, "probability")
        assert hasattr(transformer.config, "brightness_range")

    def test_augmentation_transformer_basic_transformation(self):
        """Test basic image transformation with augmentation."""
        # Create minimal config for deterministic test
        config = AugmentationConfig(
            rotation_range=(0, 0),  # No rotation
            probability=0.0,  # No flip
            brightness_range=(1.0, 1.0),  # No brightness change
            contrast_range=(1.0, 1.0),  # No contrast change
            noise_std=(0.0, 0.0),  # No noise
        )
        transformer = AugmentationTransformer(config)

        # Get synthetic data
        test_data = generate_synthetic_data()
        normal_image_data = test_data["normal_images"][0]

        # Transform data
        outputs = transformer.transform([normal_image_data])

        # Check results
        assert len(outputs) == 1
        output = outputs[0]
        assert "image" in output
        assert "annotations" in output
        assert "info" in output

        # Check that image shape is preserved
        assert output["image"].shape == normal_image_data["image"].shape
        assert len(output["annotations"]) == len(normal_image_data["annotations"])
        # Check that all annotations are bounding boxes
        for annotation in output["annotations"]:
            assert "bbox" in annotation

    def test_augmentation_transformer_empty_image(self):
        """Test augmentation with empty image (no annotations)."""
        config = AugmentationConfig(
            rotation_range=(0, 0),
            probability=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            noise_std=(0.0, 0.0),
        )
        transformer = AugmentationTransformer(config)

        test_data = generate_synthetic_data()
        empty_image_data = test_data["empty_images"][0]

        outputs = transformer.transform([empty_image_data])

        assert len(outputs) == 1
        output = outputs[0]
        assert output["image"].shape == empty_image_data["image"].shape
        assert len(output["annotations"]) == 0  # Should remain empty

    def test_augmentation_transformer_with_actual_augmentation(self):
        """Test augmentation with actual transformations enabled."""
        config = AugmentationConfig(
            rotation_range=(-15, 15),
            probability=0.5,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            noise_std=(0.01, 0.1),
        )
        transformer = AugmentationTransformer(config)

        test_data = generate_synthetic_data()
        normal_image_data = test_data["normal_images"][0]

        outputs = transformer.transform([normal_image_data])

        assert len(outputs) == 1
        output = outputs[0]

        # Image should still have same shape
        assert output["image"].shape == normal_image_data["image"].shape
        # Annotations should be preserved
        assert len(output["annotations"]) == len(normal_image_data["annotations"])
        # Check that all annotations are bounding boxes
        for annotation in output["annotations"]:
            assert "bbox" in annotation


class TestTilingTransformer:
    """Test the tiling transformer."""

    def test_tiling_transformer_initialization(self):
        """Test tiling transformer initialization with dataclass config."""
        config = TilingConfig(
            tile_size=256,
            stride=128,
            min_visibility=0.1,
            max_negative_tiles_in_negative_image=3,
            negative_positive_ratio=1.0,
        )
        transformer = TilingTransformer(config)
        assert transformer.config.tile_size == 256
        assert transformer.config.stride == 128
        assert transformer.config.min_visibility == 0.1

    def test_tiling_transformer_default_config(self):
        """Test tiling transformer with default config."""
        transformer = TilingTransformer()
        assert hasattr(transformer.config, "tile_size")
        assert hasattr(transformer.config, "stride")
        assert hasattr(transformer.config, "min_visibility")

    def test_tiling_transformer_normal_image(self):
        """Test tiling with normal sized image."""
        config = TilingConfig(
            tile_size=256,
            stride=128,
            min_visibility=0.0,  # Accept all tiles
            max_negative_tiles_in_negative_image=5,
            negative_positive_ratio=1.0,
        )
        transformer = TilingTransformer(config)

        test_data = generate_synthetic_data()
        normal_image_data = test_data["normal_images"][0]

        outputs = transformer.transform([normal_image_data])

        # Should produce multiple tiles
        assert len(outputs) > 1

        # Check each tile
        for output in outputs:
            assert "image" in output
            assert "annotations" in output
            assert "info" in output

            # Check tile size
            assert output["image"].shape[0] == 256  # height
            assert output["image"].shape[1] == 256  # width
            assert output["image"].shape[2] == 3  # channels

            # Check tile info
            assert "tile_coords" in output["info"]
            assert "tile_size" in output["info"]

    def test_tiling_transformer_empty_image(self):
        """Test tiling with empty image."""
        config = TilingConfig(
            tile_size=256,
            stride=128,
            min_visibility=0.0,
            max_negative_tiles_in_negative_image=3,
            negative_positive_ratio=1.0,
        )
        transformer = TilingTransformer(config)

        test_data = generate_synthetic_data()
        empty_image_data = test_data["empty_images"][0]

        outputs = transformer.transform([empty_image_data])

        # Should produce tiles even for empty images
        assert len(outputs) > 0

        # All tiles should be empty
        for output in outputs:
            assert len(output["annotations"]) == 0

    def test_tiling_transformer_small_image(self):
        """Test tiling with image smaller than tile size."""
        config = TilingConfig(
            tile_size=512,  # Larger than image
            stride=256,
            min_visibility=0.0,
            max_negative_tiles_in_negative_image=1,
            negative_positive_ratio=1.0,
        )
        transformer = TilingTransformer(config)

        test_data = generate_synthetic_data()
        small_image_data = test_data["small_images"][0]

        outputs = transformer.transform([small_image_data])

        # Should handle small images gracefully
        assert len(outputs) >= 1

        # Check that small image is handled properly
        for output in outputs:
            assert "image" in output
            assert "annotations" in output
            assert "info" in output

    def test_tiling_transformer_large_image(self):
        """Test tiling with large image."""
        config = TilingConfig(
            tile_size=256,
            stride=128,
            min_visibility=0.0,
            max_negative_tiles_in_negative_image=10,
            negative_positive_ratio=1.0,
        )
        transformer = TilingTransformer(config)

        test_data = generate_synthetic_data()
        large_image_data = test_data["large_images"][0]

        outputs = transformer.transform([large_image_data])

        # Should produce many tiles for large image
        assert len(outputs) > 5

        # Check tile consistency
        for output in outputs:
            assert output["image"].shape == (256, 256, 3)
            assert "tile_coords" in output["info"]


class TestTransformationPipeline:
    """Test the transformation pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = TransformationPipeline()
        assert len(pipeline) == 0

    def test_pipeline_add_transformer(self):
        """Test adding transformers to pipeline."""
        pipeline = TransformationPipeline()
        transformer = AugmentationTransformer()

        pipeline.add_transformer(transformer)
        assert len(pipeline) == 1
        assert pipeline[0] == transformer

    def test_pipeline_remove_transformer(self):
        """Test removing transformers from pipeline."""
        pipeline = TransformationPipeline()
        transformer = AugmentationTransformer()

        pipeline.add_transformer(transformer)
        assert len(pipeline) == 1

        pipeline.remove_transformer(0)
        assert len(pipeline) == 0

    def test_pipeline_clear_transformers(self):
        """Test clearing all transformers."""
        pipeline = TransformationPipeline()

        pipeline.add_transformer(AugmentationTransformer())
        pipeline.add_transformer(TilingTransformer())
        assert len(pipeline) == 2

        pipeline.clear_transformers()
        assert len(pipeline) == 0

    def test_pipeline_single_transformer(self):
        """Test pipeline with single transformer."""
        pipeline = TransformationPipeline()

        # Add augmentation transformer
        aug_config = AugmentationConfig(
            rotation_range=(0, 0),
            probability=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            noise_std=(0.0, 0.0),
        )
        transformer = AugmentationTransformer(aug_config)
        pipeline.add_transformer(transformer)

        # Test data
        test_data = generate_synthetic_data()
        normal_image_data = test_data["normal_images"][0]

        # Apply transformation
        outputs = pipeline.transform(normal_image_data)

        # Check results
        assert len(outputs) == 1
        output = outputs[0]
        assert output["image"].shape == normal_image_data["image"].shape
        assert len(output["annotations"]) == len(normal_image_data["annotations"])

    def test_pipeline_multiple_transformers(self):
        """Test pipeline with multiple transformers."""
        pipeline = TransformationPipeline()

        # Add augmentation transformer
        aug_config = AugmentationConfig(
            rotation_range=(0, 0),
            probability=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            noise_std=(0.0, 0.0),
        )
        pipeline.add_transformer(AugmentationTransformer(aug_config))

        # Add tiling transformer
        tiling_config = TilingConfig(
            tile_size=256,
            stride=128,
            min_visibility=0.0,
            max_negative_tiles_in_negative_image=3,
            negative_positive_ratio=1.0,
        )
        pipeline.add_transformer(TilingTransformer(tiling_config))

        # Test data
        test_data = generate_synthetic_data()
        normal_image_data = test_data["normal_images"][0]

        # Apply transformation
        outputs = pipeline.transform(normal_image_data)

        # Should produce multiple tiles after augmentation
        assert len(outputs) > 1

        # Check each output
        for output in outputs:
            assert "image" in output
            assert "annotations" in output
            assert "info" in output

    def test_pipeline_get_info(self):
        """Test getting pipeline information."""
        pipeline = TransformationPipeline()

        # Add transformers
        pipeline.add_transformer(AugmentationTransformer())
        pipeline.add_transformer(TilingTransformer())

        info = pipeline.get_pipeline_info()

        assert info["num_transformers"] == 2
        assert "AugmentationTransformer" in info["transformer_types"]
        assert "TilingTransformer" in info["transformer_types"]
        assert len(info["transformer_configs"]) == 2

    def test_pipeline_batch_processing(self):
        """Test batch processing with pipeline."""
        pipeline = TransformationPipeline()

        # Add simple transformer
        aug_config = AugmentationConfig(
            rotation_range=(0, 0),
            probability=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            noise_std=(0.0, 0.0),
        )
        pipeline.add_transformer(AugmentationTransformer(aug_config))

        # Test data
        test_data = generate_synthetic_data()
        batch_data = [
            test_data["normal_images"][0],
            test_data["empty_images"][0],
            test_data["small_images"][0],
        ]

        # Process batch
        outputs = pipeline.transform_batch(batch_data)

        # Should process all inputs
        assert len(outputs) == 3

        # Check each output
        for output in outputs:
            for data in output:
                assert "image" in data
                assert "annotations" in data
                assert "info" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
