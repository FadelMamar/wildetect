"""
Test script for TileUtilsv2 functionality.
"""

import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

from wildetect.core.config import ROOT

# Add the project root to the Python path
project_root = ROOT

import numpy as np
import torch
from PIL import Image
from wildetect.core.data.utils import TileUtils, TileUtilsv2

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic functionality of TileUtilsv2."""
    logger.info("Testing basic functionality...")

    # Create test image
    image_tensor = torch.randn(3, 512, 512)
    image_tensor = (image_tensor - image_tensor.min()) / (
        image_tensor.max() - image_tensor.min()
    )

    patch_size = 256
    stride = 128

    # Test TileUtilsv2
    patches_v2, offset_info_v2 = TileUtilsv2.get_patches_and_offset_info(
        image=image_tensor,
        patch_size=patch_size,
        stride=stride,
        channels=3,
        validate=False,  # Disable validation for this test
    )

    # Test original TileUtils
    patches_original, offset_info_original = TileUtils.get_patches_and_offset_info(
        image=image_tensor,
        patch_size=patch_size,
        stride=stride,
        channels=3,
        validate=False,
    )

    # Compare results
    assert (
        patches_v2.shape[0] == patches_original.shape[0]
    ), "Number of patches should match"
    assert (
        patches_v2.shape[1:] == patches_original.shape[1:]
    ), "Patch shapes should match"
    assert len(offset_info_v2["x_offset"]) == len(
        offset_info_original["x_offset"]
    ), "Offset counts should match"

    logger.info(
        f"✓ Basic functionality test passed - {patches_v2.shape[0]} patches extracted"
    )


def test_edge_cases():
    """Test edge cases for TileUtilsv2."""
    logger.info("Testing edge cases...")

    # Test with small image
    small_image = torch.randn(3, 100, 100)
    small_image = (small_image - small_image.min()) / (
        small_image.max() - small_image.min()
    )

    patches, offset_info = TileUtilsv2.get_patches_and_offset_info(
        image=small_image,
        patch_size=256,  # Larger than image
        stride=128,
        channels=3,
    )

    assert patches.shape[0] == 1, "Should return single patch for small image"
    assert patches.shape[1:] == small_image.shape, "Should return original image"

    # Test with exact patch size
    exact_image = torch.randn(3, 256, 256)
    exact_image = (exact_image - exact_image.min()) / (
        exact_image.max() - exact_image.min()
    )

    patches, offset_info = TileUtilsv2.get_patches_and_offset_info(
        image=exact_image,
        patch_size=256,
        stride=256,
        channels=3,
    )

    assert patches.shape[0] == 1, "Should return single patch for exact size"
    assert patches.shape[1:] == (3, 256, 256), "Should return correct patch size"

    logger.info("✓ Edge cases test passed")


def test_sahi_specific_features():
    """Test SAHI-specific features."""
    logger.info("Testing SAHI-specific features...")

    # Create test image
    image_tensor = torch.randn(3, 512, 512)
    image_tensor = (image_tensor - image_tensor.min()) / (
        image_tensor.max() - image_tensor.min()
    )

    patch_size = 256
    stride = 128

    # Test get_sliced_images
    sliced_images = TileUtilsv2.get_sliced_images(
        image=image_tensor,
        patch_size=patch_size,
        stride=stride,
    )

    assert len(sliced_images) > 0, "Should return sliced images"
    assert all(
        isinstance(img, Image.Image) for img in sliced_images
    ), "Should return PIL Images"

    # Test get_slice_metadata
    metadata = TileUtilsv2.get_slice_metadata(
        image=image_tensor,
        patch_size=patch_size,
        stride=stride,
    )

    assert "original_image_size" in metadata, "Should contain original image size"
    assert "num_slices" in metadata, "Should contain number of slices"
    assert "slice_dimensions" in metadata, "Should contain slice dimensions"
    assert "overlap_ratio" in metadata, "Should contain overlap ratio"

    logger.info("✓ SAHI-specific features test passed")


def test_parameter_validation():
    """Test parameter validation."""
    logger.info("Testing parameter validation...")

    # Test valid parameters
    image_shape = (3, 512, 512)
    patch_size = 256
    stride = 128

    assert TileUtilsv2.validate_patch_parameters(
        image_shape, patch_size, stride
    ), "Valid parameters should pass"

    # Test invalid parameters
    try:
        TileUtilsv2.validate_patch_parameters(
            (512,), patch_size, stride
        )  # Only 1 dimension
        assert False, "Should raise ValueError for invalid shape"
    except ValueError:
        pass

    try:
        TileUtilsv2.validate_patch_parameters(
            image_shape, -1, stride
        )  # Negative patch size
        assert False, "Should raise ValueError for negative patch size"
    except ValueError:
        pass

    try:
        TileUtilsv2.validate_patch_parameters(
            image_shape, patch_size, -1
        )  # Negative stride
        assert False, "Should raise ValueError for negative stride"
    except ValueError:
        pass

    try:
        TileUtilsv2.validate_patch_parameters(
            image_shape, patch_size, patch_size + 1
        )  # Stride > patch_size
        assert False, "Should raise ValueError for stride > patch_size"
    except ValueError:
        pass

    logger.info("✓ Parameter validation test passed")


def test_patch_count_calculation():
    """Test patch count calculation."""
    logger.info("Testing patch count calculation...")

    height, width = 512, 512
    patch_size = 256
    stride = 128

    expected_count = TileUtilsv2.get_patch_count(height, width, patch_size, stride)

    # Calculate manually
    num_patches_h = ((height - patch_size) // stride) + 1
    num_patches_w = ((width - patch_size) // stride) + 1
    manual_count = num_patches_h * num_patches_w

    assert expected_count == manual_count, "Patch count calculation should be correct"

    logger.info(f"✓ Patch count calculation test passed - {expected_count} patches")


def test_performance_comparison():
    """Compare performance between TileUtils and TileUtilsv2."""
    logger.info("Testing performance comparison...")

    import time

    # Create larger test image
    large_image = torch.randn(3, 1024, 1024)
    large_image = (large_image - large_image.min()) / (
        large_image.max() - large_image.min()
    )

    patch_size = 256
    stride = 128

    # Test TileUtils
    start_time = time.time()
    patches_original, _ = TileUtils.get_patches_and_offset_info(
        image=large_image,
        patch_size=patch_size,
        stride=stride,
        channels=3,
    )
    original_time = time.time() - start_time

    # Test TileUtilsv2
    start_time = time.time()
    patches_v2, _ = TileUtilsv2.get_patches_and_offset_info(
        image=large_image,
        patch_size=patch_size,
        stride=stride,
        channels=3,
    )
    v2_time = time.time() - start_time

    logger.info(
        f"Original TileUtils: {original_time:.4f}s, {patches_original.shape[0]} patches"
    )
    logger.info(f"TileUtilsv2: {v2_time:.4f}s, {patches_v2.shape[0]} patches")

    # Both should produce the same number of patches
    assert (
        patches_original.shape[0] == patches_v2.shape[0]
    ), "Should produce same number of patches"

    logger.info("✓ Performance comparison test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RUNNING TILE UTILS V2 TESTS")
    logger.info("=" * 60)

    try:
        test_basic_functionality()
        test_edge_cases()
        test_sahi_specific_features()
        test_parameter_validation()
        test_patch_count_calculation()
        test_performance_comparison()

        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    run_all_tests()
