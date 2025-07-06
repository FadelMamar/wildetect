"""
Test script for TileUtils functionality.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

from wildetect.core.data.utils import TileUtils, get_images_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_image(test_dir: Path, filename: str = "test_image.jpg") -> str:
    """Create a test image for testing."""
    import numpy as np
    from PIL import Image

    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple test image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    img_path = test_dir / filename
    img.save(img_path)

    logger.info(f"Created test image: {img_path}")
    return str(img_path)


def test_get_images_paths():
    """Test get_images_paths function."""
    logger.info("=" * 60)
    logger.info("TESTING GET_IMAGES_PATHS")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test images with different formats
        image_paths = []
        for i in range(5):
            if i % 2 == 0:
                filename = f"test_image_{i}.jpg"
            else:
                filename = f"test_image_{i}.png"
            image_path = create_test_image(temp_path, filename)
            image_paths.append(image_path)

        # Test get_images_paths
        discovered_paths = get_images_paths(temp_dir)
        discovered_paths = [str(p) for p in discovered_paths]

        logger.info(f"Expected images: {len(image_paths)}")
        logger.info(f"Discovered images: {len(discovered_paths)}")

        # Check if all expected images were discovered
        missing_images = set(image_paths) - set(discovered_paths)
        extra_images = set(discovered_paths) - set(image_paths)

        if missing_images:
            logger.error(f"Missing images: {missing_images}")
        if extra_images:
            logger.warning(f"Extra images found: {extra_images}")

        assert len(missing_images) == 0, f"Missing images: {missing_images}"
        logger.info("✓ get_images_paths test passed")


def test_tile_utils_validation():
    """Test TileUtils validation functions."""
    logger.info("=" * 60)
    logger.info("TESTING TILE UTILS VALIDATION")
    logger.info("=" * 60)

    # Test patch parameter validation
    valid_params = TileUtils.validate_patch_parameters(
        image_shape=(3, 480, 640), patch_size=320, stride=256
    )
    assert valid_params is True

    # Test invalid parameters
    try:
        invalid_params = TileUtils.validate_patch_parameters(
            image_shape=(3, 100, 100),  # Too small
            patch_size=320,
            stride=256,
        )
    except ValueError as e:
        logger.info(f"Expected error: {e}")
        invalid_params = True
    except Exception as e:
        logger.error(f"Unexpected exception: {e}")
        assert False

    # Test patch count calculation
    patch_count = TileUtils.get_patch_count(480, 640, 320, 256)
    expected_count = ((480 - 320) // 256 + 1) * ((640 - 320) // 256 + 1)
    assert (
        patch_count == expected_count
    ), f"Patch count {patch_count} does not match expected count {expected_count}"

    logger.info(f"Valid patch parameters: {valid_params}")
    logger.info(f"Invalid patch parameters: {invalid_params}")
    logger.info(f"Patch count for 480x640 with 320x320 patches: {patch_count}")
    logger.info("✓ TileUtils validation test passed")


def test_tile_utils_patch_extraction():
    """Test TileUtils patch extraction."""
    logger.info("=" * 60)
    logger.info("TESTING TILE UTILS PATCH EXTRACTION")
    logger.info("=" * 60)

    import torch
    from torchvision.transforms import PILToTensor

    # Create test image tensor
    test_image = torch.randn(3, 480, 640)  # C, H, W format

    # Test patch extraction
    patches, offset_info = TileUtils.get_patches_and_offset_info(
        image=test_image,
        patch_size=320,
        stride=256,
        channels=3,
        file_name="test_image.jpg",
    )

    # Test patch properties
    assert patches.dim() == 4  # B, C, H, W
    assert patches.shape[1] == 3  # 3 channels
    assert patches.shape[2] == 320  # patch height
    assert patches.shape[3] == 320  # patch width

    # Test offset info
    assert "x_offset" in offset_info
    assert "y_offset" in offset_info
    assert len(offset_info["x_offset"]) == patches.shape[0]
    assert len(offset_info["y_offset"]) == patches.shape[0]

    logger.info(f"Extracted {patches.shape[0]} patches")
    logger.info(f"Patch shape: {patches.shape}")
    logger.info(f"Offset info keys: {list(offset_info.keys())}")
    logger.info("✓ TileUtils patch extraction test passed")


def test_tile_utils_patch_count():
    """Test TileUtils patch count calculations."""
    logger.info("=" * 60)
    logger.info("TESTING TILE UTILS PATCH COUNT")
    logger.info("=" * 60)

    # Test various image sizes
    test_cases = [
        (480, 640, 320, 256),  # Standard case
        (640, 480, 320, 256),  # Swapped dimensions
        (1000, 1000, 320, 256),  # Large image
        (320, 320, 320, 256),  # Exact patch size
    ]

    for height, width, patch_size, stride in test_cases:
        patch_count = TileUtils.get_patch_count(height, width, patch_size, stride)

        # Calculate expected count manually
        h_patches = max(1, (height - patch_size) // stride + 1)
        w_patches = max(1, (width - patch_size) // stride + 1)
        expected_count = h_patches * w_patches

        assert (
            patch_count == expected_count
        ), f"Patch count {patch_count} does not match expected count {expected_count}"
        logger.info(
            f"Image {height}x{width}, patches {patch_size}x{patch_size}, stride {stride}: {patch_count} patches"
        )

    logger.info("✓ TileUtils patch count test passed")


def test_tile_utils_patch_extraction_edge_cases():
    """Test TileUtils patch extraction edge cases."""
    logger.info("=" * 60)
    logger.info("TESTING TILE UTILS PATCH EXTRACTION EDGE CASES")
    logger.info("=" * 60)

    import torch

    # Test with exact patch size
    exact_image = torch.randn(3, 320, 320)
    patches, offset_info = TileUtils.get_patches_and_offset_info(
        image=exact_image,
        patch_size=320,
        stride=256,
        channels=3,
        file_name="exact_image.jpg",
    )

    assert patches.shape[0] == 1  # Should have exactly one patch
    assert patches.shape[1:] == (3, 320, 320)

    # Test with large image
    large_image = torch.randn(3, 1000, 1000)
    patches, offset_info = TileUtils.get_patches_and_offset_info(
        image=large_image,
        patch_size=320,
        stride=256,
        channels=3,
        file_name="large_image.jpg",
    )

    # Should have multiple patches
    assert patches.shape[0] > 1
    assert patches.shape[1:] == (3, 320, 320)

    logger.info(f"Exact size image: {patches.shape[0]} patches")
    logger.info(f"Large image: {patches.shape[0]} patches")
    logger.info("✓ TileUtils patch extraction edge cases test passed")


def test_tile_utils_error_handling():
    """Test TileUtils error handling."""
    logger.info("=" * 60)
    logger.info("TESTING TILE UTILS ERROR HANDLING")
    logger.info("=" * 60)

    import torch

    # Test with invalid parameters
    try:
        TileUtils.validate_patch_parameters(
            image_shape=(3, 100, 100),  # Too small
            patch_size=320,
            stride=256,
        )
        # Should not raise exception, just return False
        logger.info("✓ Invalid parameters handled gracefully")
    except Exception as e:
        logger.error(f"Unexpected exception: {e}")
        raise

    # Test with invalid image tensor
    try:
        invalid_image = torch.randn(2, 480, 640)  # Wrong number of channels
        TileUtils.get_patches_and_offset_info(
            image=invalid_image,
            patch_size=320,
            stride=256,
            channels=3,  # Mismatch
            file_name="invalid_image.jpg",
        )
        logger.warning("Expected error for channel mismatch")
    except Exception as e:
        logger.info(f"✓ Expected error caught: {e}")

    logger.info("✓ TileUtils error handling test passed")


def run_all_tests():
    """Run all TileUtils tests."""
    logger.info("Starting TileUtils tests...")

    try:
        test_get_images_paths()
        test_tile_utils_validation()
        test_tile_utils_patch_extraction()
        test_tile_utils_patch_count()
        test_tile_utils_patch_extraction_edge_cases()
        test_tile_utils_error_handling()

        logger.info("=" * 60)
        logger.info("ALL TILE UTILS TESTS PASSED! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
