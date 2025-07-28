"""
Example demonstrating the TileUtilsv2 functionality using SAHI slicing.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image

from src.wildetect.core.data.utils import TileUtils, TileUtilsv2


def demonstrate_tile_utils_v2():
    """Demonstrate the TileUtilsv2 functionality."""
    print("=" * 60)
    print("TESTING TILE UTILS V2 (SAHI-BASED)")
    print("=" * 60)

    # Create a sample image tensor (3 channels, 800x600)
    height, width = 2000, 2000
    channels = 3

    # Create a sample image with some pattern
    image_tensor = torch.randn(channels, height, width)
    
    # Normalize to [0, 1] range for better testing
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

    print(f"Original image shape: {image_tensor.shape}")

    # Test patch extraction
    patch_size = 800
    stride = 640

    try:
        # Test TileUtilsv2
        print("\n--- Testing TileUtilsv2 ---")
        
        # Validate parameters
        is_valid = TileUtilsv2.validate_patch_parameters(
            image_shape=image_tensor.shape, patch_size=patch_size, stride=stride
        )
        print(f"Parameters valid: {is_valid}")

        # Get expected patch count
        expected_count = TileUtilsv2.get_patch_count(height, width, patch_size, stride)
        print(f"Expected patch count: {expected_count}")

        # Extract patches using TileUtilsv2
        patches_v2, offset_info_v2 = TileUtilsv2.get_patches_and_offset_info(
            image=image_tensor,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            file_name="sample_image_v2.jpg",
            validate=True,
        )

        print(f"TileUtilsv2 - Extracted patches shape: {patches_v2.shape}")
        print(f"TileUtilsv2 - Number of patches: {patches_v2.shape[0]}")
        print(f"TileUtilsv2 - Offset info keys: {list(offset_info_v2.keys())}")
        print(f"TileUtilsv2 - Number of offsets: {len(offset_info_v2['x_offset'])}")

        # Test original TileUtils for comparison
        print("\n--- Testing Original TileUtils ---")
        
        patches_original, offset_info_original = TileUtils.get_patches_and_offset_info(
            image=image_tensor,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            file_name="sample_image_original.jpg",
            validate=True,
        )

        print(f"Original - Extracted patches shape: {patches_original.shape}")
        print(f"Original - Number of patches: {patches_original.shape[0]}")
        print(f"Original - Offset info keys: {list(offset_info_original.keys())}")
        print(f"Original - Number of offsets: {len(offset_info_original['x_offset'])}")

        # Compare results
        print("\n--- Comparing Results ---")
        
        # Check if number of patches match
        patches_match = patches_v2.shape[0] == patches_original.shape[0]
        print(f"Number of patches match: {patches_match}")
        
        # Check if patch shapes match
        shape_match = patches_v2.shape == patches_original.shape
        print(f"Patch shapes match: {shape_match}")
        
        # Check if offset info matches
        offset_count_match = (
            len(offset_info_v2['x_offset']) == len(offset_info_original['x_offset'])
        )
        print(f"Offset count match: {offset_count_match}")
        
        # Test SAHI-specific features
        print("\n--- Testing SAHI-Specific Features ---")
        
        # Test get_sliced_images
        sliced_images = TileUtilsv2.get_sliced_images(
            image=image_tensor,
            patch_size=patch_size,
            stride=stride,
        )
        print(f"SAHI sliced images count: {len(sliced_images)}")
        print(f"First sliced image size: {sliced_images[0].size if sliced_images else 'N/A'}")
        
        # Test get_slice_metadata
        metadata = TileUtilsv2.get_slice_metadata(
            image=image_tensor,
            patch_size=patch_size,
            stride=stride,
        )
        print(f"SAHI metadata: {metadata}")

        print("\n✓ All TileUtilsv2 tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_edge_cases():
    """Test edge cases for TileUtilsv2."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # Test with small image
    small_image = torch.randn(3, 100, 100)
    small_image = (small_image - small_image.min()) / (small_image.max() - small_image.min())
    
    try:
        patches, offset_info = TileUtilsv2.get_patches_and_offset_info(
            image=small_image,
            patch_size=256,  # Larger than image
            stride=128,
            channels=3,
        )
        print(f"Small image test - patches shape: {patches.shape}")
        print(f"Small image test - should return original image: {patches.shape[0] == 1}")
    except Exception as e:
        print(f"Small image test error: {e}")

    # Test with exact patch size
    exact_image = torch.randn(3, 256, 256)
    exact_image = (exact_image - exact_image.min()) / (exact_image.max() - exact_image.min())
    
    try:
        patches, offset_info = TileUtilsv2.get_patches_and_offset_info(
            image=exact_image,
            patch_size=256,
            stride=256,
            channels=3,
        )
        print(f"Exact size test - patches shape: {patches.shape}")
        print(f"Exact size test - should return 1 patch: {patches.shape[0] == 1}")
    except Exception as e:
        print(f"Exact size test error: {e}")


def test_performance_comparison():
    """Compare performance between TileUtils and TileUtilsv2."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    import time

    # Create a larger test image
    large_image = torch.randn(3, 1024, 1024)
    large_image = (large_image - large_image.min()) / (large_image.max() - large_image.min())
    
    patch_size = 256
    stride = 128

    # Test TileUtils
    start_time = time.time()
    try:
        patches_original, _ = TileUtils.get_patches_and_offset_info(
            image=large_image,
            patch_size=patch_size,
            stride=stride,
            channels=3,
        )
        original_time = time.time() - start_time
        print(f"Original TileUtils time: {original_time:.4f}s")
        print(f"Original TileUtils patches: {patches_original.shape[0]}")
    except Exception as e:
        print(f"Original TileUtils error: {e}")
        original_time = None

    # Test TileUtilsv2
    start_time = time.time()
    try:
        patches_v2, _ = TileUtilsv2.get_patches_and_offset_info(
            image=large_image,
            patch_size=patch_size,
            stride=stride,
            channels=3,
        )
        v2_time = time.time() - start_time
        print(f"TileUtilsv2 time: {v2_time:.4f}s")
        print(f"TileUtilsv2 patches: {patches_v2.shape[0]}")
        
        if original_time:
            speedup = original_time / v2_time
            print(f"Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"TileUtilsv2 error: {e}")


if __name__ == "__main__":
    demonstrate_tile_utils_v2()
    test_edge_cases()
    test_performance_comparison() 