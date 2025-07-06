"""
Example demonstrating the improved TileUtils functionality.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image

from src.wildetect.core.data import DataLoader, TileUtils, create_loader


def demonstrate_tile_utils():
    """Demonstrate the TileUtils functionality."""

    # Create a sample image tensor (3 channels, 800x600)
    height, width = 600, 800
    channels = 3

    # Create a sample image with some pattern
    image_tensor = torch.randn(channels, height, width)

    print(f"Original image shape: {image_tensor.shape}")

    # Test patch extraction
    patch_size = 256
    stride = 128

    try:
        # Validate parameters
        is_valid = TileUtils.validate_patch_parameters(
            image_shape=image_tensor.shape, patch_size=patch_size, stride=stride
        )
        print(f"Parameters valid: {is_valid}")

        # Get expected patch count
        expected_count = TileUtils.get_patch_count(height, width, patch_size, stride)
        print(f"Expected patch count: {expected_count}")

        # Extract patches
        patches, offset_info = TileUtils.get_patches_and_offset_info(
            image=image_tensor,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            file_name="sample_image.jpg",
        )

        print(f"Extracted patches shape: {patches.shape}")
        print(f"Number of patches: {patches.shape[0]}")
        print(f"Offset info keys: {list(offset_info.keys())}")
        print(f"Number of offsets: {len(offset_info['x_offset'])}")

        # Verify offset information
        assert len(offset_info["x_offset"]) == patches.shape[0], "Offset count mismatch"
        assert len(offset_info["y_offset"]) == patches.shape[0], "Offset count mismatch"

        print("✓ All tests passed!")

    except Exception as e:
        print(f"Error: {e}")


def demonstrate_loader_integration():
    """Demonstrate loader integration with TileUtils."""

    # This would require actual image files
    print("Loader integration demonstration:")
    print("- TileUtils is now integrated into the DataLoader")
    print("- The loader uses TileUtils for patch extraction")
    print("- Validation is performed before tile creation")
    print("- Better error handling and logging")


if __name__ == "__main__":
    print("=== TileUtils Demonstration ===\n")

    print("1. Basic TileUtils functionality:")
    demonstrate_tile_utils()

    print("\n2. Loader integration:")
    demonstrate_loader_integration()

    print("\n=== Key Improvements ===")
    print("✓ Better error handling and validation")
    print("✓ Flexible channel support (not hard-coded to 3 channels)")
    print("✓ Comprehensive documentation and type hints")
    print("✓ Utility methods for parameter validation")
    print("✓ Integration with DataLoader for seamless usage")
    print("✓ Improved offset calculation algorithm")
