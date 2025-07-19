"""
Utility functions for image tiling and patch extraction.
"""

import logging
import math
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_images_paths(
    images_dir: str,
    patterns: tuple = ("*.jpg", "*.png", "*.jpeg", "*.tiff", "*.bmp"),
) -> List[str]:
    patterns_ = []
    for pattern in patterns:
        patterns_.append(pattern.upper())
        patterns_.append(pattern.lower())
    images_paths = chain.from_iterable([Path(images_dir).glob(p) for p in patterns_])
    images_paths = list(set(images_paths))
    return [str(path) for path in images_paths]


class TileUtils:
    """Utility class for extracting tiles/patches from images."""

    @staticmethod
    def get_patches(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
        channels: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract patches from an image tensor using unfolding.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.
            patch_size (int): Size of each patch (square patches).
            stride (int): Stride between patches.
            channels (Optional[int]): Expected number of channels. If None, uses image channels.

        Returns:
            torch.Tensor: Tensor of image patches with shape (num_patches, channels, patch_size, patch_size).

        Raises:
            ValueError: If image dimensions are invalid or patch_size > image dimensions.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image)}")

        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")

        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
            squeeze_output = True
        else:
            squeeze_output = False

        C, H, W = image.shape
        # Check if image is large enough for patches
        if H < patch_size or W < patch_size:
            raise ValueError(
                f"Image size ({H}x{W}) is smaller than patch_size ({patch_size})"
            )

        # Validate channel count if specified
        if channels is not None and C != channels:
            raise ValueError(f"Expected {channels} channels, got {C}")

        # Use unfold to create tiles
        # First unfold along height dimension
        unfolded_h = image.unfold(1, patch_size, stride)

        # Then unfold along width dimension
        tiles = unfolded_h.unfold(2, patch_size, stride)

        # Reshape to get individual tiles
        tiles = tiles.contiguous().view(C, -1, patch_size, patch_size)
        tiles = tiles.permute(
            1, 0, 2, 3
        )  # (num_patches, channels, patch_size, patch_size)

        if squeeze_output:
            tiles = tiles.squeeze(1)

        return tiles

    @staticmethod
    def get_patches_and_offset_info(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
        channels: Optional[int] = None,
        file_name: Optional[str] = None,
        validate: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract patches from an image and compute offset information.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.
            patch_size (int): Size of each patch (square patches).
            stride (int): Stride between patches.
            channels (Optional[int]): Expected number of channels. If None, uses image channels.
            file_name (Optional[str]): File name for the offset info.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - Tensor of patches with shape (num_patches, channels, patch_size, patch_size)
                - Dictionary with offset information including x_offset, y_offset, x_end, y_end, file_name

        Raises:
            ValueError: If image dimensions are invalid or patch_size > image dimensions.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image)}")

        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")

        C, H, W = image.shape

        # Validate channel count if specified
        if channels is not None and C != channels:
            raise ValueError(f"Expected {channels} channels, got {C}")

        # Handle case where image is too small for patches
        if H <= patch_size or W <= patch_size:
            offset_info = {
                "y_offset": [0],
                "x_offset": [0],
                "y_end": [H],  # Use original dimensions for annotation clipping
                "x_end": [W],
                "file_name": file_name or "unknown",
            }
            return image, offset_info

        padded_image = TileUtils.pad_image_to_patch_size(image, patch_size, stride)
        C, H, W = padded_image.shape

        # Extract patches from padded image
        tiles = TileUtils.get_patches(padded_image, patch_size, stride, channels)

        # Calculate offset information for original image dimensions
        offset_info = TileUtils._calculate_offset_info(
            H,
            W,
            patch_size,
            stride,
            file_name,
        )

        # validate tiling
        if validate:
            TileUtils.validate_results(padded_image, tiles, offset_info)

        return tiles, offset_info

    @staticmethod
    def get_padding_size(height, width, patch_size: int, stride: int):
        H = height
        W = width

        # Calculate padding needed to make dimensions multiples of patch_size
        pad_h = math.ceil((H - patch_size) / stride) * stride - (H - patch_size)
        pad_w = math.ceil((W - patch_size) / stride) * stride - (W - patch_size)

        return pad_h, pad_w

    @staticmethod
    def pad_image_to_patch_size(
        image: torch.Tensor, patch_size: int, stride: int
    ) -> torch.Tensor:
        """
        Pad image with patch_size on right and bottom only to ensure complete tile coverage.
        """
        C, H, W = image.shape

        # Calculate padding needed to make dimensions multiples of patch_size
        pad_h, pad_w = TileUtils.get_padding_size(H, W, patch_size, stride)

        # Pad only on right and bottom with zeros
        padded_image = torch.nn.functional.pad(
            image, (0, pad_w, 0, pad_h), mode="constant", value=0
        )

        return padded_image

    @staticmethod
    def validate_results(
        image: torch.Tensor, tiles: torch.Tensor, offset_info: Dict[str, Any]
    ) -> None:
        """Validate that tiles match the original image regions using advanced indexing."""
        # Convert offset info to tensors for vectorized operations
        x_offsets = torch.tensor(offset_info["x_offset"])
        y_offsets = torch.tensor(offset_info["y_offset"])
        x_ends = torch.tensor(offset_info["x_end"])
        y_ends = torch.tensor(offset_info["y_end"])

        # Extract all regions at once using advanced indexing
        extracted_regions = torch.stack(
            [
                image[:, y_offsets[i] : y_ends[i], x_offsets[i] : x_ends[i]]
                for i in range(tiles.shape[0])
            ]
        )

        # Single vectorized comparison for all tiles
        assert torch.allclose(
            tiles, extracted_regions, atol=1e-6
        ), "error in tiling. Extracted value and offsets don't match"

    @staticmethod
    def _calculate_offset_info(
        height: int,
        width: int,
        patch_size: int,
        stride: int,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate offset information for patches.

        Args:
            height (int): Image height.
            width (int): Image width.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.
            file_name (Optional[str]): File name for the offset info.

        Returns:
            Dict[str, Any]: Offset information dictionary.
        """
        # Calculate number of patches in each dimension
        x = torch.arange(0, width).reshape((1, -1))
        y = torch.arange(0, height).reshape((-1, 1))

        ones = torch.ones((3, height, width))

        xx = TileUtils.get_patches(
            ones * x,
            patch_size,
            stride,
        )

        yy = TileUtils.get_patches(ones * y, patch_size, stride)
        x_offset = (
            xx.min(keepdim=True, dim=1)[0]
            .min(keepdim=True, dim=2)[0]
            .min(keepdim=True, dim=3)[0]
            .squeeze()
        ).int()

        y_offset = (
            yy.min(keepdim=True, dim=1)[0]
            .min(keepdim=True, dim=2)[0]
            .min(keepdim=True, dim=3)[0]
            .squeeze()
        ).int()

        return {
            "y_offset": y_offset.tolist(),
            "x_offset": x_offset.tolist(),
            "y_end": (y_offset + patch_size).tolist(),
            "x_end": (x_offset + patch_size).tolist(),
            "file_name": file_name or "unknown",
        }

    @staticmethod
    def validate_patch_parameters(
        image_shape: Tuple[int, ...], patch_size: int, stride: int
    ) -> bool:
        """
        Validate patch extraction parameters.

        Args:
            image_shape (Tuple[int, ...]): Shape of the image tensor.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.

        Returns:
            bool: True if parameters are valid.

        Raises:
            ValueError: If parameters are invalid.
        """
        if len(image_shape) < 2:
            raise ValueError("Image must have at least 2 dimensions")

        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if stride <= 0:
            raise ValueError("stride must be positive")

        if stride > patch_size:
            raise ValueError("stride cannot be larger than patch_size")

        return True

    @staticmethod
    def get_patch_count(height: int, width: int, patch_size: int, stride: int) -> int:
        """
        Calculate the number of patches that will be extracted.

        Args:
            height (int): Image height.
            width (int): Image width.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.

        Returns:
            int: Number of patches.
        """
        num_patches_h = ((height - patch_size) // stride) + 1
        num_patches_w = ((width - patch_size) // stride) + 1
        patch_count = num_patches_h * num_patches_w
        return patch_count
