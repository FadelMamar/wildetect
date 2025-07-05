"""
Utility functions for image tiling and patch extraction.
"""

import torch
import logging
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class TileUtils:
    """Utility class for extracting tiles/patches from images."""
    
    @staticmethod
    def get_patches(
        image: torch.Tensor, 
        patch_size: int, 
        stride: int,
        channels: Optional[int] = None
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
        
        # Validate channel count if specified
        if channels is not None and C != channels:
            raise ValueError(f"Expected {channels} channels, got {C}")
        
        # Check if image is large enough for patches
        if H < patch_size or W < patch_size:
            raise ValueError(f"Image size ({H}x{W}) is smaller than patch_size ({patch_size})")

        # Use unfold to create tiles
        # First unfold along height dimension
        unfolded_h = image.unfold(1, patch_size, stride)

        # Then unfold along width dimension
        tiles = unfolded_h.unfold(2, patch_size, stride)

        # Reshape to get individual tiles
        tiles = tiles.contiguous().view(C, -1, patch_size, patch_size)
        tiles = tiles.permute(1, 0, 2, 3)  # (num_patches, channels, patch_size, patch_size)

        if squeeze_output:
            tiles = tiles.squeeze(1)

        return tiles

    @staticmethod
    def get_patches_and_offset_info(
        image: torch.Tensor, 
        patch_size: int, 
        stride: int,
        channels: Optional[int] = None,
        file_name: Optional[str] = None
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
            logger.debug(f"Image size ({H}x{W}) is too small for patch extraction with size {patch_size}")
            offset_info = {
                "y_offset": [0],
                "x_offset": [0],
                "y_end": [H],
                "x_end": [W],
                "file_name": file_name or "unknown"
            }
            return image.unsqueeze(0), offset_info

        # Extract patches
        tiles = TileUtils.get_patches(image, patch_size, stride, channels)

        # Calculate offset information
        offset_info = TileUtils._calculate_offset_info(H, W, patch_size, stride, file_name)

        return tiles, offset_info

    @staticmethod
    def _calculate_offset_info(
        height: int, 
        width: int, 
        patch_size: int, 
        stride: int,
        file_name: Optional[str] = None
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
        num_patches_h = (height - patch_size) // stride + 1
        num_patches_w = (width - patch_size) // stride + 1
        
        # Generate offset arrays
        y_offsets = [i * stride for i in range(num_patches_h)]
        x_offsets = [i * stride for i in range(num_patches_w)]
        
        # Generate end positions
        y_ends = [offset + patch_size for offset in y_offsets]
        x_ends = [offset + patch_size for offset in x_offsets]
        
        # Create all combinations for 2D patches
        y_offset_list = []
        x_offset_list = []
        y_end_list = []
        x_end_list = []
        
        for y_offset in y_offsets:
            for x_offset in x_offsets:
                y_offset_list.append(y_offset)
                x_offset_list.append(x_offset)
                y_end_list.append(y_offset + patch_size)
                x_end_list.append(x_offset + patch_size)
        
        return {
            "y_offset": y_offset_list,
            "x_offset": x_offset_list,
            "y_end": y_end_list,
            "x_end": x_end_list,
            "file_name": file_name or "unknown"
        }

    @staticmethod
    def validate_patch_parameters(
        image_shape: Tuple[int, ...], 
        patch_size: int, 
        stride: int
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
        
        height, width = image_shape[-2:]
        
        if height < patch_size or width < patch_size:
            raise ValueError(f"Image dimensions ({height}x{width}) must be >= patch_size ({patch_size})")
        
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
        if height < patch_size or width < patch_size:
            return 1  # Return single patch for small images
        
        num_patches_h = (height - patch_size) // stride + 1
        num_patches_w = (width - patch_size) // stride + 1
        
        return num_patches_h * num_patches_w
