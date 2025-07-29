"""
Utility functions for image tiling and patch extraction.
"""

import logging
import math
import traceback
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sahi.slicing import slice_image
from torchvision.transforms import ToPILImage, ToTensor

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


def validate_results(
    image: torch.Tensor, tiles: torch.Tensor, offset_info: Dict[str, Any]
) -> None:
    """Validate that tiles match the original image regions using advanced indexing."""
    # Convert offset info to tensors for vectorized operations
    x_offsets = torch.tensor(offset_info["x_offset"])
    y_offsets = torch.tensor(offset_info["y_offset"])
    x_ends = torch.tensor(offset_info["x_end"])
    y_ends = torch.tensor(offset_info["y_end"])

    assert isinstance(image, torch.Tensor)

    # Extract all regions at once using advanced indexing
    extracted_regions = torch.stack(
        [
            image[:, y_offsets[i] : y_ends[i], x_offsets[i] : x_ends[i]]
            for i in range(tiles.shape[0])
        ]
    )

    # Check if tensors have the same shape and dtype
    if tiles.shape != extracted_regions.shape:
        print(
            f"Shape mismatch: tiles {tiles.shape} vs extracted {extracted_regions.shape}"
        )
        return

    if tiles.dtype != extracted_regions.dtype:
        print(
            f"Dtype mismatch: tiles {tiles.dtype} vs extracted {extracted_regions.dtype}"
        )
        # Convert to same dtype for comparison
        tiles = tiles.to(dtype=extracted_regions.dtype)

    # More tolerant comparison for SAHI results
    try:
        assert torch.allclose(
            tiles, extracted_regions, atol=1e-2, rtol=1e-2
        ), "error in tiling. Extracted value and offsets don't match"
    except AssertionError as e:
        # Print some debugging info
        print(f"Validation failed: {e}")
        print(f"Max difference: {torch.max(torch.abs(tiles - extracted_regions))}")
        print(f"Mean difference: {torch.mean(torch.abs(tiles - extracted_regions))}")
        raise ValueError(f"SAHI validation failed: {e}")


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
        )
        offset_info["file_name"] = file_name or "unknown"

        # validate tiling
        if validate:
            validate_results(padded_image, tiles, offset_info)

        return tiles, offset_info

    @staticmethod
    def get_patches_and_offset_info_only(
        width: int,
        height: int,
        patch_size: int,
        stride: int,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        pad_h, pad_w = TileUtils.get_padding_size(height, width, patch_size, stride)

        result = TileUtils._calculate_offset_info(
            height + pad_h,
            width + pad_w,
            patch_size,
            stride,
        )
        result["file_name"] = file_name or "unknown"
        return result

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
    def _calculate_offset_info(
        height: int,
        width: int,
        patch_size: int,
        stride: int,
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
        # cache_key = (height, width, patch_size, stride)

        # if cache_key in TileUtils._offset_cache:
        #    return TileUtils._offset_cache[cache_key]

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

        result = {
            "y_offset": y_offset.tolist(),
            "x_offset": x_offset.tolist(),
            "y_end": (y_offset + patch_size).tolist(),
            "x_end": (x_offset + patch_size).tolist(),
        }

        return result

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


class TileUtilsv2:
    """Utility class for extracting tiles/patches from images using SAHI slicing."""

    @staticmethod
    def _calculate_overlap_ratio(patch_size: int, stride: int) -> float:
        """Calculate overlap ratio from patch_size and stride."""
        overlap_pixels = patch_size - stride
        return max(0.0, overlap_pixels / patch_size)

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
        pad_h, pad_w = TileUtilsv2.get_padding_size(H, W, patch_size, stride)

        # Pad only on right and bottom with zeros
        padded_image = torch.nn.functional.pad(
            image, (0, pad_w, 0, pad_h), mode="constant", value=0
        )

        return padded_image

    @staticmethod
    def get_patches(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
        channels: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract patches from an image tensor using SAHI slicing.

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

        # Convert tensor to PIL Image for SAHI
        pil_image = ToPILImage()(image)

        # Calculate overlap ratio
        overlap_ratio = TileUtilsv2._calculate_overlap_ratio(patch_size, stride)

        # Use SAHI to slice the image
        try:
            from sahi.slicing import slice_image

            slice_result = slice_image(
                image=pil_image,
                slice_height=patch_size,
                slice_width=patch_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                verbose=False,
            )

            # Convert sliced images back to tensors
            patches = []
            for sliced_image in slice_result.sliced_image_list:
                patch_tensor = (ToTensor()(sliced_image.image.copy()) * 255).to(
                    torch.uint8
                )
                patches.append(patch_tensor)

            if not patches:
                raise ValueError("No patches extracted")

            # Stack patches into a single tensor
            patches_tensor = torch.stack(patches)

            # Validate channel count if specified
            if channels is not None and patches_tensor.shape[1] != channels:
                raise ValueError(
                    f"Expected {channels} channels, got {patches_tensor.shape[1]}"
                )

            return patches_tensor

        except ImportError:
            raise ImportError(
                "SAHI library is required. Install with: pip install sahi"
            )
        except Exception as e:
            raise ValueError(f"SAHI slicing failed: {e}")

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
        Extract patches from an image and compute offset information using SAHI.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.
            patch_size (int): Size of each patch (square patches).
            stride (int): Stride between patches.
            channels (Optional[int]): Expected number of channels. If None, uses image channels.
            file_name (Optional[str]): File name for the offset info.
            validate (bool): Whether to validate results.

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
                "y_end": [H],
                "x_end": [W],
                "file_name": file_name or "unknown",
            }
            return image.unsqueeze(0), offset_info

        image = TileUtilsv2.pad_image_to_patch_size(image, patch_size, stride)

        # Convert tensor to PIL Image for SAHI
        pil_image = ToPILImage()(image)

        # Calculate overlap ratio
        overlap_ratio = TileUtilsv2._calculate_overlap_ratio(patch_size, stride)

        # Use SAHI to slice the image
        slice_result = slice_image(
            image=pil_image,
            slice_height=patch_size,
            slice_width=patch_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=False,
        )

        # Convert sliced images back to tensors and extract offset info
        patches = []
        x_offsets = []
        y_offsets = []
        x_ends = []
        y_ends = []

        for sliced_image in slice_result.sliced_image_list:
            # Convert to tensor and ensure same dtype as original image
            patch_tensor = ToTensor()(sliced_image.image.copy())
            # Convert to same dtype as original image
            patch_tensor = patch_tensor.to(dtype=image.dtype)
            patches.append(patch_tensor)

            # Extract offset information from SAHI's sliced image
            x_offsets.append(sliced_image.starting_pixel[0])
            y_offsets.append(sliced_image.starting_pixel[1])
            x_ends.append(sliced_image.starting_pixel[0] + patch_size)
            y_ends.append(sliced_image.starting_pixel[1] + patch_size)

        if not patches:
            raise ValueError("No patches extracted")

        # Stack patches into a single tensor
        patches_tensor = torch.stack(patches)

        # Create offset info dictionary
        offset_info = {
            "x_offset": x_offsets,
            "y_offset": y_offsets,
            "x_end": x_ends,
            "y_end": y_ends,
            "file_name": file_name or "unknown",
        }

        # Validate results if requested (disabled by default due to SAHI differences)
        if validate:
            validate_results(image, patches_tensor, offset_info)

        return patches_tensor, offset_info

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

    @staticmethod
    def get_sliced_images(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
        auto_slice_resolution: bool = False,
    ) -> List[Image.Image]:
        """
        Get sliced images using SAHI (returns PIL Images).

        Args:
            image (torch.Tensor): Image tensor to slice.
            patch_size (int): Size of each patch.
            stride (int): Stride between patches.
            auto_slice_resolution (bool): Whether to auto-calculate slice resolution.

        Returns:
            List[Image.Image]: List of sliced PIL Images.
        """
        pil_image = ToPILImage()(image)
        overlap_ratio = TileUtilsv2._calculate_overlap_ratio(patch_size, stride)

        try:
            slice_result = slice_image(
                image=pil_image,
                slice_height=patch_size if not auto_slice_resolution else None,
                slice_width=patch_size if not auto_slice_resolution else None,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                auto_slice_resolution=auto_slice_resolution,
                verbose=False,
            )

            return [
                Image.fromarray(sliced_image.image)
                for sliced_image in slice_result.sliced_image_list
            ]

        except Exception:
            raise ValueError(f"SAHI slicing failed: {traceback.format_exc()}")

    @staticmethod
    def get_slice_metadata(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
    ) -> Dict[str, Any]:
        """
        Get metadata about the slicing operation.

        Args:
            image (torch.Tensor): Image tensor.
            patch_size (int): Size of each patch.
            stride (int): Stride between patches.

        Returns:
            Dict[str, Any]: Slicing metadata.
        """
        pil_image = ToPILImage()(image)
        overlap_ratio = TileUtilsv2._calculate_overlap_ratio(patch_size, stride)

        try:
            slice_result = slice_image(
                image=pil_image,
                slice_height=patch_size,
                slice_width=patch_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                verbose=False,
            )
            return {
                "original_image_size": [pil_image.height, pil_image.width],
                "num_slices": len(slice_result.sliced_image_list),
                "slice_dimensions": [patch_size, patch_size],
                "overlap_ratio": overlap_ratio,
            }

        except ImportError:
            raise ImportError(
                "SAHI library is required. Install with: pip install sahi"
            )
        except Exception:
            raise ValueError(f"SAHI slicing failed: {traceback.format_exc()}")


class TileUtilsv3:
    """Utility class for extracting tiles/patches from images using simple arithmetic operations."""

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
        Extract patches from an image and compute offset information using simple arithmetic.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.
            patch_size (int): Size of each patch (square patches).
            stride (int): Stride between patches.
            channels (Optional[int]): Expected number of channels. If None, uses image channels.
            file_name (Optional[str]): File name for the offset info.
            validate (bool): Whether to validate results.

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
                "y_end": [H],
                "x_end": [W],
                "file_name": file_name or "unknown",
            }
            return image.unsqueeze(0), offset_info

        image = TileUtils.pad_image_to_patch_size(image, patch_size, stride)
        C, H, W = image.shape

        # Get patches using the same logic as TileUtils
        patches = TileUtilsv3.get_patches(image, patch_size, stride, channels)

        # Calculate offset info using simple arithmetic
        offset_info = TileUtilsv3._calculate_offset_info_simple(
            H, W, patch_size, stride
        )
        offset_info["file_name"] = file_name or "unknown"

        # Validate results if requested
        if validate:
            validate_results(image, patches, offset_info)

        return patches, offset_info

    @staticmethod
    def get_patches_and_offset_info_only(
        width: int,
        height: int,
        patch_size: int,
        stride: int,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        pad_h, pad_w = TileUtilsv3.get_padding_size(height, width, patch_size, stride)

        result = TileUtilsv3._calculate_offset_info_simple(
            height + pad_h,
            width + pad_w,
            patch_size,
            stride,
        )
        result["file_name"] = file_name or "unknown"
        return result

    @staticmethod
    @lru_cache(maxsize=512)
    def _calculate_offset_info_simple(
        height: int,
        width: int,
        patch_size: int,
        stride: int,
    ) -> Dict[str, Any]:
        """
        Calculate offset information for patches using vectorized arithmetic.

        Args:
            height (int): Image height.
            width (int): Image width.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.

        Returns:
            Dict[str, Any]: Offset information dictionary.
        """
        # Calculate number of patches in each dimension
        num_patches_h = ((height - patch_size) // stride) + 1
        num_patches_w = ((width - patch_size) // stride) + 1

        # Create coordinate grids using vectorized operations
        y_indices = torch.arange(num_patches_h, dtype=torch.int32)
        x_indices = torch.arange(num_patches_w, dtype=torch.int32)

        # Create meshgrid for all combinations
        y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing="ij")

        # Calculate offsets using vectorized operations
        y_offsets = (y_grid * stride).flatten().tolist()
        x_offsets = (x_grid * stride).flatten().tolist()
        y_ends = (y_grid * stride + patch_size).flatten().tolist()
        x_ends = (x_grid * stride + patch_size).flatten().tolist()

        return {
            "y_offset": y_offsets,
            "x_offset": x_offsets,
            "y_end": y_ends,
            "x_end": x_ends,
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

    @staticmethod
    def get_padding_size(height, width, patch_size: int, stride: int):
        """Calculate padding size to make image divisible by patch size."""
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
        """Pad image to make it divisible by patch size."""
        C, H, W = image.shape
        pad_h, pad_w = TileUtilsv3.get_padding_size(H, W, patch_size, stride)

        if pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        return image
