"""
Utility functions for image tiling and patch extraction.
"""

import logging
import math
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import torch
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import traceback
from tqdm import tqdm


logger = logging.getLogger(__name__)


def read_image(image_path: str) -> Image.Image:
    """Load an image from a file path."""
    image = Image.open(image_path)
    ImageOps.exif_transpose(image, in_place=True)
    return image

@lru_cache(maxsize=512)
def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get the dimensions of an image."""
    image = read_image(image_path)
    width, height = image.size
    return width, height


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
    """Utility class for extracting tiles/patches from images using simple arithmetic operations."""

    @staticmethod
    @lru_cache(maxsize=128)
    def get_padding_size(height, width, patch_size: int, stride: int):
        H = height
        W = width

        # Calculate padding needed to make dimensions multiples of patch_size
        pad_h = math.ceil((H - patch_size) / stride) * stride - (H - patch_size)
        pad_w = math.ceil((W - patch_size) / stride) * stride - (W - patch_size)

        return pad_h, pad_w

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

        patches = TileUtils.get_patches(image, patch_size, stride, channels)

        # Calculate offset info using simple arithmetic
        offset_info = TileUtils._calculate_offset_info_simple(H, W, patch_size, stride)
        offset_info["file_name"] = file_name or "unknown"

        # Validate results if requested
        if validate:
            validate_results(image, patches, offset_info)

        return patches, offset_info

    @staticmethod
    @lru_cache(maxsize=128)
    def get_offset_info(
        width: int,
        height: int,
        patch_size: int,
        stride: int,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        pad_h, pad_w = TileUtils.get_padding_size(height, width, patch_size, stride)

        result = TileUtils._calculate_offset_info_simple(
            height + pad_h,
            width + pad_w,
            patch_size,
            stride,
        )
        result["file_name"] = file_name or "unknown"
        return result

    @staticmethod
    @lru_cache(maxsize=128)
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
    def pad_image_to_patch_size(
        image: torch.Tensor, patch_size: int, stride: int
    ) -> torch.Tensor:
        """Pad image to make it divisible by patch size."""
        C, H, W = image.shape
        pad_h, pad_w = TileUtils.get_padding_size(H, W, patch_size, stride)
        with torch.no_grad():
            if pad_h > 0 or pad_w > 0:
                image = torch.nn.functional.pad(
                    image, (0, pad_w, 0, pad_h), mode="constant", value=0
                )
        return image


def save_image_tiles(
    image_path: str,
    output_path: str,
    patch_size: int,
    stride: int,
) -> List[str]:
    """
    Load an image and save all tiles to disk with numbered filenames.
    
    Args:
        image_path (str): Path to the input image file.
        output_path (str): Directory path where tiles will be saved.
        patch_size (int): Size of each tile (square tiles).
        stride (int): Stride between tiles.
    
    Returns:
        List[str]: List of paths to saved tile files.
    
    Example:
        >>> saved_tiles = save_image_tiles(
        ...     "input.jpg", 
        ...     "output_tiles/", 
        ...     patch_size=512, 
        ...     stride=256
        ... )
    """
    # Load the image using read_image
    image = read_image(image_path)
    
    # Convert PIL image to torch tensor (C, H, W) format
    import torchvision.transforms.functional as F
    image_tensor = F.to_tensor(image)
    
    # Extract tiles using TileUtils
    tiles, offset_info = TileUtils.get_patches_and_offset_info(
        image_tensor,
        patch_size=patch_size,
        stride=stride,
        file_name=Path(image_path).name
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the original filename without extension
    original_name = Path(image_path).stem
    original_ext = Path(image_path).suffix
    
    saved_paths = []
    
    # Save each tile to disk
    def _save_tile(tile, output_dir, original_name, original_ext, i):
        # Convert tensor back to PIL image
        tile_image = F.to_pil_image(tile)
        
        # Create numbered filename
        tile_filename = f"{original_name}_tile_{i:04d}{original_ext}"
        tile_path = output_dir / tile_filename

        # Save the tile
        tile_image.save(str(tile_path))
        saved_paths.append(str(tile_path))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_save_tile, tile, output_dir, original_name, original_ext, i) for i, tile in enumerate(tiles)]
        for future in futures:
            future.result()
    
    logger.debug(f"Saved {tiles.shape[0]} tiles from {image_path} to {output_path}")
    
    return saved_paths


def save_tiles_for_directory(
    images_dir: str,
    output_path: str,
    patch_size: int,
    stride: int,
    patterns: tuple = ("*.jpg", "*.png", "*.jpeg", "*.tiff", "*.bmp"),
) -> Dict[str, List[str]]:
    """
    Load all images from a directory and save tiles for each image.
    
    Args:
        images_dir (str): Directory containing input images.
        output_path (str): Directory path where tiles will be saved.
        patch_size (int): Size of each tile (square tiles).
        stride (int): Stride between tiles.
        patterns (tuple): File patterns to match for images. Defaults to common image formats.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping original image paths to lists of saved tile paths.
    
    Example:
        >>> results = save_tiles_for_directory(
        ...     images_dir="input_images/",
        ...     output_path="output_tiles/",
        ...     patch_size=512,
        ...     stride=256
        ... )
        >>> print(f"Processed {len(results)} images")
    """
    # Get all image paths from the directory
    image_paths = get_images_paths(images_dir, patterns=patterns)
    
    if not image_paths:
        logger.warning(f"No images found in directory: {images_dir}")
        return {}
    
    logger.info(f"Found {len(image_paths)} images in {images_dir}")
    
    results = {}
    
    # Process each image
    error_count = 0
    tqdm_image_paths = tqdm(total=len(image_paths), desc="Processing images")
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(save_image_tiles, image_path=image_path, output_path=output_path, patch_size=patch_size, stride=stride) for image_path in image_paths]
        for image_path, future in zip(image_paths, futures):
            try:
                saved_tiles = future.result()
                results[image_path] = saved_tiles
                tqdm_image_paths.update(1)

            except KeyboardInterrupt:
                raise KeyboardInterrupt
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                error_count += 1
                results[image_path] = []
                if error_count > 5:
                    raise Exception(f"Too many errors. Stopping. {traceback.format_exc()}")

    total_tiles = sum(len(tiles) for tiles in results.values())
    logger.info(f"Processed {len(results)} images, saved {total_tiles} total tiles to {output_path}")
    
    return results


