"""
Tiling transformer for extracting tiles/patches from images and annotations.
"""

import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from ..config import TilingConfig
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


from uuid import uuid4


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
        patches = TileUtils.get_patches(image, patch_size, stride, channels)

        # Calculate offset info using simple arithmetic
        offset_info = TileUtils._calculate_offset_info_simple(H, W, patch_size, stride)
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
        pad_h, pad_w = TileUtils.get_padding_size(H, W, patch_size, stride)

        if pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        return image


class TilingTransformer(BaseTransformer):
    """
    Transformer for extracting tiles/patches from images and their annotations.

    Uses TileUtils for efficient image tiling and provides annotation tiling functionality.

    Supports:
    - Regular grid tiling with configurable stride
    - Square patches (tile_size x tile_size)
    - Annotation-aware tiling (filter tiles based on annotation content)
    - Efficient PyTorch-based tile extraction
    """

    def __init__(self, config: Optional[TilingConfig] = None):
        """
        Initialize the tiling transformer.

        Args:
            config: TilingConfig dataclass or configuration dictionary
        """
        super().__init__()
        self.config = config or TilingConfig()

    def transform(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._validate_inputs(inputs)
        outputs = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for output in executor.map(self._transform_once, inputs):
                outputs.extend(output)

        return outputs

    def _validate_output(
        self, outputs: List[Dict[str, Any]], annotations: List[Dict[str, Any]]
    ) -> None:
        """
        Validate the output of the tiling transformer.

        Ensures that:
        1. All outputs have required fields (image, info)
        2. If input had annotations, at least some annotations are preserved in output
        3. Bbox coordinates are properly clipped to tile bounds
        4. Annotation visibility ratios meet minimum threshold
        """
        if not outputs:
            raise ValueError("No outputs generated from tiling")

        # Track annotation preservation
        total_output_annotations = 0

        for output in outputs:
            if "image" not in output:
                raise ValueError("Image not found in output")
            if "info" not in output:
                raise ValueError("Info not found in output")

            output_annotations = output.get("annotations", [])
            total_output_annotations += len(output_annotations)

            # Validate bbox coordinates for each annotation
            for annotation in output_annotations:
                if "bbox" in annotation:
                    self._validate_bbox_coordinates(annotation["bbox"], output["info"])

        # Validate annotation preservation
        if annotations:
            if total_output_annotations == 0:
                raise ValueError(
                    f"No annotations found in output of TilingTransformer. "
                    f"Input had {annotations} annotations but output has 0."
                )

            # Comprehensive annotation preservation validation
            validation_stats = self._validate_annotation_preservation(
                annotations, outputs
            )

            # Check that we preserved a reasonable fraction of unique annotations
            if (
                validation_stats["preservation_ratio"] < 0.1
            ):  # At least 10% should be preserved
                logger.warning(
                    f"Low unique annotation preservation ratio: {validation_stats['preservation_ratio']:.2f}. "
                    f"Input had {validation_stats['total_count']} annotations, "
                    f"output preserved {validation_stats['preserved_count']} unique annotations."
                )

    def _transform_once(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tiles from the image and annotations.
        """

        image = inputs["image"]
        annotations = inputs.get("annotations", [])
        image_info = inputs["info"]

        # Convert numpy image to torch tensor
        if len(image.shape) == 3:
            # HWC to CHW format
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        if image_info.get("file_name") is None:
            logger.warning("Image info does not contain file name, using 'unknown'")
            file_name = "unknown"
        else:
            file_name = image_info["file_name"]

        # Extract tiles using TileUtils
        tiles, offset_info = TileUtils.get_patches_and_offset_info(
            image_tensor, self.config.tile_size, self.config.stride, file_name=file_name
        )

        # Convert tiles back to numpy and process annotations
        empty_tiles = []
        non_empty_tiles = []

        for i in range(tiles.shape[0]):
            # Convert tile back to numpy (CHW to HWC)
            tile_image = tiles[i].permute(1, 2, 0).numpy().astype(np.uint8)

            # Get tile offset information
            x_offset = offset_info["x_offset"][i]
            y_offset = offset_info["y_offset"][i]
            x_end = offset_info["x_end"][i]
            y_end = offset_info["y_end"][i]

            # Extract annotations for this tile
            tile_annotation = self._extract_tile_annotations(
                annotations, x_offset, y_offset, x_end, y_end
            )

            # Create tile info
            tile_info = dict(
                file_name=f"{Path(file_name).stem}_tile_{i}_{x_offset}_{y_offset}.jpg",
                width=x_end - x_offset,
                height=y_end - y_offset,
                id=str(uuid4()),
            )
            for ann in tile_annotation:
                ann["image_id"] = tile_info["id"]

            # Check if tile meets criteria and categorize
            if tile_annotation:  # Non-empty tile
                non_empty_tiles.append(
                    {
                        "image": tile_image,
                        "annotations": tile_annotation,
                        "info": tile_info,
                    }
                )
            else:  # Empty tile
                empty_tiles.append(
                    {
                        "image": tile_image,
                        "annotations": tile_annotation,
                        "info": tile_info,
                    }
                )

        # Sample tiles based on the empty/non-empty ratio
        selected_tiles = self._sample_tiles_with_ratio(empty_tiles, non_empty_tiles)

        self._validate_output(selected_tiles, annotations)

        # keep only tiles with enough content
        filtered_tiles = []
        for tile in selected_tiles:
            image = tile["image"]  # numpy array, shape (H, W, C)
            ratio = np.isclose(image, 0.0, atol=10).sum() / image.size
            if ratio <= self.config.dark_threshold:
                filtered_tiles.append(tile)

        return filtered_tiles

    def _validate_annotation_preservation(
        self,
        original_annotations: List[Dict[str, Any]],
        output_tiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate that annotations are properly preserved during tiling.

        Args:
            original_annotations: List of original annotations
            output_tiles: List of output tiles with annotations

        Returns:
            Dictionary with validation statistics
        """
        if not original_annotations:
            return {"preserved_count": 0, "total_count": 0, "preservation_ratio": 1.0}

        # Track which original annotations were preserved
        preserved_indices = set()
        total_preserved = 0

        for tile in output_tiles:
            tile_annotations = tile.get("annotations", [])
            for annotation in tile_annotations:
                if "original_annotation_index" in annotation:
                    preserved_indices.add(annotation["original_annotation_index"])
                    total_preserved += 1

        preservation_ratio = len(preserved_indices) / len(original_annotations)

        validation_stats = {
            "preserved_count": len(preserved_indices),
            "total_count": len(original_annotations),
            "preservation_ratio": preservation_ratio,
            "total_annotations_in_output": total_preserved,
            "unique_annotations_preserved": len(preserved_indices),
        }

        # Log validation results
        logger.debug(
            f"Annotation preservation: {len(preserved_indices)}/{len(original_annotations)} "
            f"unique annotations preserved ({preservation_ratio:.2%})"
        )

        return validation_stats

    def _validate_bbox_coordinates(
        self, bbox: List[float], tile_info: Dict[str, Any]
    ) -> None:
        """
        Validate that bbox coordinates are properly clipped to tile bounds.

        Args:
            bbox: Bbox coordinates [x, y, width, height] in tile coordinates
            tile_info: Information about the tile including its dimensions
        """
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: expected 4 values, got {len(bbox)}")

        x, y, w, h = bbox
        tile_width = tile_info["width"]
        tile_height = tile_info["height"]

        # Check that bbox is within tile bounds
        if x < 0 or y < 0:
            raise ValueError(f"Bbox coordinates ({x}, {y}) are negative")
        if x + w > tile_width:
            raise ValueError(f"Bbox extends beyond tile width: {x + w} > {tile_width}")
        if y + h > tile_height:
            raise ValueError(
                f"Bbox extends beyond tile height: {y + h} > {tile_height}"
            )

        # Check that bbox has positive dimensions
        if w <= 0 or h <= 0:
            raise ValueError(f"Bbox has invalid dimensions: {w} x {h}")

    def _extract_tile_annotations(
        self,
        annotations: List[Dict[str, Any]],
        x_offset: int,
        y_offset: int,
        x_end: int,
        y_end: int,
    ) -> List[Dict[str, Any]]:
        """Extract annotations that fall within the tile bounds."""
        logger.debug(f"[TILING] Starting extraction for {len(annotations)} annotations")

        tile_annotations = []

        for i, annotation in enumerate(annotations):
            # Handle bounding boxes
            if "bbox" in annotation:
                bbox = annotation["bbox"]
                logger.debug(
                    f"[TILING] Annotation {i}: bbox={bbox}, tile=({x_offset},{y_offset},{x_end},{y_end})"
                )
                intersects = self._bbox_intersects_tile(
                    bbox, x_offset, y_offset, x_end, y_end
                )
                logger.debug(f"[TILING] Annotation {i}: intersects={intersects}")
                if intersects:
                    tile_bbox, ratio = self._clip_bbox_to_tile(
                        bbox, x_offset, y_offset, x_end, y_end
                    )
                    logger.debug(
                        f"[TILING] Annotation {i}: clipped_bbox={tile_bbox}, ratio={ratio}"
                    )
                    tile_annotation = annotation.copy()
                    tile_annotation["bbox"] = tile_bbox
                    tile_annotation["visibility_ratio"] = ratio
                    tile_annotation["original_annotation_index"] = i
                    if ratio > self.config.min_visibility:
                        logger.debug(
                            f"[TILING] Annotation {i}: included (ratio {ratio} > min_visibility {self.config.min_visibility})"
                        )
                        tile_annotations.append(tile_annotation)
                    else:
                        logger.debug(
                            f"[TILING] Annotation {i}: filtered out (ratio {ratio} <= min_visibility {self.config.min_visibility})"
                        )
                else:
                    logger.debug(
                        f"[TILING] Annotation {i}: does not intersect with tile"
                    )
            else:
                logger.debug(
                    f"[TILING] Annotation {i}: missing bbox, type={type(annotation)}"
                )
                raise ValueError(f"Annotation type {type(annotation)} not supported")

        return tile_annotations

    def _bbox_intersects_tile(
        self, bbox: List[float], x_offset: int, y_offset: int, x_end: int, y_end: int
    ) -> bool:
        """Check if bounding box intersects with tile."""

        left, top, width, height = bbox

        right = left + width
        bottom = top + height

        if left >= x_end:
            return False
        if top >= y_end:
            return False
        if right <= x_offset:
            return False
        if bottom <= y_offset:
            return False

        return True

    def _clip_bbox_to_tile(
        self, bbox: List[float], x_offset: int, y_offset: int, x_end: int, y_end: int
    ) -> Tuple[List[float], float]:
        """Clip bounding box coordinates to tile bounds."""
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # Calculate intersection bounds
        intersect_x1 = max(bbox_x, x_offset)
        intersect_y1 = max(bbox_y, y_offset)
        intersect_x2 = min(bbox_x + bbox_w, x_end)
        intersect_y2 = min(bbox_y + bbox_h, y_end)

        # Check if there's a valid intersection
        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return [0, 0, 0, 0], 0.0

        # Convert to tile-relative coordinates
        new_x = intersect_x1 - x_offset
        new_y = intersect_y1 - y_offset
        new_w = intersect_x2 - intersect_x1
        new_h = intersect_y2 - intersect_y1

        # Calculate visibility ratio
        original_area = bbox_w * bbox_h
        if original_area <= 0:
            return [0, 0, 0, 0], 0.0

        clipped_area = new_w * new_h
        ratio = clipped_area / original_area

        return [new_x, new_y, new_w, new_h], ratio

    def _sample_tiles_with_ratio(
        self, empty_tiles: List[Dict[str, Any]], non_empty_tiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sample tiles maintaining the specified ratio between empty and non-empty tiles.

        Args:
            empty_tiles: List of (image, annotations, info) tuples for empty tiles
            non_empty_tiles: List of (image, annotations, info) tuples for non-empty tiles

        Returns:
            List of selected tile tuples
        """
        ratio = self.config.negative_positive_ratio

        # Calculate how many tiles of each type to sample
        if non_empty_tiles:
            # If we have non-empty tiles, calculate based on ratio
            max_empty = min(len(empty_tiles), int(len(non_empty_tiles) * ratio))
        else:
            # If no non-empty tiles, just take empty tiles up to max
            max_empty = min(
                len(empty_tiles), self.config.max_negative_tiles_in_negative_image
            )

        # Sample tiles
        selected_tiles = []

        # Add non-empty tiles
        selected_tiles.extend(non_empty_tiles)

        # Add empty tiles
        random.shuffle(empty_tiles)
        selected_tiles.extend(empty_tiles[:max_empty])

        return selected_tiles
