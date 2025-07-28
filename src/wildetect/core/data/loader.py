"""
Data loader for loading images from directories as tiles.
"""

import glob
import logging
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from ..config import LoaderConfig
from .drone_image import DroneImage
from .tile import Tile
from .utils import TileUtilsv2, get_images_paths

logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """Dataset for loading images as tiles."""

    def __init__(
        self,
        config: LoaderConfig,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
    ):
        """Initialize the dataset.

        Args:
            config (LoaderConfig): Loader configuration.
            image_paths (Optional[List[str]]): List of image paths. If None, will use image_dir.
            image_dir (Optional[str]): Directory containing images. Used if image_paths is None.
        """
        self.config = config

        # Handle image paths - prioritize image_paths over image_dir
        if image_paths:
            self.image_paths = image_paths
        elif image_dir:
            self.image_paths = get_images_paths(image_dir)
        else:
            raise ValueError("Either image_paths or image_dir must be provided")

        self.pil_to_tensor = transforms.PILToTensor()

        self.tiles = self._create_tiles()

        logger.info(
            f"Created dataset with {len(self.tiles)} tiles from {len(self.image_paths)} images"
        )

        self.loaded_tiles: dict[str, np.ndarray] = dict()

    def _create_tiles_for_one_image(self, image_path: str) -> Optional[List[Tile]]:
        if not self._validate_tile_parameters(image_path):
            logger.warning(f"Skipping {image_path}: invalid tile parameters")
            return None

        # Get expected tile count for logging
        expected_count = self._get_expected_tile_count(image_path)
        logger.debug(f"Expected {expected_count} tiles for {image_path}")

        # Create base tile
        drone_image = DroneImage.from_image_path(
            image_path=image_path, flight_specs=self.config.flight_specs
        )

        # Extract sub-tiles if image is large enough
        if (
            drone_image.width is not None and drone_image.width > self.config.tile_size
        ) or (
            drone_image.height is not None
            and drone_image.height > self.config.tile_size
        ):
            sub_tiles = self._extract_sub_tiles(drone_image)
            # tiles.extend(sub_tiles)
            logger.debug(f"Created {len(sub_tiles)} sub-tiles from {image_path}")
            return sub_tiles
        else:
            # Use the original image as a single tile
            # tiles.append(drone_image)
            logger.debug(f"Using original image as single tile for {image_path}")
            return [drone_image]

    def _create_tiles(self) -> List[Tile]:
        """Create drone images from all images."""
        tiles = []
        with tqdm(
            total=len(self.image_paths), desc="Creating tiles", unit="image"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self._create_tiles_for_one_image, image_path)
                    for image_path in self.image_paths
                ]
                for future in as_completed(futures):
                    sub_tiles = future.result()
                    if isinstance(sub_tiles, list):
                        tiles.extend(sub_tiles)
                    pbar.update(1)

        logger.info(
            f"Created {len(tiles)} total tiles from {len(self.image_paths)} drone images"
        )
        return tiles

    def _extract_sub_tiles(self, base_tile: DroneImage) -> List[Tile]:
        """Extract sub-tiles from a large image using the TileUtils class."""
        sub_tiles = []

        try:
            # Load image data and convert to tensor
            # image = base_tile.load_image_data()
            width, height = base_tile.width, base_tile.height
            if width is None or height is None:
                raise ValueError(f"Width or height is None for {base_tile.image_path}")

            # image = image.convert("RGB")

            # Convert to tensor
            # image_tensor = self.pil_to_tensor(image)

            # Calculate stride based on overlap
            stride = int(self.config.tile_size * (1 - self.config.overlap))

            # Use TileUtils to extract patches and offset information
            _, offset_info = TileUtilsv2.get_patches_and_offset_info(
                image=torch.zeros(3, height, width),
                patch_size=self.config.tile_size,
                stride=stride,
                channels=3,
                file_name=str(base_tile.image_path),
            )

            # Convert patches tensor to individual PIL images and create tiles
            for i in range(len(offset_info["x_offset"])):
                # Create sub-tile
                sub_tile = Tile(
                    image_data=None,
                    image_path=str(base_tile.image_path),
                    flight_specs=self.config.flight_specs,
                    tile_gps_loc=None,
                    latitude=None,
                    longitude=None,
                    altitude=None,
                    gsd=base_tile.gsd,
                    parent_image=base_tile.image_path,
                )

                # Set offsets after creation
                sub_tile.set_offsets(
                    offset_info["x_offset"][i], offset_info["y_offset"][i]
                )

                sub_tiles.append(sub_tile)

        except Exception as e:
            # logger.warning(f"Failed to extract patches using TileUtils: {e}")
            traceback.print_exc()
            raise Exception(f"Failed to extract patches using TileUtils")

        return sub_tiles

    def _validate_tile_parameters(self, image_path: str) -> bool:
        """Validate tile parameters for an image.

        Args:
            image_path (str): Path to the image to validate.

        Returns:
            bool: True if parameters are valid for tiling.
        """
        try:
            # Load image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Validate parameters using TileUtils
            return TileUtilsv2.validate_patch_parameters(
                image_shape=(3, height, width),  # Assume RGB
                patch_size=self.config.tile_size,
                stride=int(self.config.tile_size * (1 - self.config.overlap)),
            )

        except Exception as e:
            logger.warning(f"Failed to validate tile parameters for {image_path}: {e}")
            return False

    def _get_expected_tile_count(self, image_path: str) -> int:
        """Get the expected number of tiles for an image.

        Args:
            image_path (str): Path to the image.

        Returns:
            int: Expected number of tiles.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size

            stride = int(self.config.tile_size * (1 - self.config.overlap))
            return TileUtilsv2.get_patch_count(
                height, width, self.config.tile_size, stride
            )

        except Exception as e:
            logger.warning(f"Failed to calculate tile count for {image_path}: {e}")
            raise Exception(f"Failed to calculate tile count for {image_path}")

    def __len__(self) -> int:
        return len(self.tiles)

    def _load_patch(self, tile: Tile) -> torch.Tensor:
        """Load a patch from an image."""
        stride = int(self.config.tile_size * (1 - self.config.overlap))

        if tile.image_path not in self.loaded_tiles.keys():
            with Image.open(tile.image_path) as img:
                image = self.pil_to_tensor(img.convert("RGB"))
                image = TileUtilsv2.pad_image_to_patch_size(
                    image, self.config.tile_size, stride
                )
                self.loaded_tiles[tile.image_path] = image
        else:
            image = self.loaded_tiles[tile.image_path]

        y1 = tile.y_offset
        y2 = tile.y_offset + self.config.tile_size
        x1 = tile.x_offset
        x2 = tile.x_offset + self.config.tile_size
        image = image[:, y1:y2, x1:x2]

        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a tile at the specified index."""
        tile = self.tiles[idx]
        patch = self._load_patch(tile)
        patch = patch.float() / 255.0
        return {
            "tile": tile,
            "image": patch,
            "image_path": tile.image_path,
            "tile_id": tile.id,
            "width": tile.width,
            "height": tile.height,
            "x_offset": tile.x_offset,
            "y_offset": tile.y_offset,
            "parent_image": tile.parent_image,
            "gps_loc": tile.tile_gps_loc,
            "geographic_footprint": tile.geographic_footprint,
        }


class DataLoader:
    """Data loader for loading images as tiles."""

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        config: Optional[LoaderConfig] = None,
    ):
        """Initialize the data loader.

        Args:
            image_paths: List of image paths
            image_dir: Optional directory containing images
            config: Loader configuration
        """
        if config is None:
            raise ValueError("LoaderConfig must be provided")

        self.config = config
        self.dataset = TileDataset(config, image_paths=image_paths, image_dir=image_dir)

        # Create PyTorch DataLoader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        logger.info(f"Created DataLoader with {len(self.dataset)} tiles")

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        # Separate tensors and other data
        images = torch.stack([item["image"] for item in batch])
        tiles = [item["tile"] for item in batch]

        # Create batch dictionary
        batch_dict = {
            "images": images,
            "tiles": tiles,
            "image_paths": [item["image_path"] for item in batch],
            "tile_ids": [item["tile_id"] for item in batch],
            "widths": [item["width"] for item in batch],
            "heights": [item["height"] for item in batch],
            "x_offsets": [item["x_offset"] for item in batch],
            "y_offsets": [item["y_offset"] for item in batch],
            "parent_images": [item["parent_image"] for item in batch],
            "gps_locs": [item["gps_loc"] for item in batch],
            "geographic_footprints": [item["geographic_footprint"] for item in batch],
        }

        return batch_dict

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches."""
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)

    def get_tile_by_id(self, tile_id: str) -> Optional[Tile]:
        """Get a specific tile by ID."""
        for tile in self.dataset.tiles:
            if tile.id == tile_id:
                return tile
        return None

    def get_tiles_by_image(self, image_path: str) -> List[Tile]:
        """Get all tiles from a specific image."""
        return [tile for tile in self.dataset.tiles if tile.parent_image == image_path]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_images = len(self.dataset.image_paths)
        total_tiles = len(self.dataset.tiles)

        # Calculate size statistics
        widths = [tile.width for tile in self.dataset.tiles if tile.width is not None]
        heights = [
            tile.height for tile in self.dataset.tiles if tile.height is not None
        ]

        # GPS statistics
        gps_tiles = [
            tile for tile in self.dataset.tiles if tile.tile_gps_loc is not None
        ]
        gps_coverage = len(gps_tiles) / total_tiles if total_tiles > 0 else 0

        return {
            "total_images": total_images,
            "total_tiles": total_tiles,
            "avg_tiles_per_image": total_tiles / total_images
            if total_images > 0
            else 0,
            "width_stats": {
                "min": min(widths) if widths else 0,
                "max": max(widths) if widths else 0,
                "mean": sum(widths) / len(widths) if widths else 0,
            },
            "height_stats": {
                "min": min(heights) if heights else 0,
                "max": max(heights) if heights else 0,
                "mean": sum(heights) / len(heights) if heights else 0,
            },
            "gps_coverage": gps_coverage,
            "gps_tiles": len(gps_tiles),
        }

    def create_drone_images(self) -> List[DroneImage]:
        """Create DroneImage objects from the loaded tiles.

        Returns:
            List[DroneImage]: List of DroneImage objects
        """
        drone_images: Dict[str, DroneImage] = {}

        # Group tiles by parent image
        for tile in self.dataset.tiles:
            parent_image = tile.parent_image or tile.image_path

            if parent_image not in drone_images:
                # Create new DroneImage
                drone_image = DroneImage.from_image_path(
                    image_path=parent_image, flight_specs=self.config.flight_specs
                )
                drone_images[parent_image] = drone_image

            # Add tile to drone image with its offset
            x_offset = tile.x_offset or 0
            y_offset = tile.y_offset or 0
            drone_images[parent_image].add_tile(tile, x_offset, y_offset)

        return list(drone_images.values())

    def get_drone_image_by_path(self, image_path: str) -> Union[DroneImage, None]:
        """Get a specific DroneImage by image path.

        Args:
            image_path (str): Path to the image

        Returns:
            Optional[DroneImage]: DroneImage object or None if not found
        """
        drone_images = self.create_drone_images()
        for drone_image in drone_images:
            if drone_image.image_path == image_path:
                return drone_image
        return None


def load_images_as_drone_images(
    config: LoaderConfig,
    image_paths: List[str],
    image_dir: Optional[str] = None,
    max_images: Optional[int] = None,
    shuffle: bool = False,
) -> List[DroneImage]:
    """Load images as DroneImage objects.

    Args:
        image_dir (str): Directory containing images.
        tile_size (int): Size of tiles to extract.
        overlap (float): Overlap ratio between tiles.
        max_images (Optional[int]): Maximum number of images to load.
        shuffle (bool): Whether to shuffle the drone images.
        **kwargs: Additional configuration options.

    Returns:
        List[DroneImage]: List of DroneImage objects.
    """

    if image_dir is not None:
        image_paths = get_images_paths(image_dir)

    drone_image_list = [
        DroneImage.from_image_path(
            image_path=image_path, flight_specs=config.flight_specs
        )
        for image_path in image_paths
    ]

    # Limit number of images if specified
    if max_images and len(drone_image_list) > max_images:
        drone_image_list = random.sample(drone_image_list, max_images)

    # Shuffle if requested
    if shuffle:
        random.shuffle(drone_image_list)

    return drone_image_list


def create_drone_image_loader(
    config: LoaderConfig,
    image_paths: List[str],
    image_dir: Optional[str] = None,
) -> DataLoader:
    """Create a data loader that returns DroneImage objects.

    Args:
        image_dir (str): Directory containing images.
        tile_size (int): Size of tiles to extract.
        overlap (float): Overlap ratio between tiles.
        batch_size (int): Batch size for loading.
        **kwargs: Additional configuration options.

    Returns:
        DataLoader: Configured data loader for DroneImage objects.
    """
    return DataLoader(image_paths=image_paths, image_dir=image_dir, config=config)
