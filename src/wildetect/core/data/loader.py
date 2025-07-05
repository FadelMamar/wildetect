"""
Data loader for loading images from directories as tiles.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Tuple, Union
from dataclasses import dataclass
import glob
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .tile import Tile
from .drone_image import DroneImage
from .utils import TileUtils
from ..config import LoaderConfig


logger = logging.getLogger(__name__)



class ImageTileDataset(Dataset):
    """Dataset for loading images as tiles."""
    
    def __init__(self, config: LoaderConfig):
        """Initialize the dataset.
        
        Args:
            config (LoaderConfig): Loader configuration.
        """
        self.config = config
        self.image_paths = self._get_image_paths()
        self.tiles = self._create_tiles()
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
        logger.info(f"Created dataset with {len(self.tiles)} tiles from {len(self.image_paths)} images")
    
    def _get_image_paths(self) -> List[str]:
        """Get all image paths from the directory."""
        image_dir = Path(self.config.image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        # Build search pattern
        if self.config.recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        image_paths = []
        for ext in self.config.supported_formats:
            pattern_with_ext = str(image_dir / pattern / f"*{ext}")
            image_paths.extend(glob.glob(pattern_with_ext, recursive=self.config.recursive))
            # Also check uppercase extensions
            pattern_with_ext_upper = str(image_dir / pattern / f"*{ext.upper()}")
            image_paths.extend(glob.glob(pattern_with_ext_upper, recursive=self.config.recursive))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        # Filter by size if specified - removed size filtering for now
        pass
        
        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        return image_paths
    
        
    def _create_tiles(self) -> List[Tile]:
        """Create tiles from all images."""
        tiles = []
        
        for image_path in self.image_paths:
            try:
                # Validate tile parameters before processing
                if not self._validate_tile_parameters(image_path):
                    logger.warning(f"Skipping {image_path}: invalid tile parameters")
                    continue
                
                # Get expected tile count for logging
                expected_count = self._get_expected_tile_count(image_path)
                logger.debug(f"Expected {expected_count} tiles for {image_path}")
                
                # Create base tile
                tile = Tile.from_image_path(
                    image_path=image_path,
                    flight_specs=self.config.flight_specs
                )
                
                # Extract sub-tiles if image is large enough
                if (tile.width and tile.width > self.config.tile_size) or (tile.height and tile.height > self.config.tile_size):
                    sub_tiles = self._extract_sub_tiles(tile)
                    tiles.extend(sub_tiles)
                    logger.debug(f"Created {len(sub_tiles)} sub-tiles from {image_path}")
                else:
                    # Use the original image as a single tile
                    tiles.append(tile)
                    logger.debug(f"Using original image as single tile for {image_path}")
                    
            except Exception as e:
                logger.warning(f"Could not create tiles from {image_path}: {e}")
                continue
        
        logger.info(f"Created {len(tiles)} total tiles from {len(self.image_paths)} images")
        return tiles
    
    def _extract_sub_tiles(self, base_tile: Tile) -> List[Tile]:
        """Extract sub-tiles from a large image using the TileUtils class."""
        sub_tiles = []
        
        try:
            # Load image data and convert to tensor
            image = base_tile.load_image_data()
            image = image.convert("RGB")
            
            # Convert to tensor
            from torchvision.transforms import PILToTensor
            image_tensor = PILToTensor()(image)
            
            # Calculate stride based on overlap
            stride = int(self.config.tile_size * (1 - self.config.overlap))
            
            # Use TileUtils to extract patches and offset information
            patches, offset_info = TileUtils.get_patches_and_offset_info(
                image=image_tensor,
                patch_size=self.config.tile_size,
                stride=stride,
                channels=3,
                file_name=str(base_tile.image_path)
            )
            
            # Convert patches tensor to individual PIL images and create tiles
            for i in range(patches.shape[0]):
                # Convert tensor patch back to PIL Image
                patch_tensor = patches[i]
                if patch_tensor.dim() == 3:  # C, H, W
                    patch_tensor = patch_tensor.permute(1, 2, 0)  # H, W, C
                
                # Convert to numpy and then to PIL
                patch_numpy = patch_tensor.cpu().numpy()
                if patch_numpy.max() <= 1.0:  # Normalized
                    patch_numpy = (patch_numpy * 255).astype(np.uint8)
                else:
                    patch_numpy = patch_numpy.astype(np.uint8)
                
                patch_image = Image.fromarray(patch_numpy)
                
                # Create sub-tile
                sub_tile = Tile.from_image_data(
                    image_data=patch_image,
                    flight_specs=self.config.flight_specs
                )
                
                # Set offsets from the offset_info
                x_offset = offset_info["x_offset"][i]
                y_offset = offset_info["y_offset"][i]
                sub_tile.set_offsets(x_offset, y_offset)
                sub_tile.parent_image = base_tile.image_path
                
                # Copy GPS information from parent tile
                sub_tile.tile_gps_loc = base_tile.tile_gps_loc
                sub_tile.latitude = base_tile.latitude
                sub_tile.longitude = base_tile.longitude
                sub_tile.altitude = base_tile.altitude
                
                sub_tiles.append(sub_tile)
                
        except Exception as e:
            logger.warning(f"Failed to extract patches using TileUtils: {e}")
            logger.info("Falling back to manual patch extraction")
            
            # Fallback to manual extraction
            sub_tiles = self._extract_sub_tiles_manual(base_tile)
        
        return sub_tiles
    
    def _extract_sub_tiles_manual(self, base_tile: Tile) -> List[Tile]:
        """Manual fallback method for extracting sub-tiles."""
        sub_tiles = []
        
        # Calculate stride based on overlap
        stride = int(self.config.tile_size * (1 - self.config.overlap))
        
        # Load image data
        image = base_tile.load_image_data()
        
        # Calculate tile positions
        for y in range(0, (base_tile.height or 0) - self.config.tile_size + 1, stride):
            for x in range(0, (base_tile.width or 0) - self.config.tile_size + 1, stride):
                # Crop sub-image
                sub_image = image.crop((x, y, x + self.config.tile_size, y + self.config.tile_size))
                
                # Create sub-tile
                sub_tile = Tile.from_image_data(
                    image_data=sub_image,
                    flight_specs=self.config.flight_specs
                )
                
                # Set offsets for coordinate mapping
                sub_tile.set_offsets(x, y)
                sub_tile.parent_image = base_tile.image_path
                
                # Copy GPS information from parent tile
                sub_tile.tile_gps_loc = base_tile.tile_gps_loc
                sub_tile.latitude = base_tile.latitude
                sub_tile.longitude = base_tile.longitude
                sub_tile.altitude = base_tile.altitude
                
                sub_tiles.append(sub_tile)
        
        return sub_tiles
    
    def _setup_transforms(self) -> Optional[transforms.Compose]:
        """Setup image transforms."""
        transform_list = []
        
        # Resize if specified
        if self.config.resize:
            transform_list.append(transforms.Resize(self.config.resize))
        
        # Convert to tensor
        if self.config.to_tensor:
            transform_list.append(transforms.ToTensor())
        
        # Normalize if specified
        if self.config.normalize and self.config.to_tensor:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transform_list) if transform_list else None
    
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
            return TileUtils.validate_patch_parameters(
                image_shape=(3, height, width),  # Assume RGB
                patch_size=self.config.tile_size,
                stride=int(self.config.tile_size * (1 - self.config.overlap))
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
            return TileUtils.get_patch_count(height, width, self.config.tile_size, stride)
            
        except Exception as e:
            logger.warning(f"Failed to calculate tile count for {image_path}: {e}")
            return 1  # Fallback to single tile
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a tile at the specified index."""
        tile = self.tiles[idx]
        
        # Load image data
        image = tile.load_image_data()
        
        # Apply transforms if specified
        if self.transforms:
            image = self.transforms(image)
        
        return {
            'tile': tile,
            'image': image,
            'image_path': tile.image_path,
            'tile_id': tile.id,
            'width': tile.width,
            'height': tile.height,
            'x_offset': tile.x_offset,
            'y_offset': tile.y_offset,
            'parent_image': tile.parent_image,
            'gps_loc': tile.tile_gps_loc,
            'geographic_footprint': tile.geographic_footprint
        }


class DataLoader:
    """Data loader for loading images as tiles."""
    
    def __init__(self, config: LoaderConfig):
        """Initialize the data loader.
        
        Args:
            config (LoaderConfig): Loader configuration.
        """
        self.config = config
        self.dataset = ImageTileDataset(config)
        
        # Create PyTorch DataLoader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Created DataLoader with {len(self.dataset)} tiles")
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        # Separate tensors and other data
        images = torch.stack([item['image'] for item in batch])
        tiles = [item['tile'] for item in batch]
        
        # Create batch dictionary
        batch_dict = {
            'images': images,
            'tiles': tiles,
            'image_paths': [item['image_path'] for item in batch],
            'tile_ids': [item['tile_id'] for item in batch],
            'widths': [item['width'] for item in batch],
            'heights': [item['height'] for item in batch],
            'x_offsets': [item['x_offset'] for item in batch],
            'y_offsets': [item['y_offset'] for item in batch],
            'parent_images': [item['parent_image'] for item in batch],
            'gps_locs': [item['gps_loc'] for item in batch],
            'geographic_footprints': [item['geographic_footprint'] for item in batch]
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
        heights = [tile.height for tile in self.dataset.tiles if tile.height is not None]
        
        # GPS statistics
        gps_tiles = [tile for tile in self.dataset.tiles if tile.tile_gps_loc is not None]
        gps_coverage = len(gps_tiles) / total_tiles if total_tiles > 0 else 0
        
        return {
            'total_images': total_images,
            'total_tiles': total_tiles,
            'avg_tiles_per_image': total_tiles / total_images if total_images > 0 else 0,
            'width_stats': {
                'min': min(widths) if widths else 0,
                'max': max(widths) if widths else 0,
                'mean': sum(widths) / len(widths) if widths else 0
            },
            'height_stats': {
                'min': min(heights) if heights else 0,
                'max': max(heights) if heights else 0,
                'mean': sum(heights) / len(heights) if heights else 0
            },
            'gps_coverage': gps_coverage,
            'gps_tiles': len(gps_tiles)
        }
    
    def create_drone_images(self) -> List[DroneImage]:
        """Create DroneImage objects from the loaded tiles.
        
        Returns:
            List[DroneImage]: List of DroneImage objects
        """
        drone_images = {}
        
        # Group tiles by parent image
        for tile in self.dataset.tiles:
            parent_image = tile.parent_image or tile.image_path
            
            if parent_image not in drone_images:
                # Create new DroneImage
                drone_image = DroneImage.from_image_path(
                    image_path=parent_image,
                    flight_specs=self.config.flight_specs
                )
                drone_images[parent_image] = drone_image
            
            # Add tile to drone image with its offset
            offset = (tile.x_offset or 0, tile.y_offset or 0)
            drone_images[parent_image].add_tile(tile, offset)
        
        return list(drone_images.values())
    
    def get_drone_image_by_path(self, image_path: str) -> Optional[DroneImage]:
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


def create_loader(
    image_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    batch_size: int = 1,
    **kwargs
) -> DataLoader:
    """Convenience function to create a data loader.
    
    Args:
        image_dir (str): Directory containing images.
        tile_size (int): Size of tiles to extract.
        overlap (float): Overlap ratio between tiles.
        batch_size (int): Batch size for loading.
        **kwargs: Additional configuration options.
        
    Returns:
        DataLoader: Configured data loader.
    """
    config = LoaderConfig(
        image_dir=image_dir,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        **kwargs
    )
    
    return DataLoader(config)


def load_images_as_tiles(
    image_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    max_images: Optional[int] = None,
    shuffle: bool = False,
    **kwargs
) -> List[Tile]:
    """Load images as tiles without batching.
    
    Args:
        image_dir (str): Directory containing images.
        tile_size (int): Size of tiles to extract.
        overlap (float): Overlap ratio between tiles.
        max_images (Optional[int]): Maximum number of images to load.
        shuffle (bool): Whether to shuffle the tiles.
        **kwargs: Additional configuration options.
        
    Returns:
        List[Tile]: List of tiles.
    """
    config = LoaderConfig(
        image_dir=image_dir,
        tile_size=tile_size,
        overlap=overlap,
        **kwargs
    )
    
    dataset = ImageTileDataset(config)
    tiles = dataset.tiles
    
    # Limit number of images if specified
    if max_images:
        image_paths = list(set(tile.parent_image for tile in tiles))
        if len(image_paths) > max_images:
            selected_paths = random.sample(image_paths, max_images)
            tiles = [tile for tile in tiles if tile.parent_image in selected_paths]
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(tiles)
    
    return tiles


def load_images_as_drone_images(
    image_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    max_images: Optional[int] = None,
    shuffle: bool = False,
    **kwargs
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
    config = LoaderConfig(
        image_dir=image_dir,
        tile_size=tile_size,
        overlap=overlap,
        **kwargs
    )
    
    dataset = ImageTileDataset(config)
    
    # Create DroneImage objects
    drone_images = {}
    for tile in dataset.tiles:
        parent_image = tile.parent_image or tile.image_path
        
        if parent_image not in drone_images:
            drone_image = DroneImage.from_image_path(
                image_path=parent_image,
                flight_specs=config.flight_specs
            )
            drone_images[parent_image] = drone_image
        
        # Add tile to drone image with its offset
        offset = (tile.x_offset or 0, tile.y_offset or 0)
        drone_images[parent_image].add_tile(tile, offset)
    
    drone_image_list = list(drone_images.values())
    
    # Limit number of images if specified
    if max_images and len(drone_image_list) > max_images:
        drone_image_list = random.sample(drone_image_list, max_images)
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(drone_image_list)
    
    return drone_image_list


def create_drone_image_loader(
    image_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    batch_size: int = 1,
    **kwargs
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
    config = LoaderConfig(
        image_dir=image_dir,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        **kwargs
    )
    
    return DataLoader(config)
