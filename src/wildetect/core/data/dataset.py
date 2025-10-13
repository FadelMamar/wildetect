"""
Data loader for loading images from directories as tiles.
"""

import logging
import os
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import rasterio as rio
import slidingwindow
import torch
from rasterio.windows import Window
from rasterio.warp import transform
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from ..config import LoaderConfig
from .utils import TileUtils, get_image_dimensions, read_image

logger = logging.getLogger(__name__)


class SimpleMemoryLRUCache:
    """True LRU cache that evicts least recently used items when capacity is exceeded."""

    def __init__(self, max_items: int = 2) -> None:
        self.cache = OrderedDict()
        self.max_items = max_items
        self.current_memory = 0

    def get(self, key: str):
        """Get value from cache and mark as recently used."""
        if key in self.cache:
            # Move to end (mark as recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def _get_tensor_memory(self, tensor):
        """Get memory usage of a tensor in bytes."""
        return tensor.element_size() * tensor.nelement()

    def put(self, key: str, tensor: Optional[torch.Tensor]):
        """Add item to cache and evict least recently used items if needed."""
        # If key already exists, remove it first
        if key in self.cache:
            old_tensor = self.cache.pop(key)
            if old_tensor is not None:
                self.current_memory -= self._get_tensor_memory(old_tensor)

        # Add new tensor (or None for failed loads)
        self.cache[key] = tensor
        if tensor is not None:
            self.current_memory += self._get_tensor_memory(tensor)

        # Evict least recently used items if over capacity
        self._evict()

    def _evict(self):
        """Evict least recently used items until we're at capacity."""
        while len(self.cache) > self.max_items:
            # Remove oldest (least recently used) item
            oldest_key, oldest_tensor = self.cache.popitem(last=False)
            if oldest_tensor is not None:
                self.current_memory -= self._get_tensor_memory(oldest_tensor)
            logger.debug(f"LRU evicted item: {oldest_key}")

    def memory_usage_mb(self):
        """Return current memory usage in MB."""
        return self.current_memory / (1024**2)

    def cache_info(self):
        """Return cache statistics."""
        return {
            "size": len(self.cache),
            "memory_mb": self.memory_usage_mb(),
            "max_items": self.max_items,
            "keys": list(self.cache.keys()),
        }


class TileDataset(Dataset):
    """Dataset for handling image tiling with optimized dimension-based offset calculation."""

    def __init__(
        self,
        image_paths: List[str],
        config: LoaderConfig,
        cache_size: int = 2,
    ):
        """Initialize the dataset with dimension-based optimization."""
        self.image_paths = image_paths
        self.config = config
        self.pil_to_tensor = PILToTensor()

        # Dimension cache: maps image_path -> (width, height) or None
        self.dimension_cache: Dict[str, Optional[Tuple[int, int]]] = {}

        # Offset cache: maps (width, height) -> offset_info
        self.offset_cache: Dict[Tuple[int, int], Dict] = {}

        # Group images by dimensions for efficient processing
        self.dimension_groups: Dict[Tuple[int, int], List[str]] = defaultdict(list)

        # Image cache for actual pixel data (only loaded when needed)
        self.image_cache = SimpleMemoryLRUCache(max_items=cache_size)

        # Create tiles using dimension-based approach
        self.tiles = self._create_tiles_optimized()

    def _get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions."""
        if image_path in self.dimension_cache:
            return self.dimension_cache[image_path]

        try:
            # Use PIL with minimal processing - this is already very fast
            # as it only reads the header, not the pixel data
            width, height = get_image_dimensions(image_path)
            self.dimension_cache[image_path] = (width, height)
            return (width, height)
        except Exception as e:
            logger.warning(f"Failed to get dimensions for {image_path}: {e}")
            self.dimension_cache[image_path] = None
            return None

    def _calculate_offsets_for_dimension(
        self, width: int, height: int
    ) -> Optional[Dict]:
        """Calculate tile offsets for a given dimension, using cache if available."""
        dimension_key = (width, height)

        if dimension_key in self.offset_cache:
            return self.offset_cache[dimension_key]

        try:
            # Calculate stride
            stride = int(self.config.tile_size * (1 - self.config.overlap))

            # Use TileUtils to get offset info without loading image
            offset_info = TileUtils.get_offset_info(
                width=width,
                height=height,
                patch_size=self.config.tile_size,
                stride=stride,
            )

            self.offset_cache[dimension_key] = offset_info
            return offset_info

        except Exception as e:
            logger.warning(f"Failed to calculate offsets for {width}x{height}: {e}")
            return None

    def _create_tiles_optimized(self) -> List[Dict]:
        """Create tiles using dimension-based optimization."""

        # Ultra-fast dimension collection
        progress_bar = tqdm(
            total=len(self.image_paths), desc="Fetching image dimensions"
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            for image_path, dimensions in zip(
                self.image_paths,
                executor.map(self._get_image_dimensions, self.image_paths),
            ):
                if dimensions:
                    self.dimension_groups[dimensions].append(image_path)
                else:
                    logger.warning(f"Failed to get dimensions for {image_path}")
                progress_bar.update(1)
        progress_bar.close()

        # Second pass: create tiles for each dimension group (optimized)
        tiles = []
        for (width, height), image_paths in tqdm(
            self.dimension_groups.items(),
            desc="Processing dimension groups",
            total=len(self.dimension_groups),
        ):
            # Calculate offsets once for this dimension
            offset_info = self._calculate_offsets_for_dimension(width, height)
            if not offset_info:
                logger.warning(
                    f"Skipping dimension {width}x{height}: failed to calculate offsets"
                )
                continue

            # Batch create tiles for all images with this dimension
            dimension_tiles = []
            for image_path in image_paths:
                try:
                    # Create tiles directly without DroneImage object
                    sub_tiles = self._create_tiles_for_image_path(
                        image_path, width, height, offset_info
                    )
                    if sub_tiles:
                        dimension_tiles.extend(sub_tiles)
                    else:
                        logger.warning(f"No tiles created for {image_path}")

                except Exception as e:
                    logger.error(f"Failed to create tiles for {image_path}: {e}")
                    traceback.print_exc()

            tiles.extend(dimension_tiles)
            logger.debug(
                f"Created {len(dimension_tiles)} tiles for dimension {width}x{height} "
                f"({len(image_paths)} images)"
            )

        logger.info(f"Created {len(tiles)} total tiles")
        return tiles

    def _create_tiles_for_image_path(
        self, image_path: str, width: int, height: int, offset_info: Dict
    ) -> List[Dict]:
        """Create tiles for an image path using pre-calculated offset information."""
        # Get offset lists
        x_offsets = offset_info.get("x_offset", [])
        y_offsets = offset_info.get("y_offset", [])

        if not x_offsets or not y_offsets:
            logger.warning(f"No offsets found for dimension {width}x{height}")
            return []

        # Pre-compute constant values
        tile_size = self.config.tile_size
        flight_specs = self.config.flight_specs

        # Create tiles using list comprehension with pre-computed constants
        tiles = [
            {
                "image_path": None,
                "parent_image": image_path,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": tile_size,
                "height": tile_size,
                "flight_specs": flight_specs,
            }
            for x_offset, y_offset in zip(x_offsets, y_offsets)
        ]

        return tiles

    def _load_image_lazy(self, image_path: str) -> Optional[torch.Tensor]:
        """Load image data only when needed."""
        cached_image = self.image_cache.get(image_path)
        if cached_image is not None:
            return cached_image

        try:
            image = read_image(image_path)
            # Convert to RGB and then to tensor
            image_tensor = self.pil_to_tensor(image.convert("RGB"))
            image_tensor = image_tensor / 255.0  # normalize to [0,1]

            # Calculate stride for padding
            stride = int(self.config.tile_size * (1 - self.config.overlap))

            # Pad image to patch size
            padded_image = TileUtils.pad_image_to_patch_size(
                image_tensor, self.config.tile_size, stride
            )

            self.image_cache.put(image_path, padded_image)
            return padded_image

        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {traceback.format_exc()}")
            self.image_cache.put(image_path, None)
            return None

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function to handle variable-sized batches."""
        # Extract images from batch
        images = [item.pop("image") for item in batch]
        # Stack images into a single tensor
        stacked_images = torch.stack(images)

        # Create a new batch dictionary
        collated_batch = {
            "tiles": batch,  # Keep the original batch items
            "images": stacked_images,  # Stacked image tensor
            # "batch_size": len(batch),
        }

        return collated_batch

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict:
        """Get a tile at the specified index."""
        if idx >= len(self.tiles):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.tiles)}"
            )

        tile = self.tiles[idx]

        # Load image data only when needed
        image_data = self._load_image_lazy(tile["parent_image"])
        if image_data is None:
            raise ValueError(f"Failed to load image data for {tile['parent_image']}")

        # Extract the specific tile region
        y1 = tile["y_offset"]
        y2 = tile["y_offset"] + self.config.tile_size
        x1 = tile["x_offset"]
        x2 = tile["x_offset"] + self.config.tile_size

        try:
            # Handle different tensor shapes
            if len(image_data.shape) == 3:
                # 3D tensor (C, H, W) - standard format
                patch = image_data[:, y1:y2, x1:x2]
            elif len(image_data.shape) == 2:
                # 2D tensor (H, W) - need to add channel dimension
                patch = image_data[y1:y2, x1:x2].unsqueeze(0)
            else:
                logger.error(f"Unexpected tensor shape: {image_data.shape}")
                raise ValueError(f"Unexpected tensor shape: {image_data.shape}")

            # Return dictionary with tile information
            return {
                "tile_id": f"{tile['parent_image']}_{idx}",
                "image": patch,
                "x_offset": tile["x_offset"],
                "y_offset": tile["y_offset"],
                "width": tile["width"],
                "height": tile["height"],
                "image_path": tile["image_path"],
                "flight_specs": tile.get("flight_specs"),
                "parent_image": tile["parent_image"],
            }

        except Exception as e:
            logger.error(f"Error during tensor indexing: {e}")
            logger.error(f"Tensor shape: {image_data.shape}")
            logger.error(f"Indices: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
            raise

    def get_offset_info(self, *args, **kwargs) -> Optional[Dict]:
        """Get offset info for an image path."""
        raise NotImplementedError("Offset info is not available for TileDataset")


class ImageDataset(Dataset):
    """Dataset for handling image tiling with optimized dimension-based offset calculation."""

    def __init__(self, image_paths: List[str], config: LoaderConfig):
        self.image_paths = image_paths
        self.tile_size = config.tile_size
        self.overlap = config.overlap
        self.pil_to_tensor = PILToTensor()
        self.tiler = TileUtils()
        self.offset_info_records: Dict[str, Dict] = {}
        self.stride = int(self.tile_size * (1 - self.overlap))

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # load image
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        image = self.pil_to_tensor(image.convert("RGB"))
        image = image / 255.0  # normalize to [0,1]

        # get patches and offset info
        patches, _ = self.tiler.get_patches_and_offset_info(
            image=image,
            patch_size=self.tile_size,
            stride=self.stride,
            validate=False,
            file_name=image_path,
        )
        return patches, idx

    def _collate_fn(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Custom collate function to handle variable-sized batches."""
        assert len(batch) == 1, f"Batch must contain only one tensor, got {len(batch)}"
        return batch.pop()

    def get_offset_info(
        self, image_path: Union[str, None], idx: Optional[int] = None
    ) -> Dict:
        """Get offset info for an image path."""
        assert (image_path is None) ^ (
            idx is None
        ), "image_path and idx  cannot be provided together"
        if isinstance(idx, int):
            image_path = self.image_paths[idx]
        try:
            width, height = get_image_dimensions(image_path)
            return self.tiler.get_offset_info(
                width=width,
                height=height,
                patch_size=self.tile_size,
                stride=self.stride,
                file_name=image_path,
            )
        except KeyError as e:
            logger.error(f"Offset info not found or not loaded for {image_path}")
            raise e
        except Exception as e:
            logger.error(
                f"Error getting offset info for path:{image_path} or idx:{idx}"
            )
            raise e


class RasterDataset(Dataset):
    """Dataset for predicting on raster windows.

    This dataset is useful for predicting on a large raster that is too large to fit into memory.

    Args:
        path (str): Path to raster file
        patch_size (int): Size of windows to predict on
        patch_overlap (float): Overlap between windows as fraction (0-1)
    Returns:
        A dataset of raster windows
    """

    def __init__(self, path, patch_size, patch_overlap):
        self.path = path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        self._windows_list = None
        self.longitudes = None
        self.latitudes = None

        self.src = None
        self.prepare_items()

        if path is None:
            raise ValueError("path is required for a memory raster dataset")
    
    def prepare_items(self):
        # Get raster shape without keeping file open
        src = rio.open(self.path)
        
        width = src.shape[0]
        height = src.shape[1]            

        # Generate sliding windows
        self.windows = slidingwindow.generateForSize(
            height,
            width,
            dimOrder=slidingwindow.DimOrder.ChannelHeightWidth,
            maxWindowSize=self.patch_size,
            overlapPercent=self.patch_overlap,
        )

        xs = []
        ys = []
        for win in self.windows:
            xs.append(int(win.x + win.w/2))
            ys.append(int(win.y + win.h/2))
        xs, ys = src.xy(ys, xs)
        self.longitudes, self.latitudes = transform(src_crs=src.crs, dst_crs='EPSG:4326', xs=xs, ys=ys)

        src.close()

        return
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get the item at the given index."""
        if self.src is None:
            self.src = rio.open(self.path)

        window_data, lon, lat = self.get_crop(idx)

        if idx == (len(self.windows) - 1):
            self.src.close()
            self.src = None

        return window_data, torch.tensor(self.get_crop_bounds(idx)), torch.tensor([lon, lat])

    def window_list(self):
        if self._windows_list is None:
            self._windows_list = [x.getRect() for x in self.windows]
        return self._windows_list
    
    def get_crop(self, idx):        
        window = self.windows[idx]
        # Read only first 3 channels (RGB) if there are more channels
        num_channels = min(self.src.count, 3)
        window_data = self.src.read(
            indexes=list(
                range(1, num_channels + 1)
            ),  # rasterio uses 1-based indexing
            window=Window(window.x, window.y, window.w, window.h),
        )
        # get gps coordinates in WGS84
        #lon, lat = self.src.xy(int(window.y + window.h/2), int(window.x + window.w/2))
        #lon, lat = transform(src_crs=self.src.crs, dst_crs='EPSG:4326', xs=[lon], ys=[lat])
        #lon, lat = lon[0], lat[0]
        lon, lat = self.longitudes[idx], self.latitudes[idx]

        # Convert to torch tensor and rearrange dimensions
        window_data = torch.from_numpy(window_data).float()  # Convert to torch tensor
        window_data = window_data / 255.0  # Normalize        

        return window_data, lon, lat

    def get_image_basename(self, idx):
        return os.path.basename(self.path)

    def get_crop_bounds(self, idx: int):
        window = self.windows[idx]
        return [window.x, window.y, window.w, window.h]


class TiledRasterDataset(TileDataset):

    def __init__(self, raster_paths: List[str], config: LoaderConfig, cache_size: int = 1):
        super().__init__(image_paths=raster_paths, config=config, cache_size=cache_size)

    def _get_raster_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions."""
        with rio.open(image_path) as src:
            width = src.shape[0]
            height = src.shape[1]
            return width, height
    
    def _get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions."""
        if image_path in self.dimension_cache:
            return self.dimension_cache[image_path]

        try:
            width, height = self._get_raster_dimensions(image_path)
            self.dimension_cache[image_path] = (width, height)
            return (width, height)
        except Exception as e:
            logger.warning(f"Failed to get dimensions for {image_path}: {e}")
            self.dimension_cache[image_path] = None
            return None

    def _load_image_lazy(self, image_path: str) -> Optional[torch.Tensor]:
        """Load image data only when needed."""
        cached_image = self.image_cache.get(image_path)
        if cached_image is not None:
            return cached_image

        try:
            with rio.open(image_path) as src:
                image = src.read()
            image = image.transpose(1,2,0)                # convert to CHW
            image = torch.from_numpy(image).float() / 255.0  # normalize to [0,1]

            if image.shape[2] > 3:
                image = image[:,:,:3]            

            # Calculate stride for padding
            stride = int(self.config.tile_size * (1 - self.config.overlap))

            # Pad image to patch size
            padded_image = TileUtils.pad_image_to_patch_size(
                image, self.config.tile_size, stride
            )

            self.image_cache.put(image_path, padded_image)
            return padded_image

        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {traceback.format_exc()}")
            self.image_cache.put(image_path, None)
            return None