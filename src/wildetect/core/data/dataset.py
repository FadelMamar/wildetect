"""
Data loader for loading images from directories as tiles.
"""

import logging
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor

from ..config import LoaderConfig
from .utils import TileUtils, read_image

logger = logging.getLogger(__name__)


class SimpleMemoryLRUCache:
    """True LRU cache that evicts least recently used items when capacity is exceeded."""

    def __init__(self, max_items: int = 5) -> None:
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
        self.dimension_groups: Dict[Tuple[int, int], List[str]] = {}

        # Image cache for actual pixel data (only loaded when needed)
        self.image_cache = SimpleMemoryLRUCache()

        # Create tiles using dimension-based approach
        self.tiles = self._create_tiles_optimized()

    def _get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions using the fastest possible method."""
        if image_path in self.dimension_cache:
            return self.dimension_cache[image_path]

        try:
            # Use PIL with minimal processing - this is already very fast
            # as it only reads the header, not the pixel data
            image = read_image(image_path)
            width, height = image.size
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
        for i, image_path in enumerate(self.image_paths):
            dimensions = self._get_image_dimensions(image_path)
            if dimensions:
                width, height = dimensions
                if (width, height) not in self.dimension_groups:
                    self.dimension_groups[(width, height)] = []
                self.dimension_groups[(width, height)].append(image_path)
            else:
                logger.warning(f"Failed to get dimensions for {image_path}")

        # Second pass: create tiles for each dimension group (optimized)
        tiles = []
        for (width, height), image_paths in self.dimension_groups.items():
            # Calculate offsets once for this dimension
            offset_info = self._calculate_offsets_for_dimension(width, height)
            if not offset_info:
                logger.warning(
                    f"Skipping dimension {width}x{height}: failed to calculate offsets"
                )
                continue

            # Create tiles for all images with this dimension (optimized)
            for image_path in image_paths:
                try:
                    # Create tiles directly without DroneImage object
                    sub_tiles = self._create_tiles_for_image_path(
                        image_path, width, height, offset_info
                    )
                    if sub_tiles:
                        tiles.extend(sub_tiles)
                    else:
                        logger.warning(f"No tiles created for {image_path}")

                except Exception as e:
                    logger.error(f"Failed to create tiles for {image_path}: {e}")
                    import traceback

                    traceback.print_exc()

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

        # Create tiles using list comprehension for efficiency
        tiles = [
            {
                "image_path": None,
                "parent_image": image_path,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": self.config.tile_size,
                "height": self.config.tile_size,
                "flight_specs": self.config.flight_specs,
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
        # self.offset_info_records[image_path] = offset_info
        return patches, idx

    def _collate_fn(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Custom collate function to handle variable-sized batches."""
        # batch = torch.stack(batch)
        # batch.squeeze_(0)
        assert len(batch) == 1, "Batch must contain only one tensor"
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
            image = read_image(image_path)
            width, height = image.size
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
