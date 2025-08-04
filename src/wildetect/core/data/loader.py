"""
Data loader for loading images from directories as tiles.
"""

import logging
import os
import traceback
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from wildetect.core.config import LoaderConfig
from wildetect.core.data.drone_image import DroneImage
from wildetect.core.data.tile import Tile
from wildetect.core.data.utils import TileUtilsv3

logger = logging.getLogger(__name__)


def load_images_as_drone_images(
    image_paths: List[str], flight_specs: Optional[Dict] = None
) -> List[DroneImage]:
    """Load images as DroneImage objects."""
    drone_images = []

    with tqdm(
        total=len(image_paths), desc="Loading drone images", unit="image"
    ) as pbar:
        for image_path in image_paths:
            try:
                drone_image = DroneImage.from_image_path(
                    image_path=image_path, flight_specs=flight_specs
                )
                drone_images.append(drone_image)
            except Exception as e:
                logger.warning(f"Failed to load {image_path}: {e}")
            finally:
                pbar.update(1)

    logger.info(
        f"Loaded {len(drone_images)} drone images from {len(image_paths)} paths"
    )
    return drone_images


def create_drone_image_loader(
    image_paths: List[str], config: Optional[LoaderConfig] = None
) -> "DataLoader":
    """Create a DataLoader for drone images."""
    if config is None:
        config = LoaderConfig()

    return DataLoader(image_paths=image_paths, config=config)


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
        self.image_cache: Dict[str, Optional[torch.Tensor]] = {}

        # Create tiles using dimension-based approach
        self.tiles = self._create_tiles_optimized()

    def _get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions using the fastest possible method."""
        if image_path in self.dimension_cache:
            return self.dimension_cache[image_path]

        try:
            # Use PIL with minimal processing - this is already very fast
            # as it only reads the header, not the pixel data
            with Image.open(image_path) as img:
                width, height = img.size
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

            # Use TileUtilsv3 to get offset info without loading image
            offset_info = TileUtilsv3.get_patches_and_offset_info_only(
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
                "image_path": image_path,
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
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        try:
            with Image.open(image_path) as img:
                # Convert to RGB and then to tensor
                image_tensor = self.pil_to_tensor(img.convert("RGB"))

                # Calculate stride for padding
                stride = int(self.config.tile_size * (1 - self.config.overlap))

                # Pad image to patch size
                padded_image = TileUtilsv3.pad_image_to_patch_size(
                    image_tensor, self.config.tile_size, stride
                )

                self.image_cache[image_path] = padded_image
                return padded_image

        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            self.image_cache[image_path] = None
            return None

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
        image_data = self._load_image_lazy(tile["image_path"])
        if image_data is None:
            raise ValueError(f"Failed to load image data for {tile['image_path']}")

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
                "tile_id": f"{tile['image_path']}_{idx}",
                "image": patch,
                "x_offset": tile["x_offset"],
                "y_offset": tile["y_offset"],
                "width": tile["width"],
                "height": tile["height"],
                "image_path": tile["image_path"],
                "flight_specs": tile.get("flight_specs"),
            }

        except Exception as e:
            logger.error(f"Error during tensor indexing: {e}")
            logger.error(f"Tensor shape: {image_data.shape}")
            logger.error(f"Indices: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
            raise


class DataLoader:
    """DataLoader for handling image tiling with optimized performance."""

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        config: Optional[LoaderConfig] = None,
    ):
        """Initialize the DataLoader."""
        if config is None:
            config = LoaderConfig()

        if image_paths is None and image_dir is None:
            raise ValueError("Either image_paths or image_dir must be provided")

        if image_paths is None:
            if image_dir is None:
                raise ValueError("image_dir cannot be None when image_paths is None")
            image_paths = self._get_image_paths(image_dir)

        self.dataset = TileDataset(image_paths=image_paths, config=config)

        # Create PyTorch DataLoader with Windows-optimized settings
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,  # Keep at 0 for Windows compatibility
            collate_fn=self._collate_fn,
            pin_memory=False,  # Disable pin_memory for better Windows performance
            persistent_workers=False,  # Disable for Windows compatibility
        )

    def _get_image_paths(self, image_dir: str) -> List[str]:
        """Get image paths from directory."""
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist")

        image_extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
        image_paths = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            raise ValueError(f"No image files found in {image_dir}")

        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        return image_paths

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function to handle variable-sized batches."""
        # Extract images from batch
        images = [item["image"] for item in batch]
        # Stack images into a single tensor
        stacked_images = torch.stack(images)

        # Create a new batch dictionary
        collated_batch = {
            "tiles": batch,  # Keep the original batch items
            "images": stacked_images,  # Stacked image tensor
            "batch_size": len(batch),
        }

        return collated_batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.dataloader)

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
