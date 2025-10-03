"""
Data loader for loading images from directories as tiles.
"""

import logging
import math
import os
import traceback
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..config import LoaderConfig
from .dataset import ImageDataset, TileDataset, TiledRaster
from .drone_image import DroneImage

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


class DataLoader:
    """DataLoader for handling image tiling with optimized performance."""

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        config: Optional[LoaderConfig] = None,
        use_tile_dataset: bool = True,
        raster_path: Optional[str] = None,
    ):
        """Initialize the DataLoader."""
        self.config = config or LoaderConfig()
        self.use_tile_dataset = use_tile_dataset

        if (image_paths is None) ^ (image_dir is None) ^ (raster_path is None):
            raise ValueError("Either image_paths or image_dir must be provided")

        if image_paths is None and image_dir is not None:
            image_paths = self._get_image_paths(image_dir)

        if raster_path is not None:
            self.dataset = TiledRaster(
                raster_path,
                patch_size=self.config.tile_size,
                patch_overlap=self.config.overlap,
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=None,
                pin_memory=False,
                persistent_workers=False,
                drop_last=False,
            )

        elif self.use_tile_dataset:
            self.dataset = TileDataset(image_paths=image_paths, config=self.config)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self.dataset._collate_fn,
                pin_memory=False,
                persistent_workers=False,
                drop_last=False,
            )
        else:
            self.dataset = ImageDataset(image_paths=image_paths, config=self.config)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self.dataset._collate_fn,
                pin_memory=False,
                persistent_workers=False,
                drop_last=False,
            )

    def get_offset_info(
        self, image_path: Optional[str] = None, idx: Optional[int] = None
    ) -> Dict:
        """Get offset info for the dataset."""
        assert (image_path is None) ^ (
            idx is None
        ), "One of image_path and idx must be provided"
        return self.dataset.get_offset_info(image_path=image_path, idx=idx)

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

    def __len__(self) -> int:
        """Return the number of batches."""
        if self.use_tile_dataset:
            return math.ceil(len(self.dataset) / self.config.batch_size)
        else:
            return len(self.dataset)

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
