"""Data loading on synthetic images and a small GeoTIFF."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from wildetect.core.config import FlightSpecs, LoaderConfig
from wildetect.core.data import DataLoader, TileDataset, load_images_as_drone_images


def _images(directory: Path, count: int = 3) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"frame_{index}.png"
        Image.new("RGB", (96, 64), (20 * index, 40, 80)).save(path)
        paths.append(str(path))
    return paths


def _raster(path: Path) -> None:
    data = np.zeros((3, 96, 96), dtype=np.uint8)
    data[0] = 30
    data[1] = 90
    transform = from_origin(30.0, -20.0, 0.0001, 0.0001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=96,
        width=96,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as handle:
        handle.write(data)


def test_loader_config() -> None:
    config = LoaderConfig(tile_size=640, overlap=0.2, batch_size=4, recursive=True)
    assert config.tile_size == 640
    assert config.batch_size == 4
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )
    with_flight = LoaderConfig(tile_size=640, overlap=0.2, flight_specs=flight_specs)
    assert with_flight.flight_specs is not None


def test_image_tile_dataset(tmp_path: Path) -> None:
    image_paths = _images(tmp_path / "images")
    config = LoaderConfig(tile_size=32, overlap=0.0, batch_size=2)
    dataset = TileDataset(config=config, image_paths=image_paths)
    assert len(dataset) > 0
    item = dataset[0]
    assert "tile_id" in item
    assert item["image"].shape[-2:] == (32, 32)


def test_data_loader(tmp_path: Path) -> None:
    image_paths = _images(tmp_path / "images", count=4)
    config = LoaderConfig(tile_size=32, overlap=0.0, batch_size=2, num_workers=0)
    dataloader = DataLoader(image_paths=image_paths, image_dir=None, config=config)
    batches = []
    for batch in dataloader:
        batches.append(batch)
        if len(batches) >= 2:
            break
    assert batches
    assert len(batches[0]["tiles"]) > 0


def test_load_images_as_drone_images(tmp_path: Path) -> None:
    image_paths = _images(tmp_path / "images")
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )
    drone_images = load_images_as_drone_images(
        image_paths=image_paths, flight_specs=flight_specs
    )
    assert len(drone_images) == len(image_paths)
    for drone_image in drone_images:
        assert drone_image.id is not None
        assert len(drone_image.tiles) > 0


def test_ortho_dataset(tmp_path: Path) -> None:
    raster = tmp_path / "mosaic.tif"
    _raster(raster)
    config = LoaderConfig(tile_size=32, overlap=0.0, batch_size=2, num_workers=0)
    loader = DataLoader(raster_path=str(raster), config=config)
    assert len(loader) > 0
    batch = next(iter(loader))
    windows = batch[0]
    assert windows.ndim == 4
    assert windows.shape[1] == 3
