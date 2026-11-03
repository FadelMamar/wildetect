"""Freeze the MLflow metadata contract the census pipeline reads.

Registration writes ModelMetadata keys. After download, load_registered_model
adds version, modeluri, and model_path, then copies MLflow metadata on top.
Census only consults ``batch`` and ``tilesize`` when overriding the loader.
"""

from wildtrain.models.register import ModelMetadata, ModelTask, ModelType

from wildetect.core.config import LoaderConfig
from wildetect.core.detectors.base import BaseDetectionPipeline


class _LoaderState:
    """Stand-in with the two fields override_loading_config reads."""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.loader_config = LoaderConfig(batch_size=4, tile_size=800)


def _apply(metadata: dict) -> _LoaderState:
    state = _LoaderState(metadata)
    BaseDetectionPipeline.override_loading_config(state)
    return state


def test_census_overrides_batch_and_tilesize_only() -> None:
    registered = ModelMetadata(
        batch=16,
        imgsz=640,
        task=ModelTask.DETECT,
        model_type=ModelType.DETECTOR,
    ).to_dict()
    # load_registered_model adds these before merging MLflow metadata.
    downloaded = {
        "version": "detector:1",
        "modeluri": "models:/detector/1",
        "model_path": "/tmp/models-registry/detector/1",
        **registered,
    }

    pipeline = _apply(downloaded)

    assert pipeline.loader_config.batch_size == 16
    # imgsz is not the census tile-size key, so the loader tile size stays put.
    assert pipeline.loader_config.tile_size == 800
    assert "tilesize" not in downloaded


def test_census_tile_override_requires_tilesize_key() -> None:
    pipeline = _apply({"batch": 2, "tilesize": 512, "imgsz": 800})

    assert pipeline.loader_config.batch_size == 2
    assert pipeline.loader_config.tile_size == 512
