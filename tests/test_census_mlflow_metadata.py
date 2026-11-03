"""MLflow metadata the census pipeline reads when overriding the loader.

Registration writes ``imgsz``, not ``tilesize``. Census uses ``imgsz`` as the
tile size, and still prefers an explicit ``tilesize`` when that key is present.
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


def test_registered_imgsz_overrides_census_tile_size() -> None:
    """Models already in the registry store input size as imgsz only."""
    registered = ModelMetadata(
        batch=16,
        imgsz=640,
        task=ModelTask.DETECT,
        model_type=ModelType.DETECTOR,
    ).to_dict()
    assert "tilesize" not in registered
    # load_registered_model adds these before merging MLflow metadata.
    downloaded = {
        "version": "detector:1",
        "modeluri": "models:/detector/1",
        "model_path": "/tmp/models-registry/detector/1",
        **registered,
    }

    pipeline = _apply(downloaded)

    assert pipeline.loader_config.batch_size == 16
    assert pipeline.loader_config.tile_size == 640


def test_explicit_tilesize_wins_over_registered_imgsz() -> None:
    """Metadata that already sets tilesize keeps that value."""
    pipeline = _apply({"batch": 2, "tilesize": 512, "imgsz": 800})

    assert pipeline.loader_config.batch_size == 2
    assert pipeline.loader_config.tile_size == 512


def test_missing_size_keys_leave_loader_tile_size() -> None:
    pipeline = _apply({"batch": 2})

    assert pipeline.loader_config.batch_size == 2
    assert pipeline.loader_config.tile_size == 800
