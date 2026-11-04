"""Detection pipeline against the current detector modules.

``wildetect.core.detection_pipeline`` is gone. ``DetectionPipeline`` lives in
``wildetect.core.detectors.base`` and the threaded wrapper lives in
``wildetect.core.detectors.multithreaded``. ``_postprocess`` no longer returns
a list; ``_postprocess_batch`` stores ``DroneImage`` objects on the pipeline.
These tests cover that behavior with synthetic images and no MLflow download.
"""

from pathlib import Path

from PIL import Image

from wildetect.core.config import FlightSpecs, LoaderConfig, PredictionConfig
from wildetect.core.data.detection import Detection
from wildetect.core.detectors.base import DetectionPipeline
from wildetect.core.detectors.multithreaded import (
    MultiThreadedDetectionPipeline,
    SimpleMultiThreadedDetectionPipeline,
)


def _configs() -> tuple[PredictionConfig, LoaderConfig]:
    flight_specs = FlightSpecs(
        sensor_height=24.0, focal_length=35.0, flight_height=180.0
    )
    prediction = PredictionConfig(
        mlflow_model_name="detector",
        mlflow_model_alias="demo",
        inference_service_url="http://127.0.0.1:9",
        device="cpu",
        tilesize=32,
        nms_threshold=0.5,
    )
    loader = LoaderConfig(
        tile_size=32,
        overlap=0.0,
        batch_size=1,
        flight_specs=flight_specs,
    )
    return prediction, loader


def _image(path: Path) -> str:
    Image.new("RGB", (64, 48), (20, 40, 60)).save(path)
    return str(path)


def test_postprocess_batch_keeps_the_drone_image(tmp_path: Path) -> None:
    prediction, loader = _configs()
    image = _image(tmp_path / "frame.png")
    pipeline = DetectionPipeline(config=prediction, loader_config=loader)
    detection = Detection(
        bbox=[4, 4, 20, 16],
        confidence=0.8,
        class_id=0,
        class_name="wildlife",
        parent_image=image,
    )
    pipeline._postprocess_batch(
        {
            "tiles": [
                {
                    "image_path": image,
                    "parent_image": image,
                    "x_offset": 0,
                    "y_offset": 0,
                    "width": 32,
                    "height": 32,
                }
            ],
            "detections": [[detection]],
        }
    )
    images = pipeline.get_drone_images()
    assert len(images) == 1
    assert images[0].image_path == image
    assert images[0].tiles


def test_postprocess_batch_ignores_an_empty_batch() -> None:
    prediction, loader = _configs()
    pipeline = DetectionPipeline(config=prediction, loader_config=loader)
    pipeline._postprocess_batch({"tiles": [], "detections": []})
    pipeline._postprocess_batch({"tiles": []})
    assert pipeline.get_drone_images() == []


def test_pipeline_info_reports_the_runtime_state() -> None:
    prediction, loader = _configs()
    pipeline = DetectionPipeline(config=prediction, loader_config=loader)
    info = pipeline.get_pipeline_info()
    assert info["device"] == "cpu"
    assert info["has_detection_system"] is True
    assert info["has_data_loader"] is False
    assert info["total_batches"] == 0
    assert "model_type" not in info


def test_multithreaded_pipeline_stop_and_queue() -> None:
    prediction, loader = _configs()
    pipeline = MultiThreadedDetectionPipeline(config=prediction, loader_config=loader)
    info = pipeline.get_pipeline_info()
    assert info["device"] == "cpu"
    assert info["has_detection_system"] is True
    assert pipeline.data_queue.maxsize == prediction.queue_size
    pipeline.stop_event.set()
    assert pipeline.stop_event.is_set()

    simple = SimpleMultiThreadedDetectionPipeline(
        config=prediction, loader_config=loader
    )
    simple_info = simple.get_pipeline_info()
    assert simple_info["pipeline_type"] == "simple_multithreaded"
    assert simple_info["queue_size"] == prediction.queue_size
    simple.stop()
    assert simple.stop_event.is_set()
