"""Detection training backends.

The default backend is Ultralytics and does not import the custom fork.
``use_custom_yolo: true`` selects the count/area/FP-TP model.
"""

from typing import Any, Protocol

from ...shared.schemas.detection import DetectionConfig


class DetectionBackend(Protocol):
    """Build the object whose ``train`` method runs detection."""

    name: str

    def build(self, config: DetectionConfig) -> Any:
        """Return a YOLO-like model. The backend owns only construction."""


class UltralyticsBackend:
    """Stock Ultralytics YOLO. This is the product localizer."""

    name = "ultralytics"

    def build(self, config: DetectionConfig) -> Any:
        from ultralytics import YOLO

        model = YOLO(
            config.model.architecture_file or config.model.weights,
            task="detect",
        )
        if config.model.weights is not None:
            model.load(config.model.weights)
        return model


class CustomYoloBackend:
    """Opt-in fork with count, area, and false-positive losses."""

    name = "custom_yolo"

    def build(self, config: DetectionConfig) -> Any:
        from ...trainers.yolo_utils import CustomYOLO

        kwargs = config.custom_yolo_kwargs
        if hasattr(kwargs, "model_dump"):
            kwargs = kwargs.model_dump()
        model = CustomYOLO(
            model=config.model.architecture_file or config.model.weights,
            task="detect",
            **kwargs,
        )
        if config.model.weights is not None:
            model.load(config.model.weights)
        return model


def select_detection_backend(config: DetectionConfig) -> DetectionBackend:
    """Pick the backend named by the detection config."""
    if config.use_custom_yolo:
        return CustomYoloBackend()
    return UltralyticsBackend()
