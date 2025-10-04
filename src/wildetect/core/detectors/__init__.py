"""
Detector implementations for different model types.
"""
from ..config import DetectionPipelineTypes
from .asynced import AsyncDetectionPipeline
from .base import DetectionPipeline, SimpleDetectionPipeline
from .multiprocessed import MultiProcessingDetectionPipeline
from .multithreaded import (
    MultiThreadedDetectionPipeline,
    SimpleMultiThreadedDetectionPipeline,
)
from .raster import RasterDetectionPipeline

__all__ = [
    "DetectionPipeline",
    "MultiProcessingDetectionPipeline",
    "MultiThreadedDetectionPipeline",
    "SimpleMultiThreadedDetectionPipeline",
    "AsyncDetectionPipeline",
    "SimpleDetectionPipeline",
    "RasterDetectionPipeline",
]


def get_detection_pipeline(
    pipeline_type: DetectionPipelineTypes, **kwargs
) -> DetectionPipeline:
    """Get a detection pipeline based on the pipeline type."""
    if pipeline_type == DetectionPipelineTypes.MT:
        return MultiThreadedDetectionPipeline(**kwargs)
    elif pipeline_type == DetectionPipelineTypes.MT_SIMPLE:
        return SimpleMultiThreadedDetectionPipeline(**kwargs)
    elif pipeline_type == DetectionPipelineTypes.MP:
        return MultiProcessingDetectionPipeline(**kwargs)
    elif pipeline_type == DetectionPipelineTypes.ASYNC:
        return AsyncDetectionPipeline(**kwargs)
    elif pipeline_type == DetectionPipelineTypes.SIMPLE:
        return SimpleDetectionPipeline(**kwargs)
    else:
        return DetectionPipeline(**kwargs)
