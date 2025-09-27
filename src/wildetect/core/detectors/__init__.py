"""
Detector implementations for different model types.
"""
from ..config import DetectionPipelineTypes
from .asynced import AsyncDetectionPipeline
from .base import DetectionPipeline
from .multiprocessed import MultiProcessingDetectionPipeline
from .multithreaded import MultiThreadedDetectionPipeline

__all__ = [
    "DetectionPipeline",
    "MultiProcessingDetectionPipeline",
    "MultiThreadedDetectionPipeline",
    "AsyncDetectionPipeline",
]


def get_detection_pipeline(
    pipeline_type: DetectionPipelineTypes, **kwargs
) -> DetectionPipeline:
    """Get a detection pipeline based on the pipeline type."""
    if pipeline_type == DetectionPipelineTypes.MT:
        return MultiThreadedDetectionPipeline(**kwargs)
    elif pipeline_type == DetectionPipelineTypes.MP:
        return MultiProcessingDetectionPipeline(**kwargs)
    elif pipeline_type == DetectionPipelineTypes.ASYNC:
        return AsyncDetectionPipeline(**kwargs)
    else:
        return DetectionPipeline(**kwargs)
