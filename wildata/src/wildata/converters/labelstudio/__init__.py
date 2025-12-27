"""
Label Studio annotation parser module.

This module provides Pydantic models and parsing utilities for Label Studio
JSON exports, specifically targeting object detection schemas.

Example usage:
    from wildata.converters.labelstudio import LabelStudioParser, LabelStudioExport
    
    # Load and parse annotations
    parser = LabelStudioParser.from_file("annotations.json")
    
    # Get statistics
    print(f"Tasks: {parser.task_count}")
    print(f"Labels: {parser.get_label_statistics()}")
    
    # Iterate over results (annotations and/or predictions)
    for result in parser.iter_results(source="both"):
        print(f"{result.image_path}: {result.label} at {result.bbox_pixel}")
    
    # Convert to COCO format
    coco_data = parser.to_coco_format()
"""

from .labelstudio_parser import LabelStudioParser, ParsedResult, ParsedAnnotation, ParsedPrediction
from .labelstudio_schemas import (
    Annotation,
    LabelStudioExport,
    Prediction,
    Result,
    ResultValue,
    Task,
    TaskData,
)
from .labelstudio_converter import LabelstudioConverter

__all__ = [
    # Parser
    "LabelStudioParser",
    "ParsedResult",
    "ParsedAnnotation",  # Alias for backwards compatibility
    "ParsedPrediction",  # Alias for backwards compatibility
    # Schemas
    "LabelStudioExport",
    "Task",
    "TaskData",
    "Annotation",
    "Prediction",
    "Result",
    "ResultValue",
    "LabelstudioConverter",
]
