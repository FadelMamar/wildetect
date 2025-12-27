"""
Parser for Label Studio JSON exports targeting object detection annotations.

This module provides a high-level parser class for working with Label Studio
annotation exports, with methods for extraction, conversion, and analysis.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import pandas as pd
from .labelstudio_schemas import (
    LabelStudioExport,
    Task,
    ResultOrigin
)
from tqdm import tqdm


@dataclass
class ParsedAnnotation:
    """Flattened annotation data for easy processing.
    
    Combines task, annotation, and result data into a single
    convenient structure for downstream processing.
    """
    # Task info
    task_id: int
    image_path: str
    #image_filename: str
    
    # Image dimensions
    original_width: int
    original_height: int
    
    # Annotation info
    annotation_id: int
    result_id: Optional[str]
    
    # Bounding box (percentage)
    x: float
    y: float
    width: float
    height: float
    rotation: float
    
    # Label info
    label: str
    all_labels: List[str]
    
    # Metadata    
    score: Optional[float]=None
    completed_by: Optional[int]=None
    origin: Optional[ResultOrigin]=None
    is_empty: bool=False
    
    @property
    def bbox_percent(self) -> Tuple[float, float, float, float]:
        """Get bbox as (x, y, width, height) in percentage."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def bbox_pixel(self) -> Tuple[float, float, float, float]:
        """Get bbox as (x, y, width, height) in pixels."""
        return (
            self.x / 100.0 * self.original_width,
            self.y / 100.0 * self.original_height,
            self.width / 100.0 * self.original_width,
            self.height / 100.0 * self.original_height,
        )
    
    @property
    def bbox_normalized(self) -> Tuple[float, float, float, float]:
        """Get bbox as (x, y, width, height) normalized (0-1)."""
        return (
            self.x / 100.0,
            self.y / 100.0,
            self.width / 100.0,
            self.height / 100.0,
        )
    
    @property
    def bbox_coco(self) -> List[float]:
        """Get bbox in COCO format [x, y, width, height] in pixels."""
        return list(self.bbox_pixel)
    
    @property
    def bbox_yolo(self) -> Tuple[float, float, float, float]:
        """Get bbox in YOLO format (cx, cy, w, h) normalized."""
        x_norm, y_norm, w_norm, h_norm = self.bbox_normalized
        return (
            x_norm + w_norm / 2,
            y_norm + h_norm / 2,
            w_norm,
            h_norm,
        )
    
    @property
    def area_pixels(self) -> float:
        """Get bounding box area in pixels."""
        bbox = self.bbox_pixel
        return bbox[2] * bbox[3]
    
    @property
    def is_rotated(self) -> bool:
        """Check if this annotation has rotation."""
        return abs(self.rotation) > 0.001


class LabelStudioParser:
    """High-level parser for Label Studio JSON exports.
    
    Provides methods for loading, extracting, filtering, and converting
    Label Studio annotations to various formats.
    
    Example:
        >>> parser = LabelStudioParser.from_file("annotations.json")
        >>> print(f"Tasks: {parser.task_count}")
        >>> print(f"Labels: {parser.get_label_statistics()}")
        >>> 
        >>> for ann in parser.iter_annotations():
        ...     print(f"{ann.image_filename}: {ann.label} at {ann.bbox_pixel}")
    """
    
    def __init__(self, export_data: LabelStudioExport):
        """Initialize parser with export data.
        
        Args:
            export_data: Parsed LabelStudioExport object
        """
        self.export_data = export_data
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "LabelStudioParser":
        """Load parser from a JSON file.
        
        Args:
            file_path: Path to Label Studio JSON export
            
        Returns:
            Initialized LabelStudioParser
        """
        export = LabelStudioExport.from_file(file_path)
        return cls(export)
    
    @classmethod
    def from_json(cls, json_str: str) -> "LabelStudioParser":
        """Load parser from a JSON string.
        
        Args:
            json_str: JSON string containing task list
            
        Returns:
            Initialized LabelStudioParser
        """
        export = LabelStudioExport.from_json(json_str)
        return cls(export)
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "LabelStudioParser":
        """Load parser from a list of task dictionaries.
        
        Args:
            data: List of task dictionaries
            
        Returns:
            Initialized LabelStudioParser
        """
        export = LabelStudioExport.from_list(data)
        return cls(export)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def tasks(self) -> List[Task]:
        """Get all tasks."""
        return self.export_data.tasks
    
    @property
    def task_count(self) -> int:
        """Get total number of tasks."""
        return self.export_data.task_count
    
    @property
    def annotated_task_count(self) -> int:
        """Get number of tasks with annotations."""
        return self.export_data.annotated_task_count
    
    @property
    def labels(self) -> List[str]:
        """Get all unique labels."""
        return self.export_data.get_all_labels()
    
    # =========================================================================
    # Extraction Methods
    # =========================================================================
    
    def iter_annotations(
        self,
        include_empty: bool = False,
        labels: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        show_progress: bool = True,
    ) -> Iterator[ParsedAnnotation]:
        """Iterate over all annotations as ParsedAnnotation objects.
        
        Args:
            include_empty: If True, yield placeholder for tasks with no annotations
            labels: If provided, only yield annotations with these labels
            min_score: If provided, only yield predictions with score >= min_score
            
        Yields:
            ParsedAnnotation objects for each bounding box
        """
        label_set = set(labels) if labels else None
        iter_tasks = self.tasks
        if show_progress:
            iter_tasks = tqdm(iter_tasks, desc="Processing tasks")
        
        for task in iter_tasks:
            has_results = False
            
            for annotation in task.annotations:
                for result in annotation.result:
                    # Filter by labels
                    if label_set and not any(lbl in label_set for lbl in result.labels):
                        continue
                    
                    # Filter by score
                    if min_score is not None and result.score is not None:
                        if result.score < min_score:
                            continue
                    
                    has_results = True
                    
                    # Create ParsedAnnotation for each label
                    for label in result.labels:
                        if label_set and label not in label_set:
                            continue
                        
                        yield ParsedAnnotation(
                            task_id=task.id,
                            image_path=task.image_path,
                            #image_filename=task.image_filename,
                            original_width=result.original_width,
                            original_height=result.original_height,
                            annotation_id=annotation.id,
                            result_id=result.id,
                            x=result.value.x,
                            y=result.value.y,
                            width=result.value.width,
                            height=result.value.height,
                            rotation=result.value.rotation,
                            label=label,
                            all_labels=result.labels,
                            origin=result.origin,
                            score=result.score,
                            completed_by=annotation.completed_by,
                        )
            
            # Include empty task if requested
            if include_empty and not has_results:
                yield ParsedAnnotation(
                            task_id=task.id,
                            image_path=task.image_path,
                            #image_filename=task.image_filename,
                            original_width=0,
                            original_height=0,
                            annotation_id=None,
                            result_id=None,
                            x=0,
                            y=0,
                            width=0,
                            height=0,
                            rotation=0,
                            label="EMPTY",
                            all_labels=[],
                            score=None,
                            completed_by=None,
                            origin=None,
                            is_empty=True,
                        )
    
    def get_all_annotations(
        self,
        include_empty: bool = True,
        labels: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[ParsedAnnotation]:
        """Get all annotations as a list.
        
        Args:
            include_empty: If True, include tasks with no annotations
            labels: If provided, only include annotations with these labels
            min_score: If provided, only include predictions with score >= min_score
            
        Returns:
            List of ParsedAnnotation objects
        """
        return list(self.iter_annotations(include_empty, labels, min_score))
    
    def get_annotations_for_task(self, task_id: int) -> List[ParsedAnnotation]:
        """Get all annotations for a specific task.
        
        Args:
            task_id: Task ID to filter by
            
        Returns:
            List of ParsedAnnotation objects for the task
        """
        return [ann for ann in self.iter_annotations() if ann.task_id == task_id]
    
    # =========================================================================
    # Statistics Methods
    # =========================================================================
    
    def get_label_statistics(self) -> Dict[str, int]:
        """Get count of annotations per label.
        
        Returns:
            Dictionary mapping label names to counts
        """
        return self.export_data.get_label_counts()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the export.
        
        Returns:
            Dictionary with summary statistics
        """
        annotations = self.get_all_annotations()
        
        return {
            "total_tasks": self.task_count,
            "annotated_tasks": self.annotated_task_count,
            "empty_tasks": self.task_count - self.annotated_task_count,
            "total_annotations": len(annotations),
            "unique_labels": self.labels,
            "label_counts": self.get_label_statistics(),
            "rotated_annotations": sum(1 for a in annotations if a.is_rotated),
        }
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_coco_format(
        self,
        category_mapping: Optional[Dict[str, int]] = None,
        image_id_start: int = 1,
        annotation_id_start: int = 1,
    ) -> Dict[str, Any]:
        """Convert annotations to COCO format.
        
        Args:
            category_mapping: Optional mapping of label names to category IDs.
                            If not provided, IDs are assigned alphabetically.
            image_id_start: Starting ID for images
            annotation_id_start: Starting ID for annotations
            
        Returns:
            Dictionary in COCO format with 'images', 'annotations', 'categories'
        """
        # Build category mapping
        if category_mapping is None:
            labels = sorted(self.labels)
            category_mapping = {label: i + 1 for i, label in enumerate(labels)}
        
        # Build categories
        categories = [
            {"id": cat_id, "name": name}
            for name, cat_id in sorted(category_mapping.items(), key=lambda x: x[1])
        ]
        
        # Build images and annotations
        images = []
        coco_annotations = []
        
        # Track seen image filenames to assign IDs
        image_id_map: Dict[str, int] = {}
        current_image_id = image_id_start
        current_annotation_id = annotation_id_start
        
        for task in self.tasks:
            # Get image dimensions from first result, or use defaults
            first_result = None
            for ann in task.annotations:
                if ann.result:
                    first_result = ann.result[0]
                    break
            
            if first_result:
                img_width = first_result.original_width
                img_height = first_result.original_height
            else:
                img_width = 0
                img_height = 0
            
            # Create image entry if new
            filename = task.image_filename
            if filename not in image_id_map:
                image_id_map[filename] = current_image_id
                images.append({
                    "id": current_image_id,
                    "file_name": filename,
                    "width": img_width,
                    "height": img_height,
                })
                current_image_id += 1
            
            image_id = image_id_map[filename]
            
            # Create annotations
            for annotation in task.annotations:
                for result in annotation.result:
                    bbox = result.get_coco_bbox()
                    area = result.get_coco_area()
                    
                    for label in result.labels:
                        if label not in category_mapping:
                            self.logger.warning(
                                f"Label '{label}' not in category mapping, skipping"
                            )
                            continue
                        
                        coco_annotations.append({
                            "id": current_annotation_id,
                            "image_id": image_id,
                            "category_id": category_mapping[label],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        })
                        current_annotation_id += 1
        
        return {
            "images": images,
            "annotations": coco_annotations,
            "categories": categories,
        }
    
    def to_dataframe(self, include_empty: bool = True) -> pd.DataFrame:
        """Convert annotations to a pandas DataFrame.
        
        Args:
            include_empty: If True, include empty annotations for tasks with no results
        
        Returns:
            DataFrame with one row per annotation
        """
        
        annotations = self.get_all_annotations(include_empty=include_empty)
        
        if not annotations:
            return pd.DataFrame()
        
        records = []
        for ann in annotations:
            bbox_pixel = ann.bbox_pixel
            records.append({
                "task_id": ann.task_id,
                #"image_filename": ann.image_filename,
                "image_path": ann.image_path,
                "annotation_id": ann.annotation_id,
                "result_id": ann.result_id,
                "label": ann.label,
                "x_percent": ann.x,
                "y_percent": ann.y,
                "width_percent": ann.width,
                "height_percent": ann.height,
                "x_pixel": bbox_pixel[0],
                "y_pixel": bbox_pixel[1],
                "width_pixel": bbox_pixel[2],
                "height_pixel": bbox_pixel[3],
                "rotation": ann.rotation,
                "original_width": ann.original_width,
                "original_height": ann.original_height,
                "origin": ann.origin,
                "score": ann.score,
                "completed_by": ann.completed_by,
                "is_empty": ann.is_empty,
            })
        
        return pd.DataFrame(records).convert_dtypes()
    
    # =========================================================================
    # Filtering Methods
    # =========================================================================
    
    def filter_by_labels(self, labels: List[str]) -> "LabelStudioParser":
        """Create a new parser with only specified labels.
        
        Args:
            labels: List of label names to keep
            
        Returns:
            New LabelStudioParser with filtered data
        """
        filtered_export = self.export_data.filter_by_labels(labels)
        return LabelStudioParser(filtered_export)
    
    def filter_tasks(
        self,
        has_annotations: Optional[bool] = None,
        min_annotation_count: Optional[int] = None,
    ) -> "LabelStudioParser":
        """Create a new parser with filtered tasks.
        
        Args:
            has_annotations: If True, only tasks with annotations.
                           If False, only tasks without.
            min_annotation_count: Minimum number of annotations required
            
        Returns:
            New LabelStudioParser with filtered tasks
        """
        filtered_tasks = []
        
        for task in self.tasks:
            # Check has_annotations filter
            if has_annotations is not None:
                if has_annotations and not task.has_annotations:
                    continue
                if not has_annotations and task.has_annotations:
                    continue
            
            # Check min_annotation_count filter
            if min_annotation_count is not None:
                if task.get_annotation_count() < min_annotation_count:
                    continue
            
            filtered_tasks.append(task)
        
        filtered_export = LabelStudioExport(tasks=filtered_tasks)
        return LabelStudioParser(filtered_export)
