"""
Pydantic models for Label Studio JSON export format.

This module provides type-safe models for parsing Label Studio annotation exports,
specifically targeting object detection schemas with rectangle labels.

Label Studio coordinate system:
- All bounding box values (x, y, width, height) are in percentage units (0-100)
- Origin is top-left corner
- Rotation is in degrees, clockwise
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote
from label_studio_tools.core.utils.io import get_local_path

from pydantic import BaseModel, Field, field_validator


class ResultValue(BaseModel):
    """Bounding box value for rectangle labels.
    
    All coordinates are in percentage units (0-100) relative to image dimensions.
    """
    x: float = Field(..., description="X position as percentage (0-100)")
    y: float = Field(..., description="Y position as percentage (0-100)")
    width: float = Field(..., description="Width as percentage (0-100)")
    height: float = Field(..., description="Height as percentage (0-100)")
    rotation: float = Field(default=0.0, description="Rotation in degrees (clockwise)")
    rectanglelabels: List[str] = Field(default_factory=list, description="Class labels")

    def to_pixel_bbox(
        self, 
        original_width: int, 
        original_height: int
    ) -> Tuple[float, float, float, float]:
        """Convert percentage bbox to pixel coordinates.
        
        Args:
            original_width: Image width in pixels
            original_height: Image height in pixels
            
        Returns:
            Tuple of (x, y, width, height) in pixels
        """
        px = self.x / 100.0 * original_width
        py = self.y / 100.0 * original_height
        pw = self.width / 100.0 * original_width
        ph = self.height / 100.0 * original_height
        return (px, py, pw, ph)
    
    def to_normalized_bbox(self) -> Tuple[float, float, float, float]:
        """Convert percentage bbox to normalized (0-1) coordinates.
        
        Returns:
            Tuple of (x, y, width, height) in normalized coordinates
        """
        return (
            self.x / 100.0, 
            self.y / 100.0, 
            self.width / 100.0, 
            self.height / 100.0
        )
    
    def to_coco_bbox(
        self,
        original_width: int,
        original_height: int
    ) -> List[float]:
        """Convert to COCO format [x, y, width, height] in pixels.
        
        Args:
            original_width: Image width in pixels
            original_height: Image height in pixels
            
        Returns:
            List of [x, y, width, height] in pixels
        """
        px, py, pw, ph = self.to_pixel_bbox(original_width, original_height)
        return [px, py, pw, ph]
    
    def to_yolo_bbox(self) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (center_x, center_y, width, height) normalized.
        
        Returns:
            Tuple of (cx, cy, w, h) in normalized coordinates (0-1)
        """
        # Convert to normalized
        x_norm = self.x / 100.0
        y_norm = self.y / 100.0
        w_norm = self.width / 100.0
        h_norm = self.height / 100.0
        
        # Convert to center coordinates
        cx = x_norm + w_norm / 2
        cy = y_norm + h_norm / 2
        
        return (cx, cy, w_norm, h_norm)
    
    def clip_to_bounds(self) -> "ResultValue":
        """Return a new ResultValue with coordinates clipped to [0, 100].
        
        Returns:
            New ResultValue with clipped coordinates
        """
        return ResultValue(
            x=max(0.0, min(100.0, self.x)),
            y=max(0.0, min(100.0, self.y)),
            width=max(0.0, min(100.0 - max(0.0, self.x), self.width)),
            height=max(0.0, min(100.0 - max(0.0, self.y), self.height)),
            rotation=self.rotation,
            rectanglelabels=self.rectanglelabels.copy()
        )
    
    @property
    def label(self) -> Optional[str]:
        """Get the primary label (first in the list)."""
        return self.rectanglelabels[0] if self.rectanglelabels else None
    
    @property
    def is_rotated(self) -> bool:
        """Check if this box has rotation."""
        return abs(self.rotation) > 0.001


class Result(BaseModel):
    """Single annotation result (one bounding box).
    
    Represents a single object detection annotation with its bounding box,
    label, and metadata.
    """
    id: Optional[str] = Field(default=None, description="Unique result ID")
    type: str = Field(..., description="Result type (e.g., 'rectanglelabels')")
    from_name: str = Field(..., description="Label control name in LS config")
    to_name: str = Field(..., description="Image control name in LS config")
    original_width: int = Field(..., description="Original image width in pixels")
    original_height: int = Field(..., description="Original image height in pixels")
    image_rotation: int = Field(default=0, description="Image rotation applied")
    value: ResultValue = Field(..., description="Bounding box value")
    origin: Optional[str] = Field(
        default=None, 
        description="Origin: 'manual', 'prediction', or 'prediction-changed'"
    )
    score: Optional[float] = Field(
        default=None, 
        description="Confidence score for predictions"
    )

    @property
    def label(self) -> Optional[str]:
        """Get the primary label."""
        return self.value.label
    
    @property
    def labels(self) -> List[str]:
        """Get all labels."""
        return self.value.rectanglelabels
    
    def get_pixel_bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box in pixel coordinates."""
        return self.value.to_pixel_bbox(self.original_width, self.original_height)
    
    def get_coco_bbox(self) -> List[float]:
        """Get bounding box in COCO format [x, y, width, height] pixels."""
        return self.value.to_coco_bbox(self.original_width, self.original_height)
    
    def get_coco_area(self) -> float:
        """Calculate area in pixels for COCO format."""
        bbox = self.get_pixel_bbox()
        return bbox[2] * bbox[3]


class Prediction(BaseModel):
    """Model prediction with results.
    
    Contains predictions from ML models that were shown to annotators.
    """
    id: Optional[int] = Field(default=None, description="Prediction ID")
    result: List[Result] = Field(default_factory=list, description="Prediction results")
    model_version: Optional[str] = Field(default=None, description="Model version string")
    created_ago: Optional[str] = Field(default=None, description="Human-readable creation time")
    score: Optional[float] = Field(default=None, description="Overall prediction score")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")
    task: Optional[int] = Field(default=None, description="Associated task ID")
    project: Optional[int] = Field(default=None, description="Associated project ID")

    class Config:
        extra = "ignore"  # Ignore additional fields


class Annotation(BaseModel):
    """Human annotation for a task.
    
    Contains the results of human annotation, including all bounding boxes
    and metadata about when/how the annotation was created.
    """
    id: int = Field(..., description="Annotation ID")
    completed_by: Optional[int] = Field(default=None, description="User ID who completed")
    result: List[Result] = Field(default_factory=list, description="Annotation results")
    was_cancelled: bool = Field(default=False, description="Was annotation cancelled")
    ground_truth: bool = Field(default=False, description="Is ground truth annotation")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")
    draft_created_at: Optional[datetime] = Field(default=None, description="Draft creation timestamp")
    lead_time: Optional[float] = Field(default=None, description="Time spent annotating (seconds)")
    prediction: Optional[Prediction] = Field(default=None, description="Associated prediction")
    result_count: int = Field(default=0, description="Number of results")
    unique_id: Optional[str] = Field(default=None, description="Unique annotation ID")
    import_id: Optional[str] = Field(default=None, description="Import ID")
    last_action: Optional[str] = Field(default=None, description="Last action performed")
    bulk_created: bool = Field(default=False, description="Was bulk created")
    task: Optional[int] = Field(default=None, description="Associated task ID")
    project: Optional[int] = Field(default=None, description="Associated project ID")
    updated_by: Optional[int] = Field(default=None, description="User ID who updated")
    parent_prediction: Optional[int] = Field(default=None, description="Parent prediction ID")
    parent_annotation: Optional[int] = Field(default=None, description="Parent annotation ID")
    last_created_by: Optional[int] = Field(default=None, description="Last created by user ID")

    class Config:
        extra = "ignore"  # Ignore additional fields
    
    @property
    def has_annotations(self) -> bool:
        """Check if this annotation contains any results."""
        return len(self.result) > 0
    
    @property
    def annotation_count(self) -> int:
        """Get the number of bounding boxes in this annotation."""
        return len(self.result)
    
    def get_labels(self) -> List[str]:
        """Get all unique labels in this annotation."""
        labels = set()
        for result in self.result:
            labels.update(result.labels)
        return list(labels)


class TaskData(BaseModel):
    """Task data containing the image path.
    
    The image path is typically a URL-encoded local file path or remote URL.
    """
    image: str = Field(..., description="Image path or URL")

    class Config:
        extra = "allow"  # Allow additional data fields
    
    def get_image_filename(self) -> str:
        """Extract the filename from the image path.
        
        Handles URL-encoded paths from Label Studio local file storage.
        
        Returns:
            Image filename without path
        """
        return Path(self.get_image_path()).name
    
    def get_image_path(self) -> str:
        """Get the decoded image path.
        
        Returns:
            Decoded image path
        """
        decoded = unquote(self.image)
        
        try:
            return get_local_path(decoded, download_resources=False)
        except (FileNotFoundError, Exception):
            # Fallback: parse path manually if get_local_path fails
            if "/data/local-files/?d=" in decoded:
                return decoded.split("?d=")[-1]
            return decoded


class Task(BaseModel):
    """Label Studio task containing image and annotations.
    
    A task represents a single item (image) to be annotated, along with
    all human annotations and model predictions.
    """
    id: int = Field(..., description="Task ID")
    data: TaskData = Field(..., description="Task data with image path")
    annotations: List[Annotation] = Field(default_factory=list, description="Human annotations")
    drafts: List[Any] = Field(default_factory=list, description="Draft annotations")
    predictions: List[Union[int, Prediction]] = Field(
        default_factory=list, 
        description="Predictions (IDs or full objects)"
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Task metadata")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")
    inner_id: Optional[int] = Field(default=None, description="Inner task ID")
    total_annotations: int = Field(default=0, description="Total annotation count")
    cancelled_annotations: int = Field(default=0, description="Cancelled annotation count")
    total_predictions: int = Field(default=0, description="Total prediction count")
    comment_count: int = Field(default=0, description="Comment count")
    unresolved_comment_count: int = Field(default=0, description="Unresolved comment count")
    last_comment_updated_at: Optional[datetime] = Field(default=None, description="Last comment update timestamp")
    project: Optional[int] = Field(default=None, description="Project ID")
    updated_by: Optional[int] = Field(default=None, description="User ID who updated")
    comment_authors: List[Any] = Field(default_factory=list, description="Comment authors")

    class Config:
        extra = "ignore"  # Ignore additional fields
    
    @property
    def image_filename(self) -> str:
        """Get the image filename."""
        return self.data.get_image_filename()
    
    @property
    def image_path(self) -> str:
        """Get the decoded image path."""
        return self.data.get_image_path()
    
    @property
    def has_annotations(self) -> bool:
        """Check if task has any annotations with results."""
        return any(ann.has_annotations for ann in self.annotations)
    
    def get_all_results(self) -> List[Result]:
        """Get all annotation results across all annotations."""
        results = []
        for annotation in self.annotations:
            results.extend(annotation.result)
        return results
    
    def get_annotation_count(self) -> int:
        """Get total number of bounding boxes across all annotations."""
        return sum(ann.annotation_count for ann in self.annotations)
    
    def get_labels(self) -> List[str]:
        """Get all unique labels in this task."""
        labels = set()
        for annotation in self.annotations:
            labels.update(annotation.get_labels())
        return list(labels)


class LabelStudioExport(BaseModel):
    """Root model for Label Studio JSON export.
    
    Represents the complete export file containing all tasks.
    """
    tasks: List[Task] = Field(..., description="List of all tasks")

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "LabelStudioExport":
        """Load export from a JSON file.
        
        Args:
            file_path: Path to the Label Studio JSON export file
            
        Returns:
            Parsed LabelStudioExport object
        """
        import json
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_list(data)
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "LabelStudioExport":
        """Create export from a list of task dictionaries.
        
        Args:
            data: List of task dictionaries
            
        Returns:
            Parsed LabelStudioExport object
        """
        tasks = [Task.model_validate(task) for task in data]
        return cls(tasks=tasks)
    
    @classmethod
    def from_json(cls, json_str: str) -> "LabelStudioExport":
        """Create export from a JSON string.
        
        Args:
            json_str: JSON string containing task list
            
        Returns:
            Parsed LabelStudioExport object
        """
        import json
        
        data = json.loads(json_str)
        return cls.from_list(data)
    
    @property
    def task_count(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)
    
    @property
    def annotated_task_count(self) -> int:
        """Get number of tasks with at least one annotation."""
        return sum(1 for task in self.tasks if task.has_annotations)
    
    def get_all_labels(self) -> List[str]:
        """Get all unique labels across all tasks."""
        labels = set()
        for task in self.tasks:
            labels.update(task.get_labels())
        return sorted(labels)
    
    def get_label_counts(self) -> Dict[str, int]:
        """Get count of annotations per label.
        
        Returns:
            Dictionary mapping label names to annotation counts
        """
        counts: Dict[str, int] = {}
        for task in self.tasks:
            for result in task.get_all_results():
                for label in result.labels:
                    counts[label] = counts.get(label, 0) + 1
        return counts
    
    def filter_by_labels(self, labels: List[str]) -> "LabelStudioExport":
        """Create a new export with only specified labels.
        
        Args:
            labels: List of label names to keep
            
        Returns:
            New LabelStudioExport with filtered annotations
        """
        label_set = set(labels)
        filtered_tasks = []
        
        for task in self.tasks:
            # Filter results within each annotation
            filtered_annotations = []
            for annotation in task.annotations:
                filtered_results = [
                    r for r in annotation.result 
                    if any(lbl in label_set for lbl in r.labels)
                ]
                if filtered_results:
                    # Create new annotation with filtered results
                    ann_dict = annotation.model_dump()
                    ann_dict["result"] = [r.model_dump() for r in filtered_results]
                    filtered_annotations.append(Annotation.model_validate(ann_dict))
            
            if filtered_annotations:
                task_dict = task.model_dump()
                task_dict["annotations"] = [a.model_dump() for a in filtered_annotations]
                filtered_tasks.append(Task.model_validate(task_dict))
        
        return LabelStudioExport(tasks=filtered_tasks)
