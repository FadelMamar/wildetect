"""
LabelStudio integration for WildDetect.

This module handles LabelStudio project creation, annotation job management,
and integration with FiftyOne for seamless workflow.
"""

import os
import logging
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import shutil

from label_studio_sdk import Client
from label_studio_sdk.data_manager import DataManager
from label_studio_sdk.project import Project

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class LabelStudioManager:
    """Manages LabelStudio projects and annotation workflows."""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize LabelStudio manager.
        
        Args:
            url: LabelStudio server URL
            api_key: LabelStudio API key
        """
        self.config = get_config()
        self.url = url or self.config.get('labelstudio', {}).get('url', 'http://localhost:8080')
        self.api_key = api_key or self.config.get('labelstudio', {}).get('api_key')
        
        # Initialize LabelStudio client
        self.client = Client(url=self.url, api_key=self.api_key)
        
        logger.info(f"LabelStudio manager initialized with URL: {self.url}")
    
    def create_project(self, project_name: str, description: str = "") -> Project:
        """Create a new LabelStudio project.
        
        Args:
            project_name: Name of the project
            description: Project description
            
        Returns:
            LabelStudio project object
        """
        try:
            # Create project
            project = self.client.create_project(
                title=project_name,
                description=description,
                label_config=self._get_wildlife_annotation_config()
            )
            
            logger.info(f"Created LabelStudio project: {project_name}")
            return project
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise
    
    def _get_wildlife_annotation_config(self) -> str:
        """Get LabelStudio configuration for wildlife annotation.
        
        Returns:
            LabelStudio configuration XML
        """
        config = """
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            <Label value="elephant" background="green"/>
            <Label value="giraffe" background="blue"/>
            <Label value="zebra" background="yellow"/>
            <Label value="lion" background="red"/>
            <Label value="rhino" background="purple"/>
            <Label value="buffalo" background="orange"/>
            <Label value="antelope" background="brown"/>
            <Label value="deer" background="pink"/>
            <Label value="bear" background="gray"/>
            <Label value="wolf" background="black"/>
            <Label value="fox" background="cyan"/>
            <Label value="rabbit" background="lime"/>
            <Label value="bird" background="magenta"/>
            <Label value="other" background="white"/>
          </RectangleLabels>
        </View>
        """
        return config
    
    def import_tasks_from_fiftyone(self, project: Project, dataset_name: str, 
                                  image_paths: List[str], detections: List[Dict[str, Any]] = None):
        """Import tasks from FiftyOne dataset to LabelStudio.
        
        Args:
            project: LabelStudio project
            dataset_name: FiftyOne dataset name
            image_paths: List of image paths
            detections: Optional list of detection results for pre-annotation
        """
        try:
            tasks = []
            
            for i, image_path in enumerate(image_paths):
                task = {
                    "data": {
                        "image": f"/data/upload/{os.path.basename(image_path)}"
                    }
                }
                
                # Add pre-annotations if available
                if detections and i < len(detections):
                    detection = detections[i]
                    if 'detections' in detection:
                        annotations = []
                        for det in detection['detections']:
                            annotation = {
                                "value": {
                                    "x": det['bbox'][0],
                                    "y": det['bbox'][1],
                                    "width": det['bbox'][2] - det['bbox'][0],
                                    "height": det['bbox'][3] - det['bbox'][1],
                                    "rectanglelabels": [det['class_name']]
                                },
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels"
                            }
                            annotations.append(annotation)
                        
                        task["annotations"] = [{
                            "result": annotations,
                            "completed_by": 1
                        }]
                
                tasks.append(task)
            
            # Import tasks
            project.import_tasks(tasks)
            logger.info(f"Imported {len(tasks)} tasks to LabelStudio project")
            
        except Exception as e:
            logger.error(f"Error importing tasks: {e}")
            raise
    
    def export_annotations_to_fiftyone(self, project: Project, output_dir: str) -> List[Dict[str, Any]]:
        """Export annotations from LabelStudio to FiftyOne format.
        
        Args:
            project: LabelStudio project
            output_dir: Output directory for annotations
            
        Returns:
            List of annotation dictionaries
        """
        try:
            # Get all tasks with annotations
            tasks = project.get_tasks()
            annotations = []
            
            for task in tasks:
                if task.get('annotations'):
                    # Get the first completed annotation
                    annotation = task['annotations'][0]
                    
                    # Extract bounding boxes
                    detections = []
                    for result in annotation['result']:
                        if result['type'] == 'rectanglelabels':
                            bbox = [
                                result['value']['x'],
                                result['value']['y'],
                                result['value']['x'] + result['value']['width'],
                                result['value']['y'] + result['value']['height']
                            ]
                            
                            detection = {
                                'bbox': bbox,
                                'class_name': result['value']['rectanglelabels'][0],
                                'confidence': 1.0  # Manual annotation
                            }
                            detections.append(detection)
                    
                    # Create FiftyOne format annotation
                    annotation_data = {
                        'image_path': task['data']['image'],
                        'detections': detections,
                        'total_count': len(detections),
                        'species_counts': self._count_by_species(detections)
                    }
                    
                    annotations.append(annotation_data)
            
            # Save annotations
            output_file = os.path.join(output_dir, 'labelstudio_annotations.json')
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            logger.info(f"Exported {len(annotations)} annotations to {output_file}")
            return annotations
            
        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
            raise
    
    def _count_by_species(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count detections by species.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary mapping species names to counts
        """
        counts = {}
        for detection in detections:
            species = detection['class_name']
            counts[species] = counts.get(species, 0) + 1
        return counts
    
    def get_project_stats(self, project: Project) -> Dict[str, Any]:
        """Get statistics about a LabelStudio project.
        
        Args:
            project: LabelStudio project
            
        Returns:
            Project statistics
        """
        try:
            tasks = project.get_tasks()
            annotations = project.get_annotations()
            
            stats = {
                'total_tasks': len(tasks),
                'annotated_tasks': len([t for t in tasks if t.get('annotations')]),
                'total_annotations': len(annotations),
                'completion_rate': 0
            }
            
            if stats['total_tasks'] > 0:
                stats['completion_rate'] = stats['annotated_tasks'] / stats['total_tasks']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting project stats: {e}")
            raise
    
    def create_annotation_job(self, project_name: str, image_paths: List[str], 
                            detections: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a complete annotation job.
        
        Args:
            project_name: Name of the LabelStudio project
            image_paths: List of image paths to annotate
            detections: Optional pre-annotations from detection model
            
        Returns:
            Job information dictionary
        """
        try:
            # Create project
            project = self.create_project(project_name)
            
            # Import tasks
            self.import_tasks_from_fiftyone(project, project_name, image_paths, detections)
            
            # Get project stats
            stats = self.get_project_stats(project)
            
            job_info = {
                'project_id': project.id,
                'project_name': project_name,
                'url': f"{self.url}/projects/{project.id}/data",
                'total_tasks': stats['total_tasks'],
                'completion_rate': stats['completion_rate']
            }
            
            logger.info(f"Created annotation job: {job_info}")
            return job_info
            
        except Exception as e:
            logger.error(f"Error creating annotation job: {e}")
            raise
    
    def sync_with_fiftyone(self, project: Project, fiftyone_dataset_name: str):
        """Sync LabelStudio annotations with FiftyOne dataset.
        
        Args:
            project: LabelStudio project
            fiftyone_dataset_name: FiftyOne dataset name
        """
        try:
            from .fiftyone_manager import FiftyOneManager
            
            # Export annotations from LabelStudio
            config = get_config()
            export_dir = config['paths']['annotations_dir']
            annotations = self.export_annotations_to_fiftyone(project, export_dir)
            
            # Import to FiftyOne
            fo_manager = FiftyOneManager(fiftyone_dataset_name)
            
            image_paths = [ann['image_path'] for ann in annotations]
            fo_manager.add_images(image_paths, annotations)
            
            logger.info(f"Synced {len(annotations)} annotations with FiftyOne dataset")
            
        except Exception as e:
            logger.error(f"Error syncing with FiftyOne: {e}")
            raise
    
    def get_annotation_progress(self, project: Project) -> Dict[str, Any]:
        """Get annotation progress for a project.
        
        Args:
            project: LabelStudio project
            
        Returns:
            Progress information
        """
        try:
            stats = self.get_project_stats(project)
            
            progress = {
                'total_tasks': stats['total_tasks'],
                'completed_tasks': stats['annotated_tasks'],
                'completion_rate': stats['completion_rate'],
                'remaining_tasks': stats['total_tasks'] - stats['annotated_tasks']
            }
            
            return progress
            
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            raise
    
    def export_for_training(self, project: Project, output_dir: str, format: str = "yolo"):
        """Export annotations in training format.
        
        Args:
            project: LabelStudio project
            output_dir: Output directory
            format: Export format ('yolo', 'coco', 'pascal')
        """
        try:
            annotations = self.export_annotations_to_fiftyone(project, output_dir)
            
            if format == "yolo":
                self._export_yolo_format(annotations, output_dir)
            elif format == "coco":
                self._export_coco_format(annotations, output_dir)
            elif format == "pascal":
                self._export_pascal_format(annotations, output_dir)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported annotations in {format} format to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting for training: {e}")
            raise
    
    def _export_yolo_format(self, annotations: List[Dict[str, Any]], output_dir: str):
        """Export annotations in YOLO format.
        
        Args:
            annotations: List of annotation dictionaries
            output_dir: Output directory
        """
        config = get_config()
        class_names = config['detection']['species_classes']
        
        # Create directories
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for i, annotation in enumerate(annotations):
            # Copy image
            image_path = annotation['image_path']
            image_name = f"image_{i:06d}.jpg"
            new_image_path = os.path.join(images_dir, image_name)
            shutil.copy2(image_path, new_image_path)
            
            # Create YOLO label file
            label_name = f"image_{i:06d}.txt"
            label_path = os.path.join(labels_dir, label_name)
            
            with open(label_path, 'w') as f:
                for detection in annotation['detections']:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    
                    # Get class ID
                    try:
                        class_id = class_names.index(class_name)
                    except ValueError:
                        class_id = len(class_names) - 1  # 'other' class
                    
                    # Convert to YOLO format
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def _export_coco_format(self, annotations: List[Dict[str, Any]], output_dir: str):
        """Export annotations in COCO format.
        
        Args:
            annotations: List of annotation dictionaries
            output_dir: Output directory
        """
        # COCO format implementation
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        config = get_config()
        class_names = config['detection']['species_classes']
        for i, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "wildlife"
            })
        
        # Add images and annotations
        annotation_id = 1
        for i, annotation in enumerate(annotations):
            # Add image
            image_info = {
                "id": i + 1,
                "file_name": os.path.basename(annotation['image_path']),
                "width": 1920,  # Default, should get from actual image
                "height": 1080   # Default, should get from actual image
            }
            coco_data["images"].append(image_info)
            
            # Add annotations
            for detection in annotation['detections']:
                bbox = detection['bbox']
                class_name = detection['class_name']
                
                try:
                    category_id = class_names.index(class_name) + 1
                except ValueError:
                    category_id = len(class_names)
                
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": i + 1,
                    "category_id": category_id,
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1
        
        # Save COCO file
        coco_file = os.path.join(output_dir, "annotations.json")
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def _export_pascal_format(self, annotations: List[Dict[str, Any]], output_dir: str):
        """Export annotations in Pascal VOC format.
        
        Args:
            annotations: List of annotation dictionaries
            output_dir: Output directory
        """
        # Pascal VOC format implementation
        # This would create XML files for each image
        pass  # Implementation would be similar to COCO but with XML files 