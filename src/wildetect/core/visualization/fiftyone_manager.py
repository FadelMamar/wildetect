"""
FiftyOne integration for WildDetect.

This module handles dataset creation, visualization, and annotation collection
using FiftyOne for wildlife detection datasets.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class FiftyOneManager:
    """Manages FiftyOne datasets for wildlife detection."""
    
    def __init__(self, dataset_name: Optional[str] = None):
        """Initialize FiftyOne manager.
        
        Args:
            dataset_name: Name of the dataset to use
        """
        self.config = get_config()
        self.dataset_name = dataset_name or self.config['fiftyone']['dataset_name']
        self.dataset = None
        
        # Initialize dataset
        self._init_dataset()
    
    def _init_dataset(self):
        """Initialize or load the FiftyOne dataset."""
        try:
            # Try to load existing dataset
            self.dataset = fo.load_dataset(self.dataset_name)
            logger.info(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            # Create new dataset
            self.dataset = fo.Dataset(self.dataset_name)
            logger.info(f"Created new dataset: {self.dataset_name}")
    
    def add_images(self, image_paths: List[str], detections: List[Dict[str, Any]] = None):
        """Add images to the dataset.
        
        Args:
            image_paths: List of image file paths
            detections: Optional list of detection results
        """
        if detections is None:
            detections = [None] * len(image_paths)
        
        samples = []
        for image_path, detection in zip(image_paths, detections):
            sample = fo.Sample(filepath=image_path)
            
            if detection and 'detections' in detection:
                # Add detection annotations
                detections_list = []
                for det in detection['detections']:
                    detection_obj = fo.Detection(
                        bounding_box=det['bbox'],
                        confidence=det['confidence'],
                        label=det['class_name']
                    )
                    detections_list.append(detection_obj)
                
                sample["detections"] = fo.Detections(detections=detections_list)
                sample["total_count"] = detection.get('total_count', 0)
                sample["species_counts"] = detection.get('species_counts', {})
            
            samples.append(sample)
        
        self.dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} images to dataset")
    
    def launch_app(self):
        """Launch the FiftyOne app."""
        try:
            self.dataset.launch_app()
            logger.info("FiftyOne app launched")
        except Exception as e:
            logger.error(f"Error launching FiftyOne app: {e}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'name': self.dataset.name,
            'num_samples': len(self.dataset),
            'tags': list(self.dataset.get_tags()),
            'fields': list(self.dataset.get_field_schema().keys())
        }
    
    def export_annotations(self, output_path: str, format: str = "coco"):
        """Export annotations from the dataset.
        
        Args:
            output_path: Path to save annotations
            format: Export format ('coco', 'yolo', 'pascal')
        """
        try:
            if format == "coco":
                self.dataset.export(
                    export_dir=output_path,
                    dataset_type=fo.types.COCODetectionDataset
                )
            elif format == "yolo":
                self.dataset.export(
                    export_dir=output_path,
                    dataset_type=fo.types.YOLOv5Dataset
                )
            elif format == "pascal":
                self.dataset.export(
                    export_dir=output_path,
                    dataset_type=fo.types.VOCDetectionDataset
                )
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported annotations to {output_path} in {format} format")
        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
    
    def compute_similarity(self):
        """Compute similarity between samples using FiftyOne Brain."""
        try:
            if self.config['fiftyone']['enable_brain']:
                fob.compute_similarity(self.dataset, "detections")
                logger.info("Computed similarity embeddings")
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
    
    def find_hardest_samples(self, num_samples: int = 100):
        """Find the most challenging samples for annotation."""
        try:
            if self.config['fiftyone']['enable_brain']:
                hardest = fob.compute_hardest(
                    self.dataset,
                    "detections",
                    num_samples=num_samples
                )
                logger.info(f"Found {len(hardest)} hardest samples")
                return hardest
        except Exception as e:
            logger.error(f"Error finding hardest samples: {e}")
        return None
    
    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about annotations in the dataset."""
        stats = {
            'total_samples': len(self.dataset),
            'annotated_samples': len(self.dataset.match(F("detections").exists())),
            'total_detections': 0,
            'species_counts': {}
        }
        
        # Count detections and species
        for sample in self.dataset:
            if "detections" in sample:
                detections = sample["detections"]
                if detections:
                    stats['total_detections'] += len(detections.detections)
                    
                    for detection in detections.detections:
                        species = detection.label
                        stats['species_counts'][species] = stats['species_counts'].get(species, 0) + 1
        
        return stats
    
    def create_view(self, filters: Dict[str, Any] = None) -> fo.DatasetView:
        """Create a filtered view of the dataset.
        
        Args:
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered dataset view
        """
        view = self.dataset
        
        if filters:
            if 'min_confidence' in filters:
                view = view.match(F("detections.detections.confidence") >= filters['min_confidence'])
            
            if 'species' in filters:
                view = view.match(F("detections.detections.label") == filters['species'])
            
            if 'min_detections' in filters:
                view = view.match(F("detections.detections").length() >= filters['min_detections'])
        
        return view
    
    def save_dataset(self):
        """Save the dataset to disk."""
        try:
            self.dataset.save()
            logger.info("Dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    def close(self):
        """Close the dataset."""
        if self.dataset:
            self.dataset.close()
            logger.info("Dataset closed") 