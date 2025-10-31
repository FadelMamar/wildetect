"""
Curriculum-based dataset implementations.

This module provides dataset classes that support curriculum learning by filtering
samples based on difficulty levels.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from PIL import Image
import torch
import logging
import supervision as sv
import albumentations as A
import cv2
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor

from ...shared.models import CurriculumConfig
from ..utils import load_all_detection_datasets
from ...utils.io import read_image

logger = logging.getLogger(__name__)


def group_coco_annotations_by_image_id(
        coco_annotations: list[dict],
    ) -> dict[int, list[dict]]:
        annotations = {}
        for annotation in coco_annotations:
            image_id = annotation["image_id"]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        return annotations

class CurriculumDetectionDataset(sv.DetectionDataset):
    """
    Curriculum-aware dataset that inherits from supervision.DetectionDataset.
    
    This dataset can:
    1. Filter samples based on difficulty (for difficulty-based curriculum)
    2. Resize images based on current scale (for multi-scale curriculum)
    3. Work seamlessly with the CurriculumManager
    4. Leverage supervision's lazy loading and memory efficiency
    5. Use supervision's convenience loading methods (from_coco, from_yolo, etc.)
    """
    
    def __init__(self, 
                 classes: List[str],
                 images: Union[List[str], Dict[str, np.ndarray]],
                 annotations: Dict[str, sv.Detections],
                 curriculum_config: Optional[CurriculumConfig] = None,
                 transform=None,
                 compute_difficulties: bool = True,
                 preserve_aspect_ratio: bool = True,):
        """
        Initialize curriculum detection dataset.
        
        Args:
            classes: List of class names
            images: List of image paths or dict of loaded images
            annotations: Dictionary mapping image paths to Detections
            curriculum_config: Curriculum configuration
            transform: Image transformations
            compute_difficulties: Whether to compute sample difficulties
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
        """
        super().__init__(classes, images, annotations)
        self.curriculum_config = curriculum_config
        self.transform = transform
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Compute sample difficulties if needed
        if compute_difficulties:
            self.sample_difficulties = self._compute_sample_difficulties()
        else:
            self.sample_difficulties = [0.5] * len(self.image_paths)  # Default medium difficulty
        
        # Initialize available indices (all samples initially)
        self.available_indices = list(range(len(self.image_paths)))
        
        # Current curriculum state
        self.current_difficulty = 1.0

    @classmethod
    def from_data_directory(cls,
                                 root_data_directory: str,
                                 split: str,
                                 curriculum_config: Optional[CurriculumConfig] = None,
                                 transform=None,
                                 compute_difficulties: bool = True,
                                 preserve_aspect_ratio: bool = True,):
        """
        Create curriculum detection dataset from COCO format.
        
        Args:
            root_data_directory: Path to the root data directory
            split: Dataset split to load ('train', 'val', 'test')
            curriculum_config: Curriculum configuration
            transform: Image transformations
            compute_difficulties: Whether to compute sample difficulties
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            
        Returns:
            CurriculumDetectionDataset instance
        """

        detection_dataset = load_all_detection_datasets(
            root_data_directory=root_data_directory,
            split=split
        )
        return cls(
            classes=detection_dataset.classes,
            images=detection_dataset.image_paths, # + background_images,
            annotations=detection_dataset.annotations,
            curriculum_config=curriculum_config,
            transform=transform,
            compute_difficulties=compute_difficulties,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
           
    def _compute_sample_difficulties(self) -> List[float]:
        """Compute difficulty scores for each sample using multiple strategies."""
        difficulties = []
        
        for image_path in self.image_paths:
            detections = self.annotations[image_path]
            
            if len(detections) > 0:
                # Use multiple difficulty metrics and combine them
                area_difficulty = self._difficulty_by_area(detections)
                count_difficulty = self._difficulty_by_count(detections)
                ratio_difficulty = self._difficulty_by_ratio(detections)
                
                # Combine difficulties (weighted average)
                combined_difficulty = (
                    0.4 * area_difficulty + 
                    0.3 * count_difficulty + 
                    0.3 * ratio_difficulty
                )
                difficulties.append(combined_difficulty)
            else:
                difficulties.append(0.5)  # Default medium difficulty for empty images
        
        return difficulties
    
    def _difficulty_by_area(self, detections: sv.Detections) -> float:
        """
        Compute difficulty based on detection area (smaller = harder).
        
        Args:
            detections: Detection annotations
            
        Returns:
            Difficulty score (0.0 = easiest, 1.0 = hardest)
        """
        if len(detections) == 0:
            return 0.5
        
        # Calculate average area of all detections
        areas = []
        for bbox in detections.xyxy:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)
        
        avg_area = sum(areas) / len(areas)
        
        # Normalize area to difficulty (smaller area = higher difficulty)
        # Use a reference area of 200x200 pixels
        reference_area = 200 * 200
        normalized_difficulty = 1.0 - min(1.0, avg_area / reference_area)
        
        return normalized_difficulty
    
    def _difficulty_by_count(self, detections: sv.Detections) -> float:
        """
        Compute difficulty based on number of detections (more = easier).
        
        Args:
            detections: Detection annotations
            
        Returns:
            Difficulty score (0.0 = easiest, 1.0 = hardest)
        """
        if detections.xyxy is None:
            return 0.5
        
        # More detections = lower difficulty
        # Normalize to a reasonable range (0-10 detections)
        max_expected_detections = 10
        count_difficulty = 1 - min(1.0, len(detections.xyxy) / max_expected_detections)
        
        return count_difficulty
    
    def _difficulty_by_ratio(self, detections: sv.Detections,) -> float:
        """
        Compute difficulty based on ratio between bbox and image dimensions.
        
        Args:
            detections: Detection annotations
            image_path: Path to the image
            
        Returns:
            Difficulty score (0.0 = easiest, 1.0 = hardest)
        """
        if len(detections) == 0:
            return 0.5
        
        # Reference dimensions
        img_height, img_width = 800,800
        img_area = img_height * img_width
        
        # Calculate total bbox area
        total_bbox_area = 0
        for bbox in detections.xyxy:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_bbox_area += bbox_area
        
        # Calculate coverage ratio
        coverage_ratio = total_bbox_area / img_area
        
        # Normalize ratio to difficulty
        # Very small coverage (hard to detect) = high difficulty
        # Very large coverage (easy to detect) = low difficulty
        # Medium coverage = medium difficulty
        
        if coverage_ratio < 0.01:  # Very small objects
            ratio_difficulty = 1.0
        elif coverage_ratio > 0.5:  # Very large objects
            ratio_difficulty = 0.0
        else:
            # Linear interpolation for medium ratios
            ratio_difficulty = 1.0 - (coverage_ratio - 0.01) / (0.5 - 0.01)
            ratio_difficulty = max(0.0, min(1.0, ratio_difficulty))
        
        return ratio_difficulty
            
    def update_curriculum_state(self, difficulty: float):
        """
        Update dataset based on current curriculum state.
        
        Args:
            difficulty: Current difficulty level (0.0 = easiest, 1.0 = hardest)
        """
        self.current_difficulty = difficulty
        
        # Update available samples based on difficulty
        if self.curriculum_config and self.curriculum_config.enabled:
            self._update_available_samples()
    
    def _update_available_samples(self):
        """Update which samples are available based on current difficulty."""
        if not self.curriculum_config or self.curriculum_config.type != "difficulty":
            self.available_indices = list(range(len(self.image_paths)))
            return
        
        self.available_indices = []
        
        for i, difficulty in enumerate(self.sample_difficulties):
            # Include samples that are not harder than current difficulty level
            if difficulty <= self.current_difficulty + 0.1:  # Small buffer
                self.available_indices.append(i)
        
        # Ensure we always have some samples
        if len(self.available_indices) == 0:
            self.available_indices = list(range(len(self.image_paths)))
    
    def __len__(self):
        return len(self.available_indices)
    
    def __getitem__(self, idx):
        # Map to actual sample index
        actual_idx = self.available_indices[idx]
        
        # Get image path
        image_path = self.image_paths[actual_idx]
        
        # Load image using supervision's lazy loading
        image = self._get_image(image_path)
        
        # Get annotations
        detections = self.annotations[image_path]
        
        # Convert to PIL and apply transforms
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Return the expected tuple format for supervision compatibility
        return image_path, image, detections

class PatchDataset(torch.utils.data.Dataset):
    """
    Custom dataloader for loading cropped regions of interest (ROIs) from images.
    
    This dataloader extracts crops around detection bounding boxes and returns
    the cropped image along with the corresponding label. Crops are generated
    lazily for memory efficiency.
    """
    
    def __init__(self, 
                 dataset: CurriculumDetectionDataset,
                 crop_size: int = 224,
                 max_tn_crops: int = 1,
                 p_draw_annotations: float = 0.,
                 load_as_single_class: bool = True,
                 background_class_name: str = "background",
                 single_class_name: str = "wildlife",
                 keep_classes: Optional[List[str]] = None,
                 discard_classes: Optional[List[str]] = None
                 ):
        """
        Initialize crop dataloader.
        
        Args:
            dataset: CurriculumDetectionDataset to load crops from
            crop_size: Size of the square crop (width = height)
            max_tn_crops: Maximum number of crops to extract per image
            p_draw_annotations: Probability of drawing annotations
            load_as_single_class: Whether to binarize classes to {0: background, 1: wildlife}
            background_class_name: Name for background class
            single_class_name: Name for wildlife class
            keep_classes: List of classes to keep (others become background)
            discard_classes: List of classes to discard (become background)
        """
        super().__init__()
        self.dataset = dataset
        self.crop_size = crop_size
        self.max_tn_crops = max_tn_crops
        self.p_draw_annotations = p_draw_annotations
        self.load_as_single_class = load_as_single_class
        self.background_class_name = background_class_name
        self.single_class_name = single_class_name
        self.keep_classes = keep_classes
        self.discard_classes = discard_classes

        # Multi-class: original classes + background
        self.class_mapping = {i+1: class_name for i, class_name in enumerate(self.dataset.classes)}
        self.class_mapping[0] = self.background_class_name
        
        # Pre-compute crop indices for lazy generation
        self.crop_indices = self._compute_crop_indices()
        self._update_crop_indices()

        self.pad_roi = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.crop_size,
                    min_width=self.crop_size,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    fill=0,
                ),
                A.CenterCrop(
                    height=self.crop_size,
                    width=self.crop_size,
                    p=1,
                ),
            ]
        )

        # Set up class mapping based on single class configuration
        if self.load_as_single_class:
            # Binary classification: {0: background, 1: wildlife}
            self.class_mapping = {
                0: self.background_class_name,
                1: self.single_class_name
            }            
    
    def _update_crop_indices(self,):
        if self.keep_classes:
            updated_crop_indices = [crop_info for crop_info in self.crop_indices if crop_info['class_name'] in self.keep_classes]
            self.class_mapping = {k: v for k, v in self.class_mapping.items() if v in self.keep_classes}
            logger.info(
                f"Keeping classes: {self.keep_classes}. Updated class mapping: {self.class_mapping}"
            )
            self.crop_indices = updated_crop_indices
        elif self.discard_classes:
            updated_crop_indices = [crop_info for crop_info in self.crop_indices if crop_info['class_name'] not in self.discard_classes]
            self.class_mapping = {k: v for k, v in self.class_mapping.items() if v not in self.discard_classes}
            logger.info(
                f"Discarding classes: {self.discard_classes}. Updated class mapping: {self.class_mapping}"
            )
            self.crop_indices = updated_crop_indices
              
    def _is_wildlife_class(self, class_id: int) -> bool:
        """
        Determine if a class should be treated as wildlife (class 1) or background (class 0).
        
        Args:
            class_id: Original class ID
            
        Returns:
            True if class should be wildlife, False if background
        """
        if class_id < 0:  # Random crops or invalid classes
            return False
        else:
            return True

    def _compute_crop_indices(self) -> List[Dict[str, Any]]:
        """Pre-compute crop indices for lazy generation."""
        crop_indices = []

        def iterator():
            for dataset_idx,image_path in enumerate(self.dataset.image_paths):
                img = read_image(image_path)
                w, h = img.size
                detections = self.dataset.annotations[image_path] 
                yield w,h,detections,image_path,dataset_idx

        def func(args):
            w,h,detections,image_path,dataset_idx = args
            # Add random crop indices if enabled
            if detections.is_empty() or len(detections.xyxy) == 0 or detections.xyxy is None:
                indices = self._compute_random_indices(
                    image_path, dataset_idx, h, w
                )
            else:
                # Add detection crop indices
                indices = self._compute_detection_indices(
                    detections, image_path, dataset_idx, h, w
                )
            return indices
            
        with tqdm(total=len(self.dataset.image_paths),desc="Computing crop indices",unit="images") as pbar:
            #with ThreadPoolExecutor(max_workers=3) as executor:
            for result in map(func, iterator()):
                crop_indices.extend(result)
                pbar.update(1)        
        return crop_indices
    
    def _compute_detection_indices(self, 
                                  detections: sv.Detections, 
                                  image_path: str, 
                                  dataset_idx: int,
                                  img_height: int,
                                  img_width: int) -> List[Dict[str, Any]]:
        """Compute indices for detection-based crops."""
        indices = []
                
        # Sort detections by area (largest first for better crops)
        areas = []
        for i,bbox in enumerate(detections.xyxy):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)
            class_id = detections.class_id[i]
        
            # Get bounding box coordinates
            x1, y1, x2, y2 = bbox.tolist()
            
            # Calculate center of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate crop coordinates centered on the detection
            # Use a simpler approach: ensure crop is at least crop_size
            crop_half_size = self.crop_size // 2
            
            x1_crop = max(0, int(center_x - crop_half_size))
            y1_crop = max(0, int(center_y - crop_half_size))
            x2_crop = min(img_width, int(center_x + crop_half_size))
            y2_crop = min(img_height, int(center_y + crop_half_size))
            
            # Ensure minimum crop size by adjusting if needed
            if x2_crop - x1_crop < self.crop_size:
                if x1_crop == 0:
                    x2_crop = min(img_width, self.crop_size)
                else:
                    x1_crop = max(0, x2_crop - self.crop_size)
            
            if y2_crop - y1_crop < self.crop_size:
                if y1_crop == 0:
                    y2_crop = min(img_height, self.crop_size)
                else:
                    y1_crop = max(0, y2_crop - self.crop_size)
            
            # Additional validation: ensure crop doesn't exceed image boundaries
            if img_width < self.crop_size:
                # If image is too narrow, use the entire width
                x1_crop = 0
                x2_crop = img_width
            
            if img_height < self.crop_size:
                # If image is too short, use the entire height
                y1_crop = 0
                y2_crop = img_height
            
            # Final validation
            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                # Skip this detection if crop is invalid
                continue
            
            # Additional safety check: ensure coordinates are within image bounds
            x1_crop = max(0, min(x1_crop, img_width - 1))
            y1_crop = max(0, min(y1_crop, img_height - 1))
            x2_crop = max(x1_crop + 1, min(x2_crop, img_width))
            y2_crop = max(y1_crop + 1, min(y2_crop, img_height))
                        
            indices.append({
                'dataset_idx': dataset_idx,
                'image_path': image_path,
                'crop_bbox': [x1_crop, y1_crop, x2_crop, y2_crop],
                'original_bbox': bbox,
                'label': class_id,
                'crop_type': 'detection',
                "class_name": self.dataset.classes[class_id]
            })
        
        return indices
    
    def _compute_random_indices(self, 
                               image_path: str, 
                               dataset_idx: int,
                               img_height: int,
                               img_width: int) -> List[Dict[str, Any]]:
        """Compute indices for random crops."""
        indices = []
              
        for _ in range(self.max_tn_crops):
           
            # Check if image is too small for the crop size
            if img_width < self.crop_size or img_height < self.crop_size:
                # If image is too small, use the entire image
                x1, y1 = 0, 0
                x2, y2 = img_width, img_height
            else:
                # Random position
                x1 = np.random.randint(0, img_width - self.crop_size + 1)
                y1 = np.random.randint(0, img_height - self.crop_size + 1)
                x2 = x1 + self.crop_size
                y2 = y1 + self.crop_size
            
            # Validate crop coordinates
            if x2 <= x1 or y2 <= y1:
                # Skip this crop if coordinates are invalid
                continue
            
            indices.append({
                'dataset_idx': dataset_idx,
                'image_path': image_path,
                'crop_bbox': [x1, y1, x2, y2],
                'original_bbox': None,
                'label': -1,  # -1 indicates random crop (no specific label)
                'crop_type': 'random',
                "class_name": self.background_class_name
            })
        
        return indices
    
    def _extract_crop(self, crop_info: Dict[str, Any]) -> np.ndarray:
        """Extract crop from image using pre-computed indices."""
        dataset_idx = crop_info['dataset_idx']
        x1, y1, x2, y2 = crop_info['crop_bbox']
        
        # Get original image
        _, image, _ = self.dataset[dataset_idx]
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(image)
        
        #if img_np.max() <= 1.0:
            #img_np = (img_np * 255).astype(np.uint8)
        
        # Validate crop coordinates
        h, w = img_np.shape[:2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Debug: log the coordinates and image dimensions
        logger.debug(f"Image dimensions: {w}x{h}, Crop coordinates: {x1}, {y1}, {x2}, {y2}")
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        # Handle edge case where image is exactly crop size
        if w == self.crop_size and h == self.crop_size:
            x1, y1, x2, y2 = 0, 0, w, h
        
        # Check if crop is valid
        if x2 <= x1 or y2 <= y1:
            # Return a black crop if coordinates are invalid
            raise ValueError(f"Invalid crop coordinates: {x1}, {y1}, {x2}, {y2}")
        
        # Extract crop
        crop = img_np[y1:y2, x1:x2]
        
        # Check if crop is empty
        if crop.size == 0:
            raise ValueError(f"Invalid crop coordinates: {x1}, {y1}, {x2}, {y2}")
        
        # Pad crop to square
        crop = self.pad_roi(image=crop)["image"]

        return crop
    
    def get_annotations_for_filter(self) -> List[Dict[str, Any]]:
        """
        Generate annotations in the format expected by ClassificationRebalanceFilter.
        
        Returns:
            List of annotation dictionaries with roi_id, class_name, class_id, file_name
        """
        annotations = []
        roi_id = 0
        
        for idx, crop_info in enumerate(self.crop_indices):
            # Generate filename
            dataset_idx = crop_info['dataset_idx']
            image_path = crop_info['image_path']
            original_class_id = crop_info['label']
            class_name = crop_info['class_name']
            
            # Apply single class binarization if enabled
            if self.load_as_single_class:
                # background is 0, wildlife is 1
                class_id = 0 if original_class_id < 0 else 1
                class_name = self.background_class_name if class_id == 0 else self.single_class_name
            else:
                # Multi-class: use original class
                class_id = original_class_id + 1  # offset for background
                        
            # Generate filename
            base_name = Path(image_path).stem
            file_name = f"{base_name}_roi_{roi_id:06d}.jpg"
            
            annotation = {
                "roi_id": roi_id,
                "class_name": class_name,
                "class_id": class_id,
                "file_name": file_name,
                "dataset_idx": dataset_idx,
                "crop_bbox": crop_info['crop_bbox'],
                "crop_type": crop_info['crop_type']
            }
            
            # Add original annotation info if available
            if crop_info.get('original_bbox') is not None:
                annotation["original_bbox"] = crop_info['original_bbox']
                annotation["original_annotation_id"] = idx
            
            annotations.append(annotation)
            roi_id += 1
        
        return annotations
    
    def apply_rebalance_filter(self, filter_instance:Callable) -> 'PatchDataset':
        """
        Apply ClassificationRebalanceFilter to create a balanced dataset.
        
        Args:
            filter_instance: Instance of ClassificationRebalanceFilter
            
        Returns:
            New PatchDataset with balanced crop indices
        """
        # Get annotations in the expected format
        annotations = self.get_annotations_for_filter()
        
        # Apply filter
        filtered_annotations = filter_instance(annotations)
        
        # Create new dataset with filtered indices
        filtered_indices = []
        for filtered_ann in filtered_annotations:
            roi_id = filtered_ann['roi_id']
            # Find the corresponding crop index
            for idx, crop_info in enumerate(self.crop_indices):
                if idx == roi_id:
                    filtered_indices.append(crop_info)
                    break
        
        # Create new dataset instance
        new_dataset = PatchDataset(
            dataset=self.dataset,
            crop_size=self.crop_size,
            max_tn_crops=self.max_tn_crops,
            p_draw_annotations=self.p_draw_annotations,
            load_as_single_class=self.load_as_single_class,
            background_class_name=self.background_class_name,
            single_class_name=self.single_class_name,
            keep_classes=self.keep_classes,
            discard_classes=self.discard_classes
        )
        
        # Replace crop indices with filtered ones
        new_dataset.crop_indices = filtered_indices
        
        return new_dataset

    def apply_clustering_filter(self, filter_instance:Callable) -> 'PatchDataset':
        """
        Apply ClusteringFilter (via adapter) to create a clustered dataset.
        
        Args:
            filter_instance: Instance of ClusteringFilter or CropClusteringAdapter
            
        Returns:
            New PatchDataset with clustered crop indices
        """
        # Get annotations in the expected format
        annotations = self.get_annotations_for_filter()
        
        # Apply filter (adapter handles conversion if needed)
        filtered_annotations = filter_instance(annotations)
        
        # Create new dataset with filtered indices
        filtered_indices = []
        for filtered_ann in filtered_annotations:
            roi_id = filtered_ann['roi_id']
            # Find the corresponding crop index
            for idx, crop_info in enumerate(self.crop_indices):
                if idx == roi_id:
                    filtered_indices.append(crop_info)
                    break
        
        # Create new dataset instance
        new_dataset = PatchDataset(
            dataset=self.dataset,
            crop_size=self.crop_size,
            max_tn_crops=self.max_tn_crops,
            p_draw_annotations=self.p_draw_annotations,
            load_as_single_class=self.load_as_single_class,
            background_class_name=self.background_class_name,
            single_class_name=self.single_class_name,
            keep_classes=self.keep_classes,
            discard_classes=self.discard_classes
        )
        
        # Replace crop indices with filtered ones
        new_dataset.crop_indices = filtered_indices
        
        return new_dataset
    
    def __len__(self) -> int:
        """Return number of crops."""
        return len(self.crop_indices)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a crop and its label (lazy generation).
        
        Args:
            idx: Index of the crop
            
        Returns:
            Tuple of (crop_image, label)
        """
        crop_info = self.crop_indices[idx]
        
        # Extract crop lazily
        crop = self._extract_crop(crop_info)
        
        # Convert to tensor
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        
        # Get original label
        original_label = crop_info['label']
        
        # Apply single class binarization if enabled
        if self.load_as_single_class:
            # background is 0, wildlife is 1
            label = 0 if original_label < 0 else 1
        else:
            # Multi-class: offset class id by 1 to account for background class (-1)
            label = original_label + 1 
        
        return crop_tensor, label
    
    def get_crop_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a crop.
        
        Args:
            idx: Index of the crop
            
        Returns:
            Dictionary with crop information
        """
        return self.crop_indices[idx].copy()
    
    def get_crops_by_class(self, class_id: int) -> List[int]:
        """
        Get indices of crops belonging to a specific class.
        
        Args:
            class_id: Class ID to filter by
            
        Returns:
            List of crop indices
        """
        return [i for i, crop_info in enumerate(self.crop_indices) if crop_info['label'] == class_id]
    
    def get_crops_by_type(self, crop_type: str) -> List[int]:
        """
        Get indices of crops by type (detection or random).
        
        Args:
            crop_type: Type of crop ('detection' or 'random')
            
        Returns:
            List of crop indices
        """
        return [i for i, crop_info in enumerate(self.crop_indices) if crop_info['crop_type'] == crop_type]
