"""
Integration tests for PatchDataset with ClusteringFilter functionality.

This test suite validates:
1. PatchDataset creation with mock detection data
2. ClusteringFilter application with different reduction factors
3. CropClusteringAdapter functionality and adapter pattern
4. DataLoader compatibility with clustered datasets
5. Utility methods (get_crops_by_class, get_crops_by_type, get_crop_info)
6. Class distribution analysis before/after clustering
7. Adapter information retrieval (get_filter_info)
"""

import pytest
import torch
from torchvision import transforms
from pathlib import Path
import tempfile
import shutil
from typing import Optional

# Import WildTrain modules
from wildtrain.data.curriculum.dataset import CurriculumDetectionDataset, PatchDataset
from wildtrain.shared.models import CurriculumConfig
from wildtrain.data.filters import ClusteringFilter, CropClusteringAdapter

DATASET_PATH = r"D:\workspace\data\demo-dataset"

@pytest.mark.integration
@pytest.mark.data
class TestPatchDatasetClustering:
    """Test suite for PatchDataset clustering functionality."""
    
    def __init__(self):
        """Initialize test class attributes."""
        self.dataset_path: str = DATASET_PATH
        self.curriculum_config: Optional[CurriculumConfig] = None
        self.transform: Optional[transforms.Compose] = None
        self.temp_dir: str = ""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self, real_dataset_path):
        """Set up test data and configuration."""
        
        # Create curriculum configuration
        self.curriculum_config = CurriculumConfig(
            enabled=True,
            type="difficulty",
            difficulty_strategy="linear",
            start_difficulty=0.0,
            end_difficulty=1.0,
            warmup_epochs=0,
        )
        
        # Create transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp(prefix="wildtrain_crop_clustering_test_")
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_crop_dataset_creation(self):
        """Test PatchDataset creation with mock detection data."""
        # Load base dataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        # Validate base dataset
        assert len(dataset) > 0, "Base dataset should not be empty"
        assert hasattr(dataset, 'classes'), "Dataset should have classes attribute"
        
        # Create PatchDataset
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0  # No annotations for cleaner crops
        )
        
        # Validate PatchDataset
        assert len(crop_dataset) > 0, "PatchDataset should not be empty"
        assert hasattr(crop_dataset, 'get_annotations_for_filter'), "PatchDataset should have get_annotations_for_filter method"
        
        # Test getting annotations for filtering
        annotations = crop_dataset.get_annotations_for_filter()
        assert isinstance(annotations, list), "Annotations should be a list"
        assert len(annotations) > 0, "Should have annotations for filtering"
        
        # Validate annotation format
        for ann in annotations:
            assert 'class_name' in ann, "Annotation should have class_name"
            assert 'roi_id' in ann, "Annotation should have roi_id"
    
    def test_clustering_filter_application(self):
        """Test ClusteringFilter application with different reduction factors."""
        # Load base dataset and create PatchDataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0
        )
        
        # Get annotations before clustering
        annotations_before = crop_dataset.get_annotations_for_filter()
        original_count = len(annotations_before)
        
        # Test different reduction factors
        reduction_factors = [0.3, 0.5, 0.7]
        
        for reduction_factor in reduction_factors:
            # Create clustering filter
            clustering_filter = ClusteringFilter(
                batch_size=32,
                reduction_factor=reduction_factor
            )
            
            # Create adapter
            clustering_adapter = CropClusteringAdapter(clustering_filter)
            
            # Apply filter
            clustered_crop_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)
            
            # Validate results
            assert len(clustered_crop_dataset) > 0, f"Clustered dataset should not be empty for reduction_factor={reduction_factor}"
            
            # Check that reduction was applied (allowing for some tolerance)
            expected_count = int(original_count * reduction_factor)
            actual_count = len(clustered_crop_dataset)
            
            # Allow 10% tolerance for clustering variations
            tolerance = int(original_count * 0.1)
            assert abs(actual_count - expected_count) <= tolerance, \
                f"Expected approximately {expected_count} crops, got {actual_count} for reduction_factor={reduction_factor}"
    
    def test_adapter_pattern_functionality(self):
        """Test CropClusteringAdapter functionality and adapter pattern."""
        # Load base dataset and create PatchDataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0
        )
        
        # Create clustering filter
        clustering_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.5
        )
        
        # Create adapter
        clustering_adapter = CropClusteringAdapter(clustering_filter)
        
        # Test adapter information
        adapter_info = clustering_adapter.get_filter_info()
        assert isinstance(adapter_info, dict), "Adapter info should be a dictionary"
        assert 'reduction_factor' in adapter_info, "Adapter info should contain reduction_factor"
        assert adapter_info['reduction_factor'] == 0.5, "Reduction factor should match"
        
        # Test that adapter wraps the original filter
        assert hasattr(clustering_adapter, 'clustering_filter'), "Adapter should have clustering_filter attribute"
        assert clustering_adapter.clustering_filter == clustering_filter, "Adapter should wrap the original filter"
    
    def test_dataloader_compatibility(self):
        """Test DataLoader compatibility with clustered datasets."""
        from torch.utils.data import DataLoader
        
        # Load base dataset and create PatchDataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0
        )
        
        # Apply clustering
        clustering_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.5
        )
        clustering_adapter = CropClusteringAdapter(clustering_filter)
        clustered_crop_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)
        
        # Create DataLoader
        dataloader = DataLoader(
            clustered_crop_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Test batch loading
        batch = next(iter(dataloader))
        assert isinstance(batch, (tuple, list)), "DataLoader should return tuple/list"
        assert len(batch) == 2, "DataLoader should return (crops, labels)"
        
        crops, labels = batch
        assert isinstance(crops, torch.Tensor), "Crops should be a tensor"
        assert isinstance(labels, torch.Tensor), "Labels should be a tensor"
        assert crops.shape[0] == labels.shape[0], "Batch size should match"
        assert crops.shape[1] == 3, "Crops should have 3 channels (RGB)"
        assert crops.shape[2] == 224, "Crops should have height 224"
        assert crops.shape[3] == 224, "Crops should have width 224"
    
    def test_utility_methods(self):
        """Test utility methods (get_crops_by_class, get_crops_by_type, get_crop_info)."""
        # Load base dataset and create PatchDataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0
        )
        
        # Apply clustering
        clustering_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.5
        )
        clustering_adapter = CropClusteringAdapter(clustering_filter)
        clustered_crop_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)
        
        # Test get_crops_by_class
        if len(dataset.classes) > 0:
            class_id = 0  # Test with first class
            crops_by_class = clustered_crop_dataset.get_crops_by_class(class_id)
            assert isinstance(crops_by_class, list), "get_crops_by_class should return a list"
            assert all(isinstance(idx, int) for idx in crops_by_class), "Indices should be integers"
        
        # Test get_crops_by_type
        detection_crops = clustered_crop_dataset.get_crops_by_type('detection')
        random_crops = clustered_crop_dataset.get_crops_by_type('random')
        
        assert isinstance(detection_crops, list), "get_crops_by_type should return a list for detection"
        assert isinstance(random_crops, list), "get_crops_by_type should return a list for random"
        assert all(isinstance(idx, int) for idx in detection_crops), "Detection indices should be integers"
        assert all(isinstance(idx, int) for idx in random_crops), "Random indices should be integers"
        
        # Test get_crop_info
        if len(clustered_crop_dataset) > 0:
            crop_info = clustered_crop_dataset.get_crop_info(0)
            assert isinstance(crop_info, dict), "get_crop_info should return a dictionary"
            assert 'crop_type' in crop_info, "Crop info should contain crop_type"
            assert 'class_id' in crop_info, "Crop info should contain class_id"
    
    def test_class_distribution_analysis(self):
        """Test class distribution analysis before/after clustering."""
        # Load base dataset and create PatchDataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0
        )
        
        # Analyze class distribution before clustering
        annotations_before = crop_dataset.get_annotations_for_filter()
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        # Apply clustering
        clustering_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.5
        )
        clustering_adapter = CropClusteringAdapter(clustering_filter)
        clustered_crop_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)
        
        # Analyze class distribution after clustering
        annotations_after = clustered_crop_dataset.get_annotations_for_filter()
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        # Validate that clustering preserved some samples from each class
        for class_name in class_counts_before:
            assert class_name in class_counts_after, f"Class {class_name} should still be present after clustering"
            assert class_counts_after[class_name] > 0, f"Class {class_name} should have samples after clustering"
            assert class_counts_after[class_name] <= class_counts_before[class_name], \
                f"Class {class_name} should not have more samples after clustering"
    
    def test_adapter_information_retrieval(self):
        """Test adapter information retrieval (get_filter_info)."""
        # Create clustering filter
        clustering_filter = ClusteringFilter(
            batch_size=32,
            reduction_factor=0.5
        )
        
        # Create adapter
        clustering_adapter = CropClusteringAdapter(clustering_filter)
        
        # Get filter information
        adapter_info = clustering_adapter.get_filter_info()
        
        # Validate adapter info structure
        assert isinstance(adapter_info, dict), "Adapter info should be a dictionary"
        assert 'reduction_factor' in adapter_info, "Adapter info should contain reduction_factor"
        assert 'batch_size' in adapter_info, "Adapter info should contain batch_size"
        assert 'filter_type' in adapter_info, "Adapter info should contain filter_type"
        
        # Validate values
        assert adapter_info['reduction_factor'] == 0.5, "Reduction factor should match"
        assert adapter_info['batch_size'] == 32, "Batch size should match"
        assert adapter_info['filter_type'] == 'ClusteringFilter', "Filter type should be correct"
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with invalid reduction factor
        with pytest.raises(ValueError):
            ClusteringFilter(
                batch_size=32,
                reduction_factor=1.5  # Invalid: > 1.0
            )
        
        # Test with invalid batch size
        with pytest.raises(ValueError):
            ClusteringFilter(
                batch_size=0,  # Invalid: <= 0
                reduction_factor=0.5
            )
        
        # Test with empty dataset (should handle gracefully)
        # This would be tested with an empty dataset if we had one
    
    @pytest.mark.slow
    def test_large_dataset_clustering(self):
        """Test clustering with larger dataset to ensure scalability."""
        # This test would use a larger mock dataset
        # For now, we'll test with the existing dataset but with more crops
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        # Create PatchDataset with more crops
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=5,  # More crops per image
            p_draw_annotations=0.0
        )
        
        # Apply clustering with aggressive reduction
        clustering_filter = ClusteringFilter(
            batch_size=64,  # Larger batch size
            reduction_factor=0.3  # More aggressive reduction
        )
        clustering_adapter = CropClusteringAdapter(clustering_filter)
        clustered_crop_dataset = crop_dataset.apply_clustering_filter(clustering_adapter)
        
        # Validate that clustering worked
        assert len(clustered_crop_dataset) > 0, "Clustered dataset should not be empty"
        assert len(clustered_crop_dataset) < len(crop_dataset), "Clustered dataset should be smaller" 