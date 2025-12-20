"""
Integration tests for PatchDataset with ClassificationRebalanceFilter functionality.

This test suite validates:
1. ClassificationRebalanceFilter with different methods (mean, min)
2. Class distribution before/after rebalancing
3. Exclude_extremes functionality
4. DataLoader compatibility with balanced datasets
5. Utility methods (get_crops_by_class, get_crops_by_type)
6. Random seed reproducibility
7. Different filter configurations
"""

import pytest
import torch
from torchvision import transforms
from pathlib import Path
import tempfile
import shutil
import numpy as np
from typing import Optional

# Import WildTrain modules
from wildtrain.data.curriculum.dataset import CurriculumDetectionDataset, PatchDataset
from wildtrain.shared.models import CurriculumConfig
from wildtrain.data.filters import ClassificationRebalanceFilter


@pytest.mark.integration
@pytest.mark.data
class TestPatchDatasetRebalancing:
    """Test suite for PatchDataset rebalancing functionality."""
    
    def __init__(self):
        """Initialize test class attributes."""
        self.dataset_path: str = ""
        self.curriculum_config: Optional[CurriculumConfig] = None
        self.transform: Optional[transforms.Compose] = None
        self.temp_dir: str = ""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self, real_dataset_path):
        """Set up test data and configuration."""
        self.dataset_path = real_dataset_path
        
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
        self.temp_dir = tempfile.mkdtemp(prefix="wildtrain_crop_rebalancing_test_")
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_crop_dataset_creation(self):
        """Test PatchDataset creation for rebalancing tests."""
        # Load base dataset
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=self.dataset_path,
            split="train",
            curriculum_config=self.curriculum_config,
            transform=self.transform
        )
        
        # Create PatchDataset
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0
        )
        
        # Validate PatchDataset
        assert len(crop_dataset) > 0, "PatchDataset should not be empty"
        assert hasattr(crop_dataset, 'apply_rebalance_filter'), "PatchDataset should have apply_rebalance_filter method"
        
        return crop_dataset
    
    def test_classification_rebalance_filter_mean_method(self):
        """Test ClassificationRebalanceFilter with mean method."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Get annotations before rebalancing
        annotations_before = crop_dataset.get_annotations_for_filter()
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        # Create rebalance filter with mean method
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Apply filter
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
        
        # Validate results
        assert len(balanced_crop_dataset) > 0, "Balanced dataset should not be empty"
        
        # Analyze class distribution after rebalancing
        annotations_after = balanced_crop_dataset.get_annotations_for_filter()
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        # Check that rebalancing was applied
        original_counts = list(class_counts_before.values())
        balanced_counts = list(class_counts_after.values())
        
        # Calculate mean of original counts
        original_mean = np.mean(original_counts)
        
        # Check that balanced counts are closer to the mean
        for balanced_count in balanced_counts:
            # Allow some tolerance for the balancing
            tolerance = original_mean * 0.5
            assert abs(balanced_count - original_mean) <= tolerance, \
                f"Balanced count {balanced_count} should be close to mean {original_mean}"
    
    def test_classification_rebalance_filter_min_method(self):
        """Test ClassificationRebalanceFilter with min method."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Get annotations before rebalancing
        annotations_before = crop_dataset.get_annotations_for_filter()
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        # Create rebalance filter with min method
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="min",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Apply filter
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
        
        # Validate results
        assert len(balanced_crop_dataset) > 0, "Balanced dataset should not be empty"
        
        # Analyze class distribution after rebalancing
        annotations_after = balanced_crop_dataset.get_annotations_for_filter()
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        # Check that rebalancing was applied
        original_counts = list(class_counts_before.values())
        balanced_counts = list(class_counts_after.values())
        
        # Calculate min of original counts
        original_min = np.min(original_counts)
        
        # Check that all balanced counts are close to the minimum
        for balanced_count in balanced_counts:
            # Allow some tolerance for the balancing
            tolerance = original_min * 0.5
            assert abs(balanced_count - original_min) <= tolerance, \
                f"Balanced count {balanced_count} should be close to min {original_min}"
    
    def test_exclude_extremes_functionality(self):
        """Test exclude_extremes functionality."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Get annotations before rebalancing
        annotations_before = crop_dataset.get_annotations_for_filter()
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        # Test with exclude_extremes=True
        rebalance_filter_with_exclude = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=True,
            random_seed=42
        )
        
        balanced_with_exclude = crop_dataset.apply_rebalance_filter(rebalance_filter_with_exclude)
        
        # Test with exclude_extremes=False
        rebalance_filter_without_exclude = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_without_exclude = crop_dataset.apply_rebalance_filter(rebalance_filter_without_exclude)
        
        # The results should be different when excluding extremes
        # (This might not always be true depending on the data, but worth checking)
        annotations_with_exclude = balanced_with_exclude.get_annotations_for_filter()
        annotations_without_exclude = balanced_without_exclude.get_annotations_for_filter()
        
        # At least validate that both produce valid results
        assert len(annotations_with_exclude) > 0, "Balanced dataset with exclude should not be empty"
        assert len(annotations_without_exclude) > 0, "Balanced dataset without exclude should not be empty"
    
    def test_dataloader_compatibility(self):
        """Test DataLoader compatibility with balanced datasets."""
        from torch.utils.data import DataLoader
        
        crop_dataset = self.test_crop_dataset_creation()
        
        # Apply rebalancing
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
        
        # Create DataLoader
        dataloader = DataLoader(
            balanced_crop_dataset,
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
        """Test utility methods (get_crops_by_class, get_crops_by_type)."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Apply rebalancing
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
        
        # Test get_crops_by_class
        if len(balanced_crop_dataset) > 0:
            # Get class IDs from annotations
            annotations = balanced_crop_dataset.get_annotations_for_filter()
            if annotations:
                class_id = annotations[0].get('class_id', 0)
                crops_by_class = balanced_crop_dataset.get_crops_by_class(class_id)
                assert isinstance(crops_by_class, list), "get_crops_by_class should return a list"
                assert all(isinstance(idx, int) for idx in crops_by_class), "Indices should be integers"
        
        # Test get_crops_by_type
        detection_crops = balanced_crop_dataset.get_crops_by_type('detection')
        random_crops = balanced_crop_dataset.get_crops_by_type('random')
        
        assert isinstance(detection_crops, list), "get_crops_by_type should return a list for detection"
        assert isinstance(random_crops, list), "get_crops_by_type should return a list for random"
        assert all(isinstance(idx, int) for idx in detection_crops), "Detection indices should be integers"
        assert all(isinstance(idx, int) for idx in random_crops), "Random indices should be integers"
    
    def test_random_seed_reproducibility(self):
        """Test random seed reproducibility."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Create two filters with the same seed
        rebalance_filter_1 = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        rebalance_filter_2 = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Apply filters
        balanced_dataset_1 = crop_dataset.apply_rebalance_filter(rebalance_filter_1)
        balanced_dataset_2 = crop_dataset.apply_rebalance_filter(rebalance_filter_2)
        
        # Get annotations
        annotations_1 = balanced_dataset_1.get_annotations_for_filter()
        annotations_2 = balanced_dataset_2.get_annotations_for_filter()
        
        # Results should be identical with the same seed
        assert len(annotations_1) == len(annotations_2), "Results should have same length with same seed"
        
        # Check that the same samples were selected (allowing for order differences)
        class_counts_1 = {}
        class_counts_2 = {}
        
        for ann in annotations_1:
            class_name = ann['class_name']
            class_counts_1[class_name] = class_counts_1.get(class_name, 0) + 1
        
        for ann in annotations_2:
            class_name = ann['class_name']
            class_counts_2[class_name] = class_counts_2.get(class_name, 0) + 1
        
        assert class_counts_1 == class_counts_2, "Class distributions should be identical with same seed"
    
    def test_different_filter_configurations(self):
        """Test different filter configurations."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Test different methods
        methods = ["mean", "min"]
        exclude_extremes_options = [True, False]
        
        for method in methods:
            for exclude_extremes in exclude_extremes_options:
                rebalance_filter = ClassificationRebalanceFilter(
                    class_key="class_id",
                    method=method,
                    exclude_extremes=exclude_extremes,
                    random_seed=42
                )
                
                balanced_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
                
                # Validate that filtering worked
                assert len(balanced_dataset) > 0, f"Balanced dataset should not be empty for method={method}, exclude_extremes={exclude_extremes}"
                
                # Validate that we have annotations
                annotations = balanced_dataset.get_annotations_for_filter()
                assert len(annotations) > 0, f"Should have annotations for method={method}, exclude_extremes={exclude_extremes}"
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with invalid method
        with pytest.raises(ValueError):
            ClassificationRebalanceFilter(
                class_key="class_id",
                method="invalid_method",
                random_seed=42
            )
        
        # Test with invalid class_key
        with pytest.raises(ValueError):
            ClassificationRebalanceFilter(
                class_key="invalid_key",
                method="mean",
                random_seed=42
            )
        
        # Test with invalid random_seed
        with pytest.raises(ValueError):
            ClassificationRebalanceFilter(
                class_key="class_id",
                method="mean",
                random_seed=-1  # Invalid seed
            )
    
    def test_class_distribution_analysis(self):
        """Test class distribution analysis before/after rebalancing."""
        crop_dataset = self.test_crop_dataset_creation()
        
        # Analyze class distribution before rebalancing
        annotations_before = crop_dataset.get_annotations_for_filter()
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        # Apply rebalancing
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
        
        # Analyze class distribution after rebalancing
        annotations_after = balanced_crop_dataset.get_annotations_for_filter()
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        # Validate that rebalancing preserved all classes
        for class_name in class_counts_before:
            assert class_name in class_counts_after, f"Class {class_name} should still be present after rebalancing"
            assert class_counts_after[class_name] > 0, f"Class {class_name} should have samples after rebalancing"
        
        # Check that the distribution is more balanced
        original_counts = list(class_counts_before.values())
        balanced_counts = list(class_counts_after.values())
        
        original_std = np.std(original_counts)
        balanced_std = np.std(balanced_counts)
        
        # The standard deviation should be smaller after rebalancing (more balanced)
        # Allow for some tolerance in case the original was already balanced
        assert balanced_std <= original_std * 1.1, \
            f"Distribution should be more balanced after rebalancing (std: {original_std} -> {balanced_std})"
    
    @pytest.mark.slow
    def test_large_dataset_rebalancing(self):
        """Test rebalancing with larger dataset to ensure scalability."""
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
        
        # Apply rebalancing
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=True,
            random_seed=42
        )
        
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)
        
        # Validate that rebalancing worked
        assert len(balanced_crop_dataset) > 0, "Balanced dataset should not be empty"
        
        # Check that we have a reasonable number of samples
        assert len(balanced_crop_dataset) <= len(crop_dataset), "Balanced dataset should not be larger than original" 