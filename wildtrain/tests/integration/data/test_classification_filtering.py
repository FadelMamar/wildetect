"""
Integration tests for ClassificationRebalanceFilter functionality.

This test suite validates:
1. ClassificationRebalanceFilter with synthetic annotation data
2. Different balancing strategies (mean, min)
3. Random seed reproducibility
4. Class distribution analysis
5. Edge cases (empty annotations, single class)
"""

import pytest
import numpy as np
from collections import Counter
import json
import tempfile
import os
from typing import List, Dict, Any, Optional

# Import WildTrain modules
from wildtrain.data.filters import ClassificationRebalanceFilter


@pytest.mark.integration
@pytest.mark.data
class TestClassificationFiltering:
    """Test suite for ClassificationRebalanceFilter functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Set up test data with synthetic annotations."""
        # Create imbalanced dataset (class 0: 100 samples, class 1: 50 samples, class 2: 25 samples)
        self.imbalanced_annotations: List[Dict[str, Any]] = []
        for i in range(100):  # Class 0: 100 samples
            self.imbalanced_annotations.append({
                "id": i,
                "file_name": f"image_{i:03d}.jpg",
                "class_id": 0,
                "class_name": "class_0"
            })
        
        for i in range(50):  # Class 1: 50 samples
            self.imbalanced_annotations.append({
                "id": i + 100,
                "file_name": f"image_{i + 100:03d}.jpg",
                "class_id": 1,
                "class_name": "class_1"
            })
        
        for i in range(25):  # Class 2: 25 samples
            self.imbalanced_annotations.append({
                "id": i + 150,
                "file_name": f"image_{i + 150:03d}.jpg",
                "class_id": 2,
                "class_name": "class_2"
            })
        
        # Create balanced dataset for comparison
        self.balanced_annotations: List[Dict[str, Any]] = []
        for i in range(30):  # 30 samples per class
            for class_id in range(3):
                self.balanced_annotations.append({
                    "id": i * 3 + class_id,
                    "file_name": f"balanced_image_{i * 3 + class_id:03d}.jpg",
                    "class_id": class_id,
                    "class_name": f"class_{class_id}"
                })
        
        # Create single class dataset
        self.single_class_annotations: List[Dict[str, Any]] = []
        for i in range(50):
            self.single_class_annotations.append({
                "id": i,
                "file_name": f"single_class_image_{i:03d}.jpg",
                "class_id": 0,
                "class_name": "class_0"
            })
        
        # Create empty dataset
        self.empty_annotations: List[Dict[str, Any]] = []
    
    def test_imbalanced_data_balancing_mean_method(self):
        """Test ClassificationRebalanceFilter with imbalanced data using mean method."""
        # Analyze class distribution before filtering
        class_ids_before = [ann["class_id"] for ann in self.imbalanced_annotations]
        class_counts_before = Counter(class_ids_before)
        
        print("Class distribution before filtering:", dict(class_counts_before))
        
        # Create and apply filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_annotations = filter_instance(self.imbalanced_annotations)
        
        # Analyze class distribution after filtering
        class_ids_after = [ann["class_id"] for ann in balanced_annotations]
        class_counts_after = Counter(class_ids_after)
        
        print("Class distribution after filtering:", dict(class_counts_after))
        
        # Validate results
        assert len(balanced_annotations) > 0, "Filtered annotations should not be empty"
        
        # Check that all classes are still present
        assert len(class_counts_after) == len(class_counts_before), "All classes should still be present"
        
        # Check that the distribution is more balanced
        original_counts = list(class_counts_before.values())
        balanced_counts = list(class_counts_after.values())
        
        original_std = np.std(original_counts)
        balanced_std = np.std(balanced_counts)
        
        # The standard deviation should be smaller after balancing
        assert balanced_std <= original_std, \
            f"Distribution should be more balanced (std: {original_std} -> {balanced_std})"
        
        # Check that we didn't lose too many samples
        assert len(balanced_annotations) >= min(original_counts), \
            "Should keep at least the minimum number of samples per class"
    
    def test_imbalanced_data_balancing_min_method(self):
        """Test ClassificationRebalanceFilter with imbalanced data using min method."""
        # Analyze class distribution before filtering
        class_ids_before = [ann["class_id"] for ann in self.imbalanced_annotations]
        class_counts_before = Counter(class_ids_before)
        
        print("Class distribution before filtering (min method):", dict(class_counts_before))
        
        # Create and apply filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="min",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_annotations = filter_instance(self.imbalanced_annotations)
        
        # Analyze class distribution after filtering
        class_ids_after = [ann["class_id"] for ann in balanced_annotations]
        class_counts_after = Counter(class_ids_after)
        
        print("Class distribution after filtering (min method):", dict(class_counts_after))
        
        # Validate results
        assert len(balanced_annotations) > 0, "Filtered annotations should not be empty"
        
        # Check that all classes are still present
        assert len(class_counts_after) == len(class_counts_before), "All classes should still be present"
        
        # Check that all classes have the same number of samples (min method)
        balanced_counts = list(class_counts_after.values())
        min_count = min(class_counts_after.values())
        
        for count in balanced_counts:
            assert count == min_count, f"All classes should have {min_count} samples with min method"
    
    def test_balanced_data_preservation(self):
        """Test that already balanced data is preserved."""
        # Analyze class distribution before filtering
        class_ids_before = [ann["class_id"] for ann in self.balanced_annotations]
        class_counts_before = Counter(class_ids_before)
        
        print("Class distribution before filtering (balanced):", dict(class_counts_before))
        
        # Create and apply filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_annotations = filter_instance(self.balanced_annotations)
        
        # Analyze class distribution after filtering
        class_ids_after = [ann["class_id"] for ann in balanced_annotations]
        class_counts_after = Counter(class_ids_after)
        
        print("Class distribution after filtering (balanced):", dict(class_counts_after))
        
        # Validate results
        assert len(balanced_annotations) > 0, "Filtered annotations should not be empty"
        
        # Check that the distribution remains balanced
        original_counts = list(class_counts_before.values())
        balanced_counts = list(class_counts_after.values())
        
        # The counts should be similar (allowing for small variations)
        for orig_count, bal_count in zip(original_counts, balanced_counts):
            assert abs(orig_count - bal_count) <= 2, \
                f"Balanced data should be preserved (original: {orig_count}, balanced: {bal_count})"
    
    def test_random_seed_reproducibility(self):
        """Test random seed reproducibility."""
        # Create two filters with the same seed
        filter_1 = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        filter_2 = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Apply filters
        result_1 = filter_1(self.imbalanced_annotations)
        result_2 = filter_2(self.imbalanced_annotations)
        
        # Results should be identical with the same seed
        assert len(result_1) == len(result_2), "Results should have same length with same seed"
        
        # Check class distributions
        class_counts_1 = Counter([ann["class_id"] for ann in result_1])
        class_counts_2 = Counter([ann["class_id"] for ann in result_2])
        
        assert class_counts_1 == class_counts_2, "Class distributions should be identical with same seed"
        
        # Test with different seeds
        filter_3 = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=123
        )
        
        result_3 = filter_3(self.imbalanced_annotations)
        
        # Results might be different with different seeds
        # (This is expected behavior, not a requirement)
        print(f"Results with seed 42: {len(result_1)} samples")
        print(f"Results with seed 123: {len(result_3)} samples")
    
    def test_exclude_extremes_functionality(self):
        """Test exclude_extremes functionality."""
        # Test with exclude_extremes=True
        filter_with_exclude = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=True,
            random_seed=42
        )
        
        result_with_exclude = filter_with_exclude(self.imbalanced_annotations)
        
        # Test with exclude_extremes=False
        filter_without_exclude = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        result_without_exclude = filter_without_exclude(self.imbalanced_annotations)
        
        # Both should produce valid results
        assert len(result_with_exclude) > 0, "Result with exclude should not be empty"
        assert len(result_without_exclude) > 0, "Result without exclude should not be empty"
        
        # The results might be different
        print(f"With exclude_extremes=True: {len(result_with_exclude)} samples")
        print(f"With exclude_extremes=False: {len(result_without_exclude)} samples")
    
    def test_single_class_data(self):
        """Test handling of single class data."""
        # Analyze class distribution before filtering
        class_ids_before = [ann["class_id"] for ann in self.single_class_annotations]
        class_counts_before = Counter(class_ids_before)
        
        print("Class distribution before filtering (single class):", dict(class_counts_before))
        
        # Create and apply filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_annotations = filter_instance(self.single_class_annotations)
        
        # Analyze class distribution after filtering
        class_ids_after = [ann["class_id"] for ann in balanced_annotations]
        class_counts_after = Counter(class_ids_after)
        
        print("Class distribution after filtering (single class):", dict(class_counts_after))
        
        # Validate results
        assert len(balanced_annotations) > 0, "Filtered annotations should not be empty"
        
        # Check that the single class is preserved
        assert len(class_counts_after) == 1, "Single class should be preserved"
        assert 0 in class_counts_after, "Class 0 should still be present"
        
        # The number of samples should be reasonable
        assert class_counts_after[0] <= len(self.single_class_annotations), \
            "Should not have more samples than original"
    
    def test_empty_annotations(self):
        """Test handling of empty annotations."""
        # Create and apply filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Should handle empty annotations gracefully
        balanced_annotations = filter_instance(self.empty_annotations)
        
        # Result should be empty
        assert len(balanced_annotations) == 0, "Empty input should produce empty output"
    
    def test_different_class_keys(self):
        """Test filter with different class key names."""
        # Create annotations with different class key
        annotations_with_category = [
            {"id": i, "file_name": f"image_{i:03d}.jpg", "category_id": i % 3, "class_name": f"class_{i % 3}"}
            for i in range(30)
        ]
        
        # Test with category_id as class key
        filter_instance = ClassificationRebalanceFilter(
            class_key="category_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        balanced_annotations = filter_instance(annotations_with_category)
        
        # Validate results
        assert len(balanced_annotations) > 0, "Filtered annotations should not be empty"
        
        # Check that all classes are still present
        class_ids_after = [ann["category_id"] for ann in balanced_annotations]
        class_counts_after = Counter(class_ids_after)
        
        assert len(class_counts_after) == 3, "All 3 classes should be present"
        
        # The number of samples should be reasonable
        assert class_counts_after[0] <= len(annotations_with_category), \
            "Should not have more samples than original"
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with invalid method - validation happens in __call__, not __init__
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="invalid_method",
            random_seed=42
        )
        
        # The error should be raised when calling the filter, not during initialization
        with pytest.raises(ValueError, match="Invalid method: invalid_method"):
            filter_instance(self.imbalanced_annotations)
        
        # Test with invalid class_key (key that doesn't exist in annotations)
        filter_instance = ClassificationRebalanceFilter(
            class_key="nonexistent_key",
            method="mean",
            random_seed=42
        )
        
        # Should handle missing key gracefully or raise appropriate error
        try:
            result = filter_instance(self.imbalanced_annotations)
            # If it doesn't raise an error, the result should be empty or handle the missing key
            assert isinstance(result, list), "Result should be a list"
        except (KeyError, ValueError):
            # This is also acceptable behavior
            pass
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create larger synthetic dataset
        large_annotations = []
        for i in range(1000):
            class_id = i % 5  # 5 classes
            large_annotations.append({
                "id": i,
                "file_name": f"image_{i:03d}.jpg",
                "class_id": class_id,
                "class_name": f"class_{class_id}"
            })
        
        # Create and apply filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Time the operation
        import time
        start_time = time.time()
        balanced_annotations = filter_instance(large_annotations)
        end_time = time.time()
        
        # Validate results
        assert len(balanced_annotations) > 0, "Filtered annotations should not be empty"
        
        # Check performance (should complete within reasonable time)
        processing_time = end_time - start_time
        assert processing_time < 10.0, f"Processing should be fast (took {processing_time:.2f}s)"
        
        print(f"Processed {len(large_annotations)} annotations in {processing_time:.2f}s")
        
        # Check that the distribution is more balanced
        class_counts_before = Counter([ann["class_id"] for ann in large_annotations])
        class_counts_after = Counter([ann["class_id"] for ann in balanced_annotations])
        
        original_std = np.std(list(class_counts_before.values()))
        balanced_std = np.std(list(class_counts_after.values()))
        
        assert balanced_std <= original_std, \
            f"Distribution should be more balanced (std: {original_std} -> {balanced_std})"
    
    def test_filter_statistics(self):
        """Test that filter provides useful statistics."""
        # Create filter
        filter_instance = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=False,
            random_seed=42
        )
        
        # Apply filter
        balanced_annotations = filter_instance(self.imbalanced_annotations)
        
        # Check that filter has useful attributes
        assert hasattr(filter_instance, 'class_key'), "Filter should have class_key attribute"
        assert hasattr(filter_instance, 'method'), "Filter should have method attribute"
        assert hasattr(filter_instance, 'exclude_extremes'), "Filter should have exclude_extremes attribute"
        assert hasattr(filter_instance, 'random_seed'), "Filter should have random_seed attribute"
        
        # Validate attribute values
        assert filter_instance.class_key == "class_id"
        assert filter_instance.method == "mean"
        assert filter_instance.exclude_extremes == False
        assert filter_instance.random_seed == 42 