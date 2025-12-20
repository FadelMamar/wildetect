"""
Example demonstrating ROI dataset clustering using ClusteringFilter.

This example shows how to:
1. Load ROI datasets using load_all_splits_concatenated
2. Apply ClusteringFilter to reduce dataset size while maintaining diversity
3. Use the clustered dataset with PyTorch DataLoader
4. Analyze clustering results and dataset diversity
"""

import torch
from torchvision import transforms
import traceback
from torch.utils.data import DataLoader
from pathlib import Path

from wildata.datasets.roi import load_all_splits_concatenated, ROIDataset
from wildtrain.data.filters.algorithms import ClusteringFilter
from wildtrain.models.feature_extractor import FeatureExtractor


def main():
    """Main example function."""
    
    print("=" * 70)
    print("ROI Dataset Clustering Example")
    print("=" * 70)
    
    # 1. Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # 2. Load ROI datasets for all splits
        print("📁 Loading ROI datasets...")
        datasets = load_all_splits_concatenated(
            root_data_directory=r"D:\workspace\data\demo-dataset",  # Your data directory
            splits=["train", "val"],  # Load train and validation splits
            transform={"train": transform, "val": transform},
            load_as_single_class=True,  # Convert to binary classification
            background_class_name="background",
            single_class_name="wildlife"
        )

        print(f"✅ ROI datasets loaded successfully!")
        print(f"✅ Available splits: {list(datasets.keys())}")
        
        # Focus on training split for clustering example
        train_dataset = datasets["train"]
        print(f"✅ Training dataset size: {len(train_dataset)}")
        print(f"✅ Classes: {train_dataset.class_mapping}")

        # 3. Analyze class distribution before clustering
        print("\n📊 Class distribution before clustering:")
        class_counts_before = {}
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            class_id = label.item()
            class_name = train_dataset.class_mapping[class_id]
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        for class_name, count in class_counts_before.items():
            print(f"   {class_name}: {count} samples")

        # 4. Create ClusteringFilter
        print("\n🔄 Creating ClusteringFilter...")
        
        # Initialize feature extractor for computing embeddings
        feature_extractor = FeatureExtractor()
        
        # Create the clustering filter
        clustering_filter = ClusteringFilter(
            feature_extractor=feature_extractor,
            batch_size=8,
            reduction_factor=0.5  # Keep 50% of samples
        )

        # 5. Apply clustering filter
        print("\n🔍 Applying clustering filter...")
        
        # Get all samples for filtering
        all_samples = []
        for i in range(len(train_dataset)):
            image, label = train_dataset[i]
            sample_info = {
                "index": i,
                "class_id": label.item(),
                "class_name": train_dataset.class_mapping[label.item()],
                "file_name": train_dataset.get_image_path(i) if hasattr(train_dataset, 'get_image_path') else f"sample_{i}"
            }
            all_samples.append(sample_info)
        
        print(f"Processing {len(all_samples)} images...")
        
        # Apply clustering filter
        filtered_samples = clustering_filter(all_samples)
        
        print(f"✅ Clustering filter applied successfully!")
        print(f"✅ Samples after clustering: {len(filtered_samples)}")
        print(f"✅ Reduction: {len(all_samples) - len(filtered_samples)} samples removed")

        # 6. Create filtered dataset
        print("\n📦 Creating filtered dataset...")
        
        # Create indices for filtered samples
        filtered_indices = [sample["index"] for sample in filtered_samples]
        
        # Create a subset of the original dataset
        from torch.utils.data import Subset
        filtered_dataset = Subset(train_dataset, filtered_indices)
        
        print(f"✅ Filtered dataset created successfully!")
        print(f"✅ Filtered dataset size: {len(filtered_dataset)}")

        # 7. Analyze class distribution after clustering
        print("\n📊 Class distribution after clustering:")
        class_counts_after = {}
        for i in range(len(filtered_dataset)):
            _, label = filtered_dataset[i]
            class_id = label.item()
            class_name = train_dataset.class_mapping[class_id]
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        for class_name, count in class_counts_after.items():
            print(f"   {class_name}: {count} samples")

        # 8. Test DataLoader compatibility
        print("\n🧪 Testing DataLoader compatibility...")
        
        # Create DataLoader with filtered dataset
        dataloader = DataLoader(
            filtered_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Set to 0 for this example
        )

        # Test a few batches
        print("Testing batch loading:")
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"   Batch {batch_idx + 1}: images shape={images.shape}, labels={labels.tolist()}")
            if batch_idx >= 2:  # Only test first 3 batches
                break

        # 9. Demonstrate utility methods and analysis
        print("\n🔧 Dataset analysis and utility methods:")
        
        # Analyze clustering quality
        print("   Clustering quality analysis:")
        print(f"     - Original dataset size: {len(train_dataset)}")
        print(f"     - Filtered dataset size: {len(filtered_dataset)}")
        print(f"     - Reduction factor: {len(filtered_dataset) / len(train_dataset):.2f}")
        
        # Check class balance preservation
        print("   Class balance preservation:")
        for class_name in class_counts_before.keys():
            before_count = class_counts_before[class_name]
            after_count = class_counts_after.get(class_name, 0)
            preservation_ratio = after_count / before_count if before_count > 0 else 0
            print(f"     - {class_name}: {before_count} → {after_count} ({preservation_ratio:.2f})")
        
        # Sample some filtered samples
        print("\n📸 Sample filtered samples:")
        for i in range(min(3, len(filtered_dataset))):
            image, label = filtered_dataset[i]
            class_name = train_dataset.class_mapping[label.item()]
            print(f"   Sample {i}: class={class_name}, image shape={image.shape}")

        # 10. Save clustering results
        print("\n💾 Saving clustering results...")
        
        # Create results directory
        results_dir = Path("results/roi_clustering")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save clustering summary
        clustering_summary = {
            "original_size": len(train_dataset),
            "filtered_size": len(filtered_dataset),
            "reduction_factor": len(filtered_dataset) / len(train_dataset),
            "class_distribution_before": class_counts_before,
            "class_distribution_after": class_counts_after,
            "filtered_indices": filtered_indices
        }
        
        import json
        with open(results_dir / "clustering_summary.json", "w") as f:
            json.dump(clustering_summary, f, indent=2)
        
        print(f"✅ Clustering results saved to {results_dir}")

    except FileNotFoundError:
        print(traceback.format_exc())
        print("❌ Error: Data directory not found. Please update the root_data_directory path.")
        raise

    except Exception as e:
        print(f"❌ Error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
