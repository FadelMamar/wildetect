"""
Example demonstrating PatchDataset with ClusteringFilter using adapter pattern.

This example shows how to:
1. Create a PatchDataset from detection annotations
2. Use CropClusteringAdapter to apply ClusteringFilter to crop annotations
3. Use the clustered dataset with PyTorch DataLoader
4. Analyze clustering results and crop diversity
"""

import torch
from torchvision import transforms
import traceback
from torch.utils.data import DataLoader

from wildtrain.data.curriculum.dataset import CurriculumDetectionDataset, PatchDataset
from wildtrain.shared.models import CurriculumConfig
from wildtrain.data.filters import CropClusteringFilter, ClassificationRebalanceFilter


def main():
    """Main example function."""
    
    print("=" * 70)
    print("PatchDataset with ClusteringFilter (Adapter Pattern) Example")
    print("=" * 70)
    
    # 1. Create curriculum configuration
    curriculum_config = CurriculumConfig(
        enabled=True,
        type="difficulty",
        difficulty_strategy="linear",
        start_difficulty=0.0,
        end_difficulty=1.0,
        warmup_epochs=0,
    )

    # 2. Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    try:
        # 3. Load base dataset
        print("📁 Loading base dataset...")
        dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=r"D:\workspace\data\demo-dataset",  # Your images directory
            split="train",
            curriculum_config=curriculum_config,
            transform=transform
        )

        print(f"✅ Base dataset loaded successfully!")
        print(f"✅ Number of samples: {len(dataset)}")
        print(f"✅ Classes: {dataset.classes}")

        # 4. Create PatchDataset
        print("\n🌾 Creating PatchDataset...")
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0  # No annotations for cleaner crops
        )

        print(f"✅ PatchDataset created successfully!")
        print(f"✅ Number of crops: {len(crop_dataset)}")

        # 5. Analyze class distribution before clustering
        print("\n📊 Class distribution before clustering:")
        annotations_before = crop_dataset.get_annotations_for_filter()
        
        # Count crops per class
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        for class_name, count in class_counts_before.items():
            print(f"   {class_name}: {count} crops")

        # 6. Create ClusteringFilter and adapter
        print("\n🔄 Creating ClusteringFilter with adapter...")
        
        # Create the base clustering filter
        clustering_filter = CropClusteringFilter(
            crop_dataset=crop_dataset,
            batch_size=8,
            reduction_factor=0.5  # Keep 50% of crops
        )

        #rebalance_filter = ClassificationRebalanceFilter(
        #    class_key="class_id",
        #    method="mean",
        #     exclude_extremes=True,
        #     random_seed=42
        # )
        
        # Apply filter to create clustered dataset
        clustered_crop_dataset = (crop_dataset
                                    #.apply_rebalance_filter(rebalance_filter)
                                    .apply_clustering_filter(clustering_filter)
                                    )

        print(f"✅ Clustered PatchDataset created successfully!")
        print(f"✅ Number of crops after clustering: {len(clustered_crop_dataset)}")

        # 7. Analyze class distribution after clustering
        print("\n📊 Class distribution after clustering:")
        annotations_after = clustered_crop_dataset.get_annotations_for_filter()
        
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        for class_name, count in class_counts_after.items():
            print(f"   {class_name}: {count} crops")

        # 8. Test DataLoader compatibility
        print("\n🧪 Testing DataLoader compatibility...")
        
        # Create DataLoader with clustered dataset
        dataloader = DataLoader(
            clustered_crop_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Set to 0 for this example
        )

        # Test a few batches
        print("Testing batch loading:")
        for batch_idx, (crops, labels) in enumerate(dataloader):
            print(f"   Batch {batch_idx + 1}: crops shape={crops.shape}, labels={labels.tolist()}")
            if batch_idx >= 2:  # Only test first 3 batches
                break

        # 9. Demonstrate utility methods
        print("\n🔧 Testing utility methods:")
        
        # Get crops by class (assuming class_id 0 is wildlife)
        if len(dataset.classes) > 0:
            wildlife_crops = clustered_crop_dataset.get_crops_by_class(0)
            print(f"   Wildlife crops: {len(wildlife_crops)} indices")
        
        # Get crops by type
        detection_crops = clustered_crop_dataset.get_crops_by_type('detection')
        random_crops = clustered_crop_dataset.get_crops_by_type('random')
        print(f"   Detection crops: {len(detection_crops)} indices")
        print(f"   Random crops: {len(random_crops)} indices")
        
        # Get detailed info for first crop
        if len(clustered_crop_dataset) > 0:
            crop_info = clustered_crop_dataset.get_crop_info(0)
            print(f"   First crop info keys: {list(crop_info.keys())}")


    except FileNotFoundError:
        print(traceback.format_exc())
        raise

    except Exception as e:
        print(f"❌ Error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()