"""
Example demonstrating PatchDataset with ClassificationRebalanceFilter.

This example shows how to:
1. Create a PatchDataset from detection annotations
2. Apply ClassificationRebalanceFilter to balance class distribution
3. Use the balanced dataset with PyTorch DataLoader
4. Analyze class distributions before and after rebalancing
"""

import torch
from torchvision import transforms
from tqdm import tqdm
import traceback
from pathlib import Path

# Import our modules
from wildtrain.data.curriculum.dataset import CurriculumDetectionDataset, PatchDataset
from wildtrain.shared.models import CurriculumConfig
from wildtrain.data.filters import ClassificationRebalanceFilter


def main():
    """Main example function."""
    
    print("PatchDataset with Rebalancing Example")
    print("=" * 60)
    
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
        print("\n Creating PatchDataset...")
        crop_dataset = PatchDataset(
            dataset=dataset,
            crop_size=224,
            max_tn_crops=1,
            p_draw_annotations=0.0,  # No annotations for cleaner crops
            load_as_single_class=True,
            background_class_name="background",
            single_class_name="wildlife",
            keep_classes=None,
            discard_classes=["rocks","vegetation","other","termite mound"]
        )

        print(f"✅ PatchDataset created successfully!")
        print(f"✅ Number of patches: {len(crop_dataset)}")

        # 5. Analyze class distribution before rebalancing
        print("\n📊 Class distribution before rebalancing:")
        annotations_before = crop_dataset.get_annotations_for_filter()
        
        # Count crops per class
        class_counts_before = {}
        for ann in annotations_before:
            class_name = ann['class_name']
            class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1
        
        for class_name, count in class_counts_before.items():
            print(f"   {class_name}: {count} crops")

        # 6. Create and apply ClassificationRebalanceFilter
        print("\n🔄 Applying ClassificationRebalanceFilter...")
        
        # Create filter with mean balancing
        rebalance_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=True,
            random_seed=42
        )

        # Apply filter to create balanced dataset
        balanced_crop_dataset = crop_dataset.apply_rebalance_filter(rebalance_filter)

        print(f"✅ Balanced PatchDataset created successfully!")
        print(f"✅ Number of patches after balancing: {len(balanced_crop_dataset)}")

        # 7. Analyze class distribution after rebalancing
        print("\n📊 Class distribution after rebalancing:")
        annotations_after = balanced_crop_dataset.get_annotations_for_filter()
        
        class_counts_after = {}
        for ann in annotations_after:
            class_name = ann['class_name']
            class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1
        
        for class_name, count in class_counts_after.items():
            print(f"   {class_name}: {count} crops")

        # 8. Test DataLoader compatibility
        print("\n🧪 Testing DataLoader compatibility...")
        
        from torch.utils.data import DataLoader
        
        # Create DataLoader with balanced dataset
        dataloader = DataLoader(
            balanced_crop_dataset,
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
                
        # Get crops by type
        detection_crops = balanced_crop_dataset.get_crops_by_type('detection')
        random_crops = balanced_crop_dataset.get_crops_by_type('random')
        print(f"   Detection crops: {len(detection_crops)} indices")
        print(f"   Random crops: {len(random_crops)} indices")
        
        # Get detailed info for first crop
        if len(balanced_crop_dataset) > 0:
            crop_info = balanced_crop_dataset.get_crop_info(0)
            print(f"   First crop info keys: {list(crop_info.keys())}")

        # 10. Demonstrate different filter methods
        print("\n🔄 Testing different filter methods:")
        
        # Min-based filter
        min_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="min",
            exclude_extremes=False,
            random_seed=42
        )
        
        min_balanced = crop_dataset.apply_rebalance_filter(min_filter)
        print(f"   Min-balanced dataset: {len(min_balanced)} crops")
        
        # Mean-based filter (exclude extremes)
        mean_filter = ClassificationRebalanceFilter(
            class_key="class_id",
            method="mean",
            exclude_extremes=True,
            random_seed=42
        )
        
        mean_balanced = crop_dataset.apply_rebalance_filter(mean_filter)
        print(f"   Mean-balanced dataset: {len(mean_balanced)} crops")

        # 11. Show annotation format
        print("\n📝 Example annotation format:")
        if len(annotations_after) > 0:
            example_ann = annotations_after[0]
            print("   Example annotation:")
            for key, value in example_ann.items():
                print(f"     {key}: {value}")

        print("\n✅ All tests completed successfully!")

    except FileNotFoundError:
        print("⚠️  Dataset files not found. This is expected if you don't have the data files.")
        print("   Replace the paths with your actual dataset paths.")
        print("\n   Expected paths:")
        print("   - Images: D:\\workspace\\data\\demo-dataset\\datasets")
        print("   - Annotations: D:\\workspace\\data\\demo-dataset\\datasets\\savmap\\annotations\\train.json")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
