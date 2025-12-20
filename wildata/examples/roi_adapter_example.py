#!/usr/bin/env python3
"""
Example script demonstrating the ROI adapter functionality.

This script shows how to:
1. Use the ROI adapter to convert object detection to classification
2. Handle images with and without annotations
3. Use custom callback functions for ROI generation
4. Configure different ROI extraction parameters
"""

from pathlib import Path

from wildata.pipeline import PathManager,Loader,FrameworkDataManager
from wildata.config import ROIConfig


def main():
    """Example of saving ROI data to disk."""
    print("\n=== Saving ROI Data to Disk ===")
    
    ROOT = Path(r"D:\workspace\data\demo-dataset")
    SOURCE_PATH = r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\data_config.yaml"

    roi_config = ROIConfig(random_roi_count=1,
                                    roi_box_size=128,
                                    min_roi_size=32,
                                    dark_threshold=0.5,
                                    roi_callback=None)

    loader = Loader()
    split = "train"
    dataset_name = "savmap"

    dataset_info, split_coco_data = loader.load(SOURCE_PATH, "yolo", dataset_name, bbox_tolerance=5, split_name=split)

    path_manager = PathManager(ROOT)
    framework_data_manager = FrameworkDataManager(path_manager)
    framework_data_manager.create_roi_format(dataset_name=dataset_name,
                                                coco_data=split_coco_data[split],
                                                split=split,
                                                roi_config=roi_config,
                                                draw_original_bboxes=True,
                                                )
        
    
    print(f"Saved ROI data to {ROOT}")
    

if __name__ == "__main__":
    print("ROI Adapter Examples")
    print("=" * 50)
    
    try:
        # Run examples
        # example_basic_usage()
        # example_with_callback()
        main()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc() 