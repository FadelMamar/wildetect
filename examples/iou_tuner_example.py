"""
Example script to test IoUTuner with Label Studio annotations.

Can use synthetic data with duplicates demo, or real Label Studio JSON exports.
"""

import logging
import os
import tempfile
import pandas as pd
from wildetect.utils.iou_tuner import IoUTuner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Path to your Label Studio JSON file - update this to your file
LS_JSON_PATH = r"D:\workspace\repos\wildetect\project-200-at-2025-12-27-16-33-97d6912d.json"


def load_from_labelstudio(json_path: str) -> pd.DataFrame:
    """Load annotations from Label Studio JSON export."""
    from wildata.converters import LabelStudioParser

    parser = LabelStudioParser.from_file(json_path)
    print(parser.get_summary())
    
    df_annotations = parser.to_dataframe()
    return df_annotations


def create_synthetic_with_duplicates():
    """Create synthetic data demonstrating duplicate handling.
    
    Scenario: Same buffalo detected in consecutive frames (images 1 and 2).
    Without duplicate handling: 2 TPs (overcounting)
    With duplicate handling: 1 TP (correct - it's the same animal)
    """
    records = []
    
    # Task 1: Buffalo in frame 1 - GT and prediction
    records.append({
        "task_id": 1, "result_id": "gt1_t1", "label": "buffalo",
        "x_pixel": 100.0, "y_pixel": 100.0, "width_pixel": 50.0, "height_pixel": 50.0,
        "origin": "manual", "score": None,
    })
    records.append({
        "task_id": 1, "result_id": "pred1_t1", "label": "buffalo",
        "x_pixel": 102.0, "y_pixel": 102.0, "width_pixel": 48.0, "height_pixel": 48.0,
        "origin": "prediction", "score": 0.95,
    })
    
    # Task 2: SAME buffalo appears in frame 2 (duplicate)
    records.append({
        "task_id": 2, "result_id": "gt1_t2", "label": "buffalo",
        "x_pixel": 120.0, "y_pixel": 110.0, "width_pixel": 50.0, "height_pixel": 50.0,
        "origin": "manual", "score": None,
    })
    records.append({
        "task_id": 2, "result_id": "pred1_t2", "label": "buffalo",
        "x_pixel": 122.0, "y_pixel": 108.0, "width_pixel": 48.0, "height_pixel": 52.0,
        "origin": "prediction", "score": 0.92,
    })
    
    # Task 3: Different animal - elephant
    records.append({
        "task_id": 3, "result_id": "gt1_t3", "label": "elephant",
        "x_pixel": 200.0, "y_pixel": 200.0, "width_pixel": 100.0, "height_pixel": 80.0,
        "origin": "manual", "score": None,
    })
    records.append({
        "task_id": 3, "result_id": "pred1_t3", "label": "elephant",
        "x_pixel": 205.0, "y_pixel": 198.0, "width_pixel": 95.0, "height_pixel": 82.0,
        "origin": "prediction", "score": 0.88,
    })
    
    df = pd.DataFrame(records)
    
    # Create duplicates CSV - marks pred1_t1 and pred1_t2 as same animal (group 1)
    duplicates_data = [
        {"season": "Wet", "camp": "K1", "Project ID": 1, "image": 1, 
         "bounding box": "pred1_t1", "species": "buffalo", "duplicate": 1},
        {"season": "Wet", "camp": "K1", "Project ID": 1, "image": 2, 
         "bounding box": "pred1_t2", "species": "buffalo", "duplicate": 1},
    ]
    df_duplicates = pd.DataFrame(duplicates_data)
    
    # Save to temp file
    dup_csv_path = os.path.join(tempfile.gettempdir(), "synthetic_duplicates.csv")
    df_duplicates.to_csv(dup_csv_path, index=False)
    
    return df, dup_csv_path


def main(use_labelstudio: bool = True, with_duplicates: bool = True, n_trials: int = 20):
    """Run IoUTuner with Label Studio data or synthetic data.
    
    Args:
        use_labelstudio: If True, load from Label Studio JSON
        with_duplicates: If True, use synthetic data with duplicates demo
        n_trials: Number of Optuna trials
    """
    print("=" * 60)
    print("IoUTuner Example - Duplicate Handling Demo")
    print("=" * 60)
    
    duplicates_csv_path = None
    
    if use_labelstudio:
        print(f"\nLoading from Label Studio: {LS_JSON_PATH}")
        df = load_from_labelstudio(LS_JSON_PATH)
    elif with_duplicates:
        print("\nUsing synthetic data WITH duplicates")
        df, duplicates_csv_path = create_synthetic_with_duplicates()
        print(f"  Duplicates CSV: {duplicates_csv_path}")
    else:
        print("\nUsing synthetic data WITHOUT duplicates")
        df, _ = create_synthetic_with_duplicates()
    
    print(f"\nDataFrame summary:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Tasks: {df['task_id'].nunique()}")
    print(f"  - Predictions: {len(df[df['origin'] != 'manual'])}")
    print(f"  - Groundtruth: {len(df[df['origin'] == 'manual'])}")
    
    print("\n" + "=" * 60)
    print(f"Running IoUTuner optimization ({n_trials} trials)")
    print("=" * 60)
    
    tuner = IoUTuner(
        df_annotations=df,
        duplicates_csv_path=duplicates_csv_path,
        nms_iou_range=(0.3, 0.8),
        match_iou_range=(0.3, 0.7),
        n_trials=n_trials,
    )
    
    result = tuner.run()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best NMS IoU threshold:   {result['best_nms_iou_threshold']:.4f}")
    print(f"Best Match IoU threshold: {result['best_match_iou_threshold']:.4f}")
    print(f"Best F1-score:            {result['best_f1_score']:.4f}")
    print(f"Precision:                {result['best_precision']:.4f}")
    print(f"Recall:                   {result['best_recall']:.4f}")
    
    if duplicates_csv_path:
        print("\n" + "-" * 40)
        print("With duplicates collapsed:")
        print("  - Buffalo in frames 1 & 2 = 1 animal (not 2)")
        print("  - Expected: 2 unique animals (1 buffalo, 1 elephant)")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
