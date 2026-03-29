"""
Example script to test IoUTuner with Label Studio annotations.

Can use synthetic data with duplicates demo, or real Label Studio JSON exports.
"""

import logging
import pandas as pd
from wildetect.utils.iou_tuner import IoUTuner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Path to your Label Studio JSON file - update this to your file
LS_JSON_PATH = r"project-200-at-2025-12-27-16-33-97d6912d.json"


def load_from_labelstudio(json_path: str) -> pd.DataFrame:
    """Load annotations from Label Studio JSON export."""
    from wildata.converters import LabelStudioParser

    parser = LabelStudioParser.from_file(json_path)
    print(parser.get_summary())
    
    df_annotations = parser.to_dataframe()
    return df_annotations

def main( n_trials: int = 50):
    """Run IoUTuner with Label Studio data or synthetic data.
    
    Args:
        n_trials: Number of Optuna trials
    """
    print("=" * 60)
    print("IoUTuner Example - Duplicate Handling Demo")
    print("=" * 60)
    
    duplicates_csv_path = None
    
    df = load_from_labelstudio(LS_JSON_PATH)
    
    print(f"\nDataFrame summary:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Tasks: {df['task_id'].nunique()}")
    print(f"  - Predictions: {len(df[df['source'] == 'prediction'])}")
    print(f"  - Groundtruth: {len(df[df['source'] == 'annotation'])}")
    
    print("\n" + "=" * 60)
    print(f"Running IoUTuner optimization ({n_trials} trials)")
    print("=" * 60)
    
    tuner = IoUTuner(
        df_annotations=df[df['source'] == 'annotation'],
        df_predictions=df[df['source'] == 'prediction'],
        duplicates_csv_path=duplicates_csv_path,
        merging_iou_range=(0., 1.),
        matching_iou_range=(0., 1.),
        n_trials=n_trials,
        class_agnostic=True,
        overlap_metrics=["iou", "ios"]
    )
    
    result = tuner.run(run_name="iou-tuner-demo")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"{k}: {v}")
    
    if duplicates_csv_path:
        print("\n" + "-" * 40)
        print("With duplicates collapsed:")
        print("  - Buffalo in frames 1 & 2 = 1 animal (not 2)")
        print("  - Expected: 2 unique animals (1 buffalo, 1 elephant)")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
