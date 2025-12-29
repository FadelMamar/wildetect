"""Custom IoU Tuner with Label Studio SDK Integration.

This script reads duplicate annotations from a CSV file and retrieves task data
directly from Label Studio server via SDK to run IoU threshold optimization.

The approach treats one duplicate as the ground truth annotation and all duplicates
(including itself) as predictions, optimizing to improve prediction quality.

Usage:
    python scripts/custom_iou_tunner.py --csv_path="animal duplicates.csv" --n_trials=50
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from wildetect.core.visualization.labelstudio_manager import LabelStudioManager
from wildetect.utils.iou_tuner import IoUTuner
from wildata.converters.labelstudio.labelstudio_schemas import Result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CustomIoUTunerConfig:
    """Configuration for custom IoU tuner."""

    csv_path: str
    base_url: str = "http://localhost:8080"
    n_trials: int = 50
    class_agnostic: bool = True
    merging_iou_range: Tuple[float, float] = (0.1, 0.9)
    matching_iou_range: Tuple[float, float] = (0.1, 0.7)
    conf_threshold_range: Tuple[float, float] = (0.0, 0.5)
    run_name: str = "custom-iou-tuner"


class LabelStudioDataLoader:
    """Load task data from Label Studio based on CSV duplicate information."""

    def __init__(
        self,
        csv_path: str,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ):
        """Initialize the data loader.

        Args:
            csv_path: Path to the CSV file with duplicate information
            base_url: Label Studio server URL
            api_key: Label Studio API key (defaults to LABEL_STUDIO_API_KEY env var)
        """
        self.csv_path = csv_path
        self.api_key = api_key or os.environ.get("LABEL_STUDIO_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Label Studio API key not provided. "
                "Set LABEL_STUDIO_API_KEY environment variable or pass api_key parameter."
            )

        self.ls_manager = LabelStudioManager(
            url=base_url,
            api_key=self.api_key,
            download_resources=False,
        )

        # Load and validate CSV
        self.df_csv = pd.read_csv(csv_path)
        self._validate_csv()

        # Rename 'image' column to 'task_id' for IoUTuner compatibility
        self.df_csv = self.df_csv.rename(columns={"image": "task_id"})

        logger.info(f"Loaded CSV with {len(self.df_csv)} rows")
        logger.info(f"Unique tasks: {self.df_csv['task_id'].nunique()}")
        logger.info(f"Unique duplicate groups: {self.df_csv['duplicate'].nunique()}")

    def _validate_csv(self):
        """Validate required columns exist in CSV."""
        required_columns = ["image", "bounding box", "species", "duplicate"]
        missing = [col for col in required_columns if col not in self.df_csv.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def get_unique_task_ids(self) -> list:
        """Extract unique task IDs from CSV."""
        return self.df_csv["task_id"].unique().tolist()

    def get_valid_bbox_ids(self) -> set:
        """Get set of valid bounding box IDs from CSV."""
        return set(self.df_csv["bounding box"].unique())

    def get_task_species_map(self) -> dict:
        """Get mapping of task_id to set of valid species labels (normalized)."""
        task_species = {}
        for _, row in self.df_csv.iterrows():
            task_id = row["task_id"]
            species = str(row["species"]).strip().lower()
            if task_id not in task_species:
                task_species[task_id] = set()
            task_species[task_id].add(species)
        return task_species

    def _parse_result_to_row(self, result: "Result", task_id: int) -> dict:
        """Convert a Result object to a DataFrame row.

        Args:
            result: Parsed Result schema object
            task_id: Label Studio task ID

        Returns:
            Dict with pixel coordinates and metadata for IoUTuner
        """
        # Use the Result schema's built-in methods
        x_pixel, y_pixel, width_pixel, height_pixel = result.get_pixel_bbox()

        # Determine source from origin field
        if result.origin:
            source = (
                "prediction"
                if result.origin in ["prediction", "prediction-changed"]
                else "annotation"
            )
        else:
            # Default to annotation if origin not specified
            source = "annotation"

        # Get score - annotations default to 1.0
        score = result.score if result.score is not None else 1.0

        return {
            "task_id": task_id,
            "x_pixel": x_pixel,
            "y_pixel": y_pixel,
            "width_pixel": width_pixel,
            "height_pixel": height_pixel,
            "label": result.label or "unknown",
            "score": score,
            "origin": source,
            "bbox_id": result.id,
        }


    def fetch_task_data(self, task_id: int, valid_bbox_ids: set, valid_species: set) -> list:
        """Fetch task data from Label Studio and filter by valid bbox IDs or species.

        Args:
            task_id: Label Studio task ID
            valid_bbox_ids: Set of valid bbox IDs from CSV
            valid_species: Set of valid species labels (normalized) for this task

        Returns:
            List of parsed Result rows matching valid_bbox_ids or valid_species
        """
        

        try:
            task = self.ls_manager.get_task(task_id)
            if task is None:
                logger.warning(f"Task {task_id} not found")
                return []

            results = []

            # Process annotations
            if hasattr(task, "annotations") and task.annotations:
                for ann in task.annotations:
                    if hasattr(ann, "result") and ann.result:
                        for r in ann.result:
                            # Skip non-rectanglelabels or results missing required fields
                            if r.get("type") != "rectanglelabels":
                                continue
                            if "original_width" not in r or "original_height" not in r:
                                continue
                            try:
                                parsed_result = Result.model_validate(r)
                                # Include if bbox ID matches OR label matches species
                                result_label = (parsed_result.label or "").strip().lower()
                                if r.get("id") in valid_bbox_ids or result_label in valid_species:
                                    row = self._parse_result_to_row(parsed_result, task_id)
                                    results.append(row)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to parse annotation result {r.get('id')} in task {task_id}: {e}"
                                )

            # Process predictions
            if hasattr(task, "predictions") and task.predictions:
                for pred in task.predictions:
                    if hasattr(pred, "result") and pred.result:
                        for r in pred.result:
                            # Skip non-rectanglelabels or results missing required fields
                            if r.get("type") != "rectanglelabels":
                                continue
                            if "original_width" not in r or "original_height" not in r:
                                continue
                            try:
                                parsed_result = Result.model_validate(r)
                                # Include if bbox ID matches OR label matches species
                                result_label = (parsed_result.label or "").strip().lower()
                                if r.get("id") in valid_bbox_ids or result_label in valid_species:
                                    row = self._parse_result_to_row(parsed_result, task_id)
                                    # Override source for predictions
                                    row["origin"] = "prediction"
                                    results.append(row)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to parse prediction result {r.get('id')} in task {task_id}: {e}"
                                )

            return results

        except Exception as e:
            logger.warning(f"Failed to fetch task {task_id}: {e}")
            return []


    def _get_bbox_rows_for_duplicate_group(
        self, group_df: pd.DataFrame, results_cache: dict
    ) -> Tuple[list, list]:
        """Process a duplicate group to create annotation and prediction rows.

        For each duplicate group:
        - First bounding box becomes the annotation (ground truth)
        - All bounding boxes (including first) become predictions

        Args:
            group_df: DataFrame subset for one duplicate group
            results_cache: Cache mapping bbox_id to parsed result row

        Returns:
            Tuple of (annotation_rows, prediction_rows)
        """
        annotation_rows = []
        prediction_rows = []

        # Get all bounding boxes for this group
        bbox_ids = group_df["bounding box"].tolist()

        if not bbox_ids:
            return annotation_rows, prediction_rows

        for idx, bbox_id in enumerate(bbox_ids):
            # Look up the pre-parsed result by bbox_id
            row = results_cache.get(bbox_id)
            if not row:
                logger.warning(f"Bbox {bbox_id} not found in results cache")
                continue

            # Make a copy to avoid modifying the cached row
            row = row.copy()

            # First bbox is annotation (ground truth), all are predictions
            if idx == 0:
                annotation_rows.append(row)

            # All bboxes (including first) go into predictions
            prediction_rows.append(row)

        return annotation_rows, prediction_rows

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert all duplicate groups to annotation and prediction DataFrames.

        Returns:
            Tuple of (df_annotations, df_predictions) with required columns for IoUTuner
        """
        # Get valid bbox IDs from CSV
        valid_bbox_ids = self.get_valid_bbox_ids()
        logger.info(f"Valid bbox IDs from CSV: {len(valid_bbox_ids)}")

        # Build task to species mapping
        task_species_map = self.get_task_species_map()
        logger.info(f"Task species map built for {len(task_species_map)} tasks")

        # Get unique task IDs and fetch all task data upfront
        task_ids = self.get_unique_task_ids()
        logger.info(f"Fetching {len(task_ids)} tasks from Label Studio...")

        # Build results cache indexed by bbox_id
        results_cache = {}
        for task_id in tqdm(task_ids, desc="Fetching tasks"):
            valid_species = task_species_map.get(task_id, set())
            results = self.fetch_task_data(task_id, valid_bbox_ids, valid_species)
            for row in results:
                bbox_id = row.get("bbox_id")
                if bbox_id:
                    results_cache[bbox_id] = row

        logger.info(f"Successfully fetched {len(results_cache)} matching results")

        # Process each duplicate group
        all_annotation_rows = []
        all_prediction_rows = []

        for dup_id, group_df in tqdm(
            self.df_csv.groupby("duplicate"),
            desc="Processing duplicate groups",
        ):
            ann_rows, pred_rows = self._get_bbox_rows_for_duplicate_group(
                group_df, results_cache
            )
            all_annotation_rows.extend(ann_rows)
            all_prediction_rows.extend(pred_rows)

        # Create DataFrames
        df_annotations = pd.DataFrame(all_annotation_rows)
        df_predictions = pd.DataFrame(all_prediction_rows)

        # Ensure required columns exist
        required_cols = [
            "task_id",
            "x_pixel",
            "y_pixel",
            "width_pixel",
            "height_pixel",
            "origin",
            "score",
            "label",
        ]

        for col in required_cols:
            if col not in df_annotations.columns:
                df_annotations[col] = None
            if col not in df_predictions.columns:
                df_predictions[col] = None

        logger.info(f"Created {len(df_annotations)} annotation rows")
        logger.info(f"Created {len(df_predictions)} prediction rows")

        return df_annotations, df_predictions


def main(
    csv_path: str = "animal-duplicates.csv",
    base_url: str = "http://localhost:8080",
    n_trials: int = 50,
    class_agnostic: bool = True,
    run_name: str = "custom-iou-tuner",
):
    """Run custom IoU tuner with Label Studio data.

    Args:
        csv_path: Path to CSV file with duplicate annotations
        base_url: Label Studio server URL
        n_trials: Number of Optuna optimization trials
        class_agnostic: Whether to treat all classes as one
        run_name: Name for the Optuna study
    """
    print("=" * 60)
    print("Custom IoU Tuner - Label Studio Integration")
    print("=" * 60)

    # Load data from Label Studio
    loader = LabelStudioDataLoader(csv_path=csv_path, base_url=base_url)
    df_annotations, df_predictions = loader.to_dataframes()
    stem = Path(csv_path).stem
    df_annotations.to_csv(Path(csv_path).with_stem(stem + "_annotations"), index=False)
    df_predictions.to_csv(Path(csv_path).with_stem(stem + "_predictions"), index=False)

    if df_annotations.empty or df_predictions.empty:
        logger.error("No data loaded. Check CSV and Label Studio connection.")
        return

    print(f"\nDataFrame summary:")
    print(f"  - Annotation rows: {len(df_annotations)}")
    print(f"  - Prediction rows: {len(df_predictions)}")
    print(f"  - Unique tasks (annotations): {df_annotations['task_id'].nunique()}")
    print(f"  - Unique tasks (predictions): {df_predictions['task_id'].nunique()}")

    exit(1)

    print("\n" + "=" * 60)
    print(f"Running IoUTuner optimization ({n_trials} trials)")
    print("=" * 60)

    # Initialize and run IoUTuner
    tuner = IoUTuner(
        df_annotations=df_annotations,
        df_predictions=df_predictions,
        merging_iou_range=(0.0, 1.0),
        matching_iou_range=(0.0, 1.0),
        n_trials=n_trials,
        class_agnostic=class_agnostic,
        overlap_metrics=["iou", "ios"],
    )

    result = tuner.run(run_name=run_name)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in result.items():
        if k != "study":
            print(f"{k}: {v}")

    return result


if __name__ == "__main__":
    import fire

    fire.Fire(main)
