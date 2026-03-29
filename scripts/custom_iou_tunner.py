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

from wildetect.core.visualization.geographic import visualize_geographic_bounds
from wildetect.core.visualization.labelstudio_manager import LabelStudioManager
from wildetect.core.data import DroneImage
from wildetect.core.config_models import FlightSpecs
from wildetect.core.flight import GeographicMerger,CentroidProximityRemovalStrategy,GPSOverlapStrategy
from wildetect.utils.iou_tuner import IoUTuner
from wildata.converters.labelstudio.labelstudio_schemas import Result, Task

import fire

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
        self.df_csv = self.df_csv.rename(columns={"image": "task_id","bounding box":"bbox_id"})
        self._validate_csv()

        logger.info(f"Loaded CSV with {len(self.df_csv)} rows")
        logger.info(f"Unique tasks: {self.df_csv['task_id'].nunique()}")
        logger.info(f"Unique duplicate groups: {self.df_csv['duplicate'].nunique()}")

    def _validate_csv(self):
        """Validate required columns exist in CSV."""
        required_columns = ["task_id", "bbox_id", "species", "duplicate"]
        missing = [col for col in required_columns if col not in self.df_csv.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def get_unique_task_ids(self) -> list:
        """Extract unique task IDs from CSV."""
        return self.df_csv["task_id"].unique().tolist()

    def get_valid_bbox_ids(self) -> set:
        """Get set of valid bbox_id IDs from CSV."""
        return set(self.df_csv["bbox_id"].unique())

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

    def _parse_result_to_row(self, result: Result, task_id: int) -> dict:
        """Convert a Result object to a DataFrame row.

        Args:
            result: Parsed Result schema object
            task_id: Label Studio task ID

        Returns:
            Dict with pixel coordinates and metadata for IoUTuner
        """
        # Use the Result schema's built-in methods
        x_pixel, y_pixel, width_pixel, height_pixel = result.get_pixel_bbox()
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
            "origin": str(result.origin),
            "bbox_id": result.id,
        }

    def fetch_task_data(self, task_id: int, valid_bbox_ids: set, valid_species: set = set()) -> list:
        """Fetch task data from Label Studio and filter by valid bbox IDs or species.

        Args:
            task_id: Label Studio task ID
            valid_bbox_ids: Set of valid bbox IDs from CSV
            valid_species: Set of valid species labels (normalized) for this task

        Returns:
            List of parsed Result rows matching valid_bbox_ids or valid_species
        """

        try:
            task: Task = self.ls_manager.get_task(task_id,as_sdk_task=False)
            task = task.filter(valid_bbox_ids=valid_bbox_ids,valid_species=valid_species)
            results = []
            task_image_path = task.image_path
            for ann in task.annotations:
                for r in ann.result:                   
                    try:
                        row = self._parse_result_to_row(r, task_id)
                        row["image_path"] = task_image_path
                        results.append(row)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse annotation result {r.id} in task {task_id}: {e}"
                        )

            return results

        except Exception as e:
            logger.warning(f"Failed to fetch task {task_id}: {e}")
            return []

    def to_dataframe(self) -> pd.DataFrame:
        """Fetch task data from Label Studio and return as DataFrame.

        Returns:
            DataFrame with bounding box data for all valid bbox IDs
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
            #valid_species = task_species_map.get(task_id, set())
            results = self.fetch_task_data(task_id, valid_bbox_ids,)
            for row in results:
                bbox_id = row.get("bbox_id")
                if bbox_id:
                    results_cache[bbox_id] = row

        logger.info(f"Successfully fetched {len(results_cache)} matching results")

        # Create DataFrame directly from cache values
        df = pd.DataFrame(list(results_cache.values()))

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
            "image_path",
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        df = df.merge(self.df_csv.drop('task_id', axis=1),on='bbox_id',how='outer')

        logger.info(f"Created {len(df)} rows")
        
        return df


def load_detections(csv_path: str = "animal-duplicates.csv",
    base_url: str = "http://localhost:8080",):

    # Load data from Label Studio
    loader = LabelStudioDataLoader(csv_path=csv_path, base_url=base_url)
    df = loader.to_dataframe()
    stem = Path(csv_path).stem
    df.to_csv(Path(csv_path).with_stem(stem + "_detections"), index=False)

    if df.empty:
        logger.error("No data loaded. Check CSV and Label Studio connection.")
        return

    print(f"\nDataFrame summary:")
    print(f"  - Rows: {len(df)}")
    print(f"  - Unique tasks: {df['task_id'].nunique()}")


def load_task_as_droneimage(task_id: int, base_url: str = "http://localhost:8080",):
    """Load a task from Label Studio as a DroneImage object."""

    ls_manager = LabelStudioManager(
    url=base_url,
    api_key='4f3c25bad9334596c5b2c3b270a2d3105c8b5d4a',
    download_resources=False,
    )

    task = ls_manager.get_task(task_id,as_sdk_task=False)
    flight_specs = FlightSpecs(sensor_height=24,focal_length=35,flight_height=180)
    droneimage = DroneImage.from_ls_task(task,flight_specs=flight_specs)

    print(droneimage)


def visualize_duplicates(csv_path: str = "animal-duplicates.csv",
    base_url: str = "http://localhost:8080"):
    """Visualize duplicates on a map."""

    ls_manager = LabelStudioManager(
        url=base_url,
        api_key='4f3c25bad9334596c5b2c3b270a2d3105c8b5d4a',
        download_resources=False,
    )
    flight_specs = FlightSpecs(sensor_height=24,focal_length=35,flight_height=180)
    df = pd.read_csv(csv_path).rename(columns={"image": "task_id","bounding box":"bbox_id"})
    droneimages = []
    for task_id in map(int, df['task_id']):
        task = ls_manager.get_task(task_id,as_sdk_task=False)    
        droneimage = DroneImage.from_ls_task(task,flight_specs=flight_specs)
        droneimage.predictions = droneimage.annotations # Set predictions to annotations for duplicate removal tuning
        droneimages.append(droneimage)

    visualize_geographic_bounds(droneimages,output_path="duplicates_visualization.html")

def load_remove_duplicates(csv_path: str = "animal-duplicates.csv",
    base_url: str = "http://localhost:8080",):
    """Load duplicates from CSV as DroneImage objects."""
    
    #loader = LabelStudioDataLoader(csv_path=csv_path, base_url=base_url)
    #df = loader.to_dataframe()
    df = pd.read_csv(csv_path).rename(columns={"image": "task_id","bounding box":"bbox_id"})
    df['task_id'] = df['task_id'].apply(int)

    ls_manager = LabelStudioManager(
        url=base_url,
        api_key='4f3c25bad9334596c5b2c3b270a2d3105c8b5d4a',
        download_resources=False,
    )
    flight_specs = FlightSpecs(sensor_height=24,focal_length=35,flight_height=180)

    merger = GeographicMerger()
    
    for group, df_group in df.groupby("duplicate"):
        print(f"Group {group}: {len(df_group)} rows")
        task_ids = set(df_group['task_id'])
        print("Task ids: ",task_ids)

        try:
            images = []
            for task_id in task_ids:
                task = ls_manager.get_task(task_id,as_sdk_task=False)
                task = task.filter(valid_bbox_ids=set(df_group['bbox_id'].unique()))
                droneimage = DroneImage.from_ls_task(task,flight_specs=flight_specs)
                droneimage.predictions = droneimage.annotations # Set predictions to annotations for duplicate removal tuning
                images.append(droneimage)        
            images = merger.run(images,iou_threshold=0.5,min_overlap_threshold=0.2)
        except Exception as e:
            print(e)     

        break


def run_duplicate_tuner(
    csv_path: str = "animal-duplicates.csv",
    base_url: str = "http://localhost:8080",
    verbose: bool = False,
    n_trials: int = 150,
    iou_min: float = -1.0,
    iou_max: float = 0.9,
    overlap_min: float = 0.0,
    overlap_max: float = 0.5,
    sensor_height:int=24, 
    focal_length:int=35, 
    flight_height:int=180,
    run_name: str = "duplicate-tuner",
    optuna_storage: str = "sqlite:///duplicate-tuner.db",
    optuna_load_if_exists: bool = True,
):
    """Run hyperparameter tuning to optimize duplicate removal thresholds.
    
    Args:
        csv_path: Path to CSV file with duplicate annotations
        base_url: Label Studio server URL
        n_trials: Number of Optuna optimization trials
        iou_min: Minimum value for iou_threshold search range
        iou_max: Maximum value for iou_threshold search range
        overlap_min: Minimum value for min_overlap_threshold search range
        overlap_max: Maximum value for min_overlap_threshold search range
    """
    from wildetect.utils.duplicate_removal_tuner import DuplicateRemovalTuner
    
    print("=" * 60)
    print("Duplicate Removal Hyperparameter Tuning")
    print("=" * 60)
    
    ls_manager = LabelStudioManager(
        url=base_url,
        api_key='4f3c25bad9334596c5b2c3b270a2d3105c8b5d4a',
        download_resources=False,
    )
    flight_specs = FlightSpecs(sensor_height=sensor_height, focal_length=focal_length, flight_height=flight_height)
    
    tuner = DuplicateRemovalTuner(
        csv_path=csv_path,
        ls_manager=ls_manager,
        flight_specs=flight_specs,
        iou_threshold_range=(iou_min, iou_max),
        min_overlap_threshold_range=(overlap_min, overlap_max),
        n_trials=n_trials,
        verbose=verbose,
    )
    result = tuner.run(optuna_load_if_exists=optuna_load_if_exists,
                        run_name=run_name,
                        optuna_storage=optuna_storage
                    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best iou_threshold: {result['best_iou_threshold']:.4f}")
    print(f"Best min_overlap_threshold: {result['best_min_overlap_threshold']:.4f}")
    print(f"Best MAE: {result['best_mae']:.4f}")
    print(f"Removal accuracy: {result['removal_accuracy']:.1%}")
    print(f"Groups evaluated: {result['total_groups']}")
    print("=" * 60)
    
    #return result

if __name__ == "__main__":
    fire.Fire()
