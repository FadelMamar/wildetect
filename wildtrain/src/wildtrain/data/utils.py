from torchvision import transforms as T
import torch
from wildata.datasets.detection import load_detection_dataset
from wildata.pipeline.path_manager import PathManager
from pathlib import Path
import supervision as sv
import traceback

from ..utils.logging import get_logger
from ..utils.io import read_image


logger = get_logger(__name__)


def load_image(path: str) -> torch.Tensor:
    image = read_image(path).convert("RGB")
    return T.PILToTensor()(image)

def load_all_detection_datasets(
    root_data_directory: str,
    split: str,
) -> sv.DetectionDataset:
    """
    Load all available detection datasets for a given split.
    Returns a list of dictionaries containing (dataset, class_mapping, dataset_name).
    Skips datasets that do not have the requested split.
    
    Args:
        root_data_directory (str): Root directory containing the datasets
        split (str): Dataset split to load ('train', 'val', 'test')
        
        
    Returns:
        List[Tuple[DetectionDataset, dict, str]]: List of (dataset, class_mapping, dataset_name) tuples
    """
    path_manager = PathManager(Path(root_data_directory))
    all_datasets = path_manager.list_datasets()
    detection_datasets = []

    logger.info(f"Loading datasets: {all_datasets}, split: {split}")
    
    for dataset_name in all_datasets:
        # Check if split exists by checking for annotations file
        annotations_file = path_manager.get_dataset_split_annotations_file(
            dataset_name, split
        )
        if not annotations_file.exists():
            continue
            
        try:
            dataset, class_mapping = load_detection_dataset(
                root_data_directory=root_data_directory,
                dataset_name=dataset_name,
                split=split
            )
            detection_datasets.append(dataset)
            logger.info(
                f"Loaded detection dataset: {dataset_name} for split: {split} with {len(dataset)} samples"
            )
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # merge all datasets into one
    merged_dataset = sv.DetectionDataset.merge(detection_datasets)
    logger.info(
        f"Successfully loaded {len(detection_datasets)} detection datasets for split: {split}"
    )
    return merged_dataset