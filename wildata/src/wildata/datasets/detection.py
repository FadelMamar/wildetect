import json
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import supervision as sv
from supervision.dataset.core import DetectionDataset

from ..logging_config import get_logger
from ..pipeline import PathManager

logger = get_logger("DETECTION_DATASET")


def _load_data(
    images_directory_path: str,
    annotations_path: str,
    source_format: str = "coco",
    force_mask: bool = False,
    is_obb: bool = False,
    data_yaml_path: Optional[str] = None,
):
    if source_format == "coco":
        return DetectionDataset.from_coco(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path,
        )
    elif source_format == "yolo":
        if data_yaml_path is None:
            raise ValueError("data_yaml_path is required for YOLO format")

        return DetectionDataset.from_yolo(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_path,
            data_yaml_path=data_yaml_path,
            force_masks=force_mask,
            is_obb=is_obb,
        )
    else:
        raise ValueError(f"Invalid format: {source_format}")


def load_detection_dataset(root_data_directory: str, dataset_name: str, split: str):
    path_manager = PathManager(Path(root_data_directory))
    # images_dir = path_manager.get_dataset_split_images_dir(dataset_name, split)

    annotations_file = path_manager.get_dataset_split_annotations_file(
        dataset_name, split
    )
    dataset = _load_data(
        str(path_manager.data_dir),
        annotations_path=str(annotations_file),
        source_format="coco",
    )

    with open(annotations_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        class_mapping = {cat["id"]: cat["name"] for cat in data["categories"]}

    return dataset, class_mapping
