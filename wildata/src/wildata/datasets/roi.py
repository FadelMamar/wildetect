import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype
from torchvision.transforms.v2 import functional as F

from ..adapters.utils import read_image
from ..logging_config import get_logger
from ..pipeline.path_manager import PathManager

logger = get_logger("ROI_DATASET")


class ROIDataset(Dataset):
    """
    PyTorch Dataset for loading ROI datasets (images and labels) for a given split.

    Args:
        dataset_name (str): Name of the dataset.
        split (str): One of 'train', 'val', or 'test'.
        path_manager (PathManager): Instance for path resolution.
        transform (callable, optional): Optional transform to be applied on a sample.

    Example:
        >>> ds = ROIDataset(dataset_name="demo-dataset", split="train", root_data_directory="/path/to/data")
        >>> img, label = ds[0]
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        root_data_directory: Path | str,
        transform: Optional[Callable] = None,
        load_as_single_class: bool = False,
        background_class_name: str = "background",
        single_class_name: str = "wildlife",
        keep_classes: Optional[list[str]] = None,
        discard_classes: Optional[list[str]] = [
            "vegetation",
            "termite mound",
            "rocks",
            "other",
            "label",
        ],
        resample_function: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.path_manager = PathManager(Path(root_data_directory))
        self.transform = transform

        self.load_as_single_class = load_as_single_class
        self.single_class_name = single_class_name
        self.background_class_name = background_class_name
        self.keep_classes = keep_classes
        self.discard_classes = discard_classes
        self._single_class_mapping = {
            1: self.single_class_name,
            0: self.background_class_name,
        }
        self._multi_class_single_class_mapping: dict[int, int] = {}

        assert (
            (self.keep_classes is not None) + (self.discard_classes is not None) <= 1
        ), f"Cannot specify both keep_classes and discard_classes. keep_classes: {self.keep_classes}, discard_classes: {self.discard_classes}"

        # Resolve directories
        self.images_dir = self.path_manager.get_framework_split_image_dir(
            dataset_name, framework="roi", split=split
        )
        self.labels_dir = self.path_manager.get_framework_split_annotations_dir(
            dataset_name, framework="roi", split=split
        )

        # Load class mapping
        class_mapping_path = self.labels_dir / "class_mapping.json"
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
            assert isinstance(class_mapping, dict), "Class mapping must be a dictionary"
            self._class_mapping = {int(k): v for k, v in class_mapping.items()}

        # Load ROI labels
        roi_labels_path = self.labels_dir / "roi_labels.json"
        with open(roi_labels_path, "r", encoding="utf-8") as f:
            self.roi_labels = json.load(f)

        self._update_class_mapping()

        # Keep labels with class_id in the class mapping
        updated_roi_labels = []
        for label_info in self.roi_labels:
            class_id = label_info["class_id"]
            if int(class_id) in self._class_mapping.keys():
                updated_roi_labels.append(label_info)

        # updated roi_labels
        if self.load_as_single_class:
            for label_info in updated_roi_labels:
                label_info["class_id"] = self._multi_class_single_class_mapping[
                    label_info["class_id"]
                ]

        self.load_image = Compose([PILToTensor(), ToDtype(torch.float32, scale=True)])

        if resample_function:
            updated_roi_labels = resample_function(updated_roi_labels)

        logger.info(
            f"Loaded {len(updated_roi_labels)}/{len(self.roi_labels)} ROI samples for dataset:{dataset_name}; split:{split}."
        )

        self.roi_labels = updated_roi_labels

    @property
    def class_mapping(
        self,
    ):
        if self.load_as_single_class:
            return self._single_class_mapping
        return self._class_mapping

    def _update_class_mapping(self):
        if self.keep_classes:
            self._class_mapping = {
                k: v for k, v in self._class_mapping.items() if v in self.keep_classes
            }
            logger.debug(
                f"Keeping classes: {self.keep_classes}. Updated class mapping: {self._class_mapping}"
            )
        elif self.discard_classes:
            self._class_mapping = {
                k: v
                for k, v in self._class_mapping.items()
                if v not in self.discard_classes
            }
            logger.debug(
                f"Discarding classes: {self.discard_classes}. Updated class mapping: {self._class_mapping}"
            )

        if self.load_as_single_class:
            if self.background_class_name not in self._class_mapping.values():
                raise ValueError(
                    f"background class '{self.background_class_name}' not found in class mapping: {self._class_mapping}"
                )

            background_class_id = [
                k
                for k, v in self._class_mapping.items()
                if v == self.background_class_name
            ][0]
            single_class_id = 1
            self._multi_class_single_class_mapping = {
                k: single_class_id
                for k, v in self._class_mapping.items()
                if k != background_class_id
            }
            self._multi_class_single_class_mapping[background_class_id] = 0

    def __len__(self):
        return len(self.roi_labels)

    def get_label(self, idx) -> int:
        label_info = self.roi_labels[idx]
        label = label_info["class_id"]
        return label

    def get_image(self, idx: int) -> torch.Tensor:
        label_info = self.roi_labels[idx]
        img_path = (self.images_dir / label_info["file_name"]).as_posix()
        image = read_image(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return self.load_image(image)

    def get_image_path(self, idx: int) -> str:
        label_info = self.roi_labels[idx]
        img_path = (self.images_dir / label_info["file_name"]).as_posix()
        return img_path

    def __getitem__(self, idx: int):
        image = self.get_image(idx)
        label = self.get_label(idx)
        return image, torch.tensor([label]).int()


def load_all_roi_datasets(
    root_data_directory: Path,
    split: str,
    transform: Optional[dict[str, Callable]] = None,
    concat: bool = False,
    load_as_single_class: bool = False,
    background_class_name: str = "background",
    single_class_name: str = "wildlife",
    keep_classes: Optional[list[str]] = None,
    discard_classes: Optional[list[str]] = None,
    resample_function: Optional[Callable] = None,
) -> dict[str, ROIDataset] | ConcatDataset:
    """
    Load all available ROI datasets for a given split.
    Returns a dict mapping dataset_name to ROIDataset.
    Skips datasets that do not have the requested split.
    The transform argument should be a dict with keys 'train' and 'val'.
    The 'train' transform is used for the train split, and the 'val' transform is used for both val and test splits.
    If concat=True, returns (ConcatDataset, class_mapping) instead, after checking all class_mappings are identical.
    """
    path_manager = PathManager(root_data_directory)
    all_datasets = path_manager.list_framework_datasets(framework="roi")
    roi_datasets = {}
    for dataset_name in all_datasets:
        # Check if split exists (by checking for roi_labels.json in split dir)
        split_labels_dir = path_manager.get_framework_split_annotations_dir(
            dataset_name, framework="roi", split=split
        )
        roi_labels_path = split_labels_dir / "roi_labels.json"
        if not roi_labels_path.exists():
            continue
        # Select transform based on split
        split_transform = None
        if transform:
            if split == "train":
                split_transform = transform.get("train")
            else:
                split_transform = transform.get("val")
        try:
            ds = ROIDataset(
                dataset_name=dataset_name,
                split=split,
                root_data_directory=root_data_directory,
                transform=split_transform,
                load_as_single_class=load_as_single_class,
                background_class_name=background_class_name,
                single_class_name=single_class_name,
                keep_classes=keep_classes,
                discard_classes=discard_classes,
                resample_function=resample_function,
            )
            roi_datasets[dataset_name] = ds
        except Exception as e:
            # logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise ValueError(
                f"Error loading dataset {dataset_name}: {traceback.format_exc()}"
            )

    if concat and len(roi_datasets) > 1:
        # Check all class_mappings are identical
        class_mappings = [ds.class_mapping for ds in roi_datasets.values()]
        first_mapping = class_mappings[0]
        for mapping in class_mappings[1:]:
            if mapping != first_mapping:
                raise ValueError(
                    "Class mappings are not identical across datasets. Cannot concatenate."
                )
        concat_dataset = ConcatDataset(list(roi_datasets.values()))
        return concat_dataset

    elif concat and len(roi_datasets) == 1:
        # If only one dataset, return it directly
        return next(iter(roi_datasets.values()))

    return roi_datasets


def load_all_splits_concatenated(
    root_data_directory: Path,
    splits: list[str] = ["train", "val", "test"],
    transform: Optional[dict[str, Callable]] = None,
    load_as_single_class: bool = False,
    background_class_name: str = "background",
    single_class_name: str = "wildlife",
    keep_classes: Optional[list[str]] = None,
    discard_classes: Optional[list[str]] = None,
    resample_function: Optional[Callable] = None,
) -> Dict[str, ConcatDataset] | Dict[str, ROIDataset]:
    """
    Load and concatenate all available ROI datasets for each split.
    Returns a dictionary mapping split names to ConcatDataset objects.
    Only includes splits that have at least one dataset.
    Raises ValueError if class mappings are not identical for a split.
    """
    result = {}
    for split in splits:
        concat_dataset = load_all_roi_datasets(
            root_data_directory=root_data_directory,
            split=split,
            transform=transform,
            concat=True,
            load_as_single_class=load_as_single_class,
            background_class_name=background_class_name,
            single_class_name=single_class_name,
            keep_classes=keep_classes,
            discard_classes=discard_classes,
            resample_function=resample_function,
        )
        if not concat_dataset:
            continue
        result[split] = concat_dataset
    return result
