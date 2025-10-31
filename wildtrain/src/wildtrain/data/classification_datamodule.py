import os
from typing import Any, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, ConcatDataset
import lightning as L
from torchvision import transforms as T
from wildata.datasets.roi import load_all_splits_concatenated, ROIDataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
from ..utils.logging import get_logger
from .filters import ClassificationRebalanceFilter
from .curriculum.dataset import PatchDataset, CurriculumDetectionDataset
from .curriculum.mixins import CurriculumDataModuleMixin
from ..shared.models import CurriculumConfig
from omegaconf import DictConfig, OmegaConf
from ..utils.transforms import create_transforms

logger = get_logger(__name__)


def compute_dataset_stats(
    dataset: Union[ROIDataset, ConcatDataset],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation of images in a dataset.

    Args:
        dataset: ROIDataset or ConcatDataset containing images
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        max_samples: Maximum number of samples to use (None for all samples)

    Returns:
        Tuple of (mean, std) tensors with shape (C,) where C is number of channels
    """
    # Create a dataloader with the existing dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Initialize accumulators
    mean_acc = torch.zeros(3)  # Assuming RGB images
    std_acc = torch.zeros(3)
    total_pixels = 0
    num_samples = 0

    logger.info("Computing dataset statistics...")
    for batch in tqdm(dataloader, desc="Computing mean/std"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # Assume first element is images
        else:
            images = batch

        if not isinstance(images, torch.Tensor):
            continue

        # Ensure images are in the right format (B, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

        batch_size_actual = images.size(0)

        # Compute mean for this batch
        batch_mean = images.mean(dim=[0, 2, 3])  # Mean across batch, height, width
        batch_pixels = images.size(0) * images.size(2) * images.size(3)

        # Accumulate mean
        mean_acc += batch_mean * batch_pixels
        total_pixels += batch_pixels
        num_samples += batch_size_actual

    # Compute final mean
    mean = mean_acc / total_pixels

    # Reset for std computation
    logger.info("Computing standard deviation...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    for batch in tqdm(dataloader, desc="Computing std"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        if not isinstance(images, torch.Tensor):
            continue

        if images.dim() == 3:
            images = images.unsqueeze(0)

        batch_size_actual = images.size(0)

        # Compute variance for this batch
        batch_var = ((images - mean.view(1, 3, 1, 1)) ** 2).mean(dim=[0, 2, 3])
        batch_pixels = images.size(0) * images.size(2) * images.size(3)

        # Accumulate variance
        std_acc += batch_var * batch_pixels
        num_samples += batch_size_actual

    # Compute final std
    std = torch.sqrt(std_acc / total_pixels)

    logger.info(f"Dataset statistics computed from {num_samples} samples:")

    return mean, std


class ClassificationDataModule(L.LightningDataModule, CurriculumDataModuleMixin):
    """
    Unified LightningDataModule for multiclass image classification.
    
    Supports two dataset types:
    1. Pre-computed crops (ROI datasets)
    2. Dynamic crop extraction from detection datasets
    """

    def __init__(
        self,
        root_data_directory: str,
        batch_size: int = 8,
        transforms: Optional[dict[str, Any]] = None,
        dataset_type: str = "roi",  # "roi" or "crop"
        rebalance_method: str = "mean",
        rebalance_exclude_extremes: bool = True,
        rebalance: bool = False,
        # ROI dataset parameters
        load_as_single_class: bool = False,
        background_class_name: str = "background",
        single_class_name: str = "wildlife",
        keep_classes: Optional[list[str]] = None,
        discard_classes: Optional[list[str]] = None,
        # Crop dataset parameters
        crop_size: int = 224,
        max_tn_crops: int = 1,
        p_draw_annotations: float = 0.0,
        # Curriculum parameters
        curriculum_config: Optional[Any] = None,
        compute_difficulties: bool = True,
        preserve_aspect_ratio: bool = True,
        num_workers: int = 8,
    ):
        # Initialize the curriculum mixin first
        if curriculum_config is not None:
            # Convert dict to CurriculumConfig if needed
            if isinstance(curriculum_config, dict) or isinstance(curriculum_config, DictConfig):
                curriculum_config = CurriculumConfig(**curriculum_config)
            elif not isinstance(curriculum_config, CurriculumConfig):
                logger.warning(f"Invalid curriculum_config type: {type(curriculum_config)}. Disabling curriculum.")
                curriculum_config = None
        
        CurriculumDataModuleMixin.__init__(self, curriculum_config)
        L.LightningDataModule.__init__(self)
        #super().__init__()
        
        self.batch_size = batch_size
        self.root_data_directory = Path(root_data_directory).resolve()
        self.dataset_type = dataset_type

        if not load_as_single_class:
            raise ValueError("Current workflow does not support multi-class datasets")
        
        self.num_workers = num_workers

        # ROI dataset configuration
        self.single_class_config = {
            "load_as_single_class": load_as_single_class,
            "background_class_name": background_class_name,
            "single_class_name": single_class_name,
            "keep_classes": keep_classes,
            "discard_classes": discard_classes,
        }

        # Crop dataset configuration
        self.crop_config = {
            "crop_size": crop_size,
            "max_tn_crops": max_tn_crops,
            "p_draw_annotations": p_draw_annotations,
        }

        # Curriculum configuration
        self.curriculum_config = curriculum_config
        self.compute_difficulties = compute_difficulties
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.transforms = transforms
        
        # Initialize datasets
        self.train_dataset: Optional[Union[ROIDataset, ConcatDataset, PatchDataset]] = None
        self.val_dataset: Optional[Union[ROIDataset, ConcatDataset, PatchDataset]] = None
        self.test_dataset: Optional[Union[ROIDataset, ConcatDataset, PatchDataset]] = None
        self.class_mapping: dict[int, str] = {}

        # Rebalancing
        if rebalance:
            self.resample_func = ClassificationRebalanceFilter(
                class_key="class_id", 
                random_seed=41, 
                method=rebalance_method,
                exclude_extremes=rebalance_exclude_extremes
            )
            logger.info(f"Rebalancing dataset with {rebalance_method} count and {'excluding' if rebalance_exclude_extremes else 'including'} extremes")
        else:
            self.resample_func: Optional[ClassificationRebalanceFilter] = None

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ClassificationDataModule':
        """
        Create ClassificationDataModule from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ClassificationDataModule instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        logger.info(f"Creating ClassificationDataModule from config: {config_path}")
        
        return cls.from_dict_config(config)

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> 'ClassificationDataModule':
        """
        Create ClassificationDataModule from DictConfig object.
        
        Args:
            config: DictConfig object containing dataset configuration
            
        Returns:
            ClassificationDataModule instance
        """
        dataset_config = config.dataset
        
        # Extract parameters using helper method
        params = cls._extract_config_params(dataset_config)

        params["num_workers"] = config.train.num_workers
                
        return cls(**params)

    @classmethod
    def _extract_config_params(cls, dataset_config: DictConfig) -> dict:
        """
        Extract parameters from dataset configuration.
        
        Args:
            dataset_config: Dataset configuration section
            
        Returns:
            Dictionary of parameters for ClassificationDataModule constructor
        """
        # Extract basic parameters
        root_data_directory = dataset_config.root_data_directory
        batch_size = dataset_config.get("batch_size", 8)
        dataset_type = dataset_config.get("dataset_type", "roi")
        
        # Handle transforms - convert to dict if it's a ListConfig
        transforms = dataset_config.get("transforms")
        if transforms is not None:
            # Convert ListConfig to dict if needed
            if hasattr(transforms, '_content'):
                transforms = dict(transforms)
            # Process transforms using the create_transforms function
            transforms = create_transforms(transforms)
        
        # ROI dataset parameters
        single_class_config = dataset_config.get("single_class", {})
        load_as_single_class = single_class_config.get("enable", False)
        background_class_name = single_class_config.get("background_class_name", "background")
        single_class_name = single_class_config.get("single_class_name", "wildlife")
        keep_classes = single_class_config.get("keep_classes")
        discard_classes = single_class_config.get("discard_classes")
        
        # Convert to proper types if needed
        if keep_classes is not None and not isinstance(keep_classes, list):
            keep_classes = [keep_classes] if keep_classes else None
        if discard_classes is not None and not isinstance(discard_classes, list):
            discard_classes = [discard_classes] if discard_classes else None
        
        # Rebalancing
        rebalance = dataset_config.get("rebalance", False)
        
        # Crop dataset parameters
        crop_size = dataset_config.get("crop_size", 224)
        max_tn_crops = dataset_config.get("max_tn_crops", 1)
        p_draw_annotations = dataset_config.get("p_draw_annotations", 0.0)
        
        # Curriculum parameters
        curriculum_config = dataset_config.get("curriculum_config")
        compute_difficulties = dataset_config.get("compute_difficulties", True)
        preserve_aspect_ratio = dataset_config.get("preserve_aspect_ratio", True)
        return {
            "root_data_directory": root_data_directory,
            "batch_size": batch_size,
            "transforms": transforms,
            "dataset_type": dataset_type,
            "load_as_single_class": load_as_single_class,
            "background_class_name": background_class_name,
            "single_class_name": single_class_name,
            "keep_classes": keep_classes,
            "discard_classes": discard_classes,
            "rebalance": rebalance,
            "crop_size": crop_size,
            "max_tn_crops": max_tn_crops,
            "p_draw_annotations": p_draw_annotations,
            "curriculum_config": curriculum_config,
            "compute_difficulties": compute_difficulties,
            "preserve_aspect_ratio": preserve_aspect_ratio,
        }

    def _get_class_mapping(self, dataset: Union[ROIDataset, ConcatDataset, PatchDataset]) -> dict[int, str]:
        """Get class mapping from dataset."""
        if isinstance(dataset, ConcatDataset):
            if isinstance(dataset.datasets[0], ROIDataset):
                return dataset.datasets[0].class_mapping
        elif isinstance(dataset, ROIDataset):
            return dataset.class_mapping
        elif isinstance(dataset, PatchDataset):
            return dataset.class_mapping
        else:
            raise ValueError(f"Dataset {type(dataset)} not supported")
        return {}  # Fallback return

    def _load_roi_dataset(self, splits: list[str], resample_function=None) -> dict[str, Union[ROIDataset, ConcatDataset]]:
        """Load ROI dataset using wildata."""
        # Ensure proper types for keep_classes and discard_classes
        keep_classes = self.single_class_config["keep_classes"]
        discard_classes = self.single_class_config["discard_classes"]
        
        # Convert to proper types if needed
        if keep_classes is not None and not isinstance(keep_classes, list):
            keep_classes = [keep_classes] if keep_classes else None
        if discard_classes is not None and not isinstance(discard_classes, list):
            discard_classes = [discard_classes] if discard_classes else None
        
        return load_all_splits_concatenated(
            self.root_data_directory,
            splits=splits,
            transform=self.transforms,
            resample_function=resample_function,
            load_as_single_class=bool(self.single_class_config["load_as_single_class"]),
            background_class_name=str(self.single_class_config["background_class_name"]),
            single_class_name=str(self.single_class_config["single_class_name"]),
            keep_classes=keep_classes,
            discard_classes=discard_classes,
        )

    def _load_crop_dataset(self, split: str) -> PatchDataset:
        """Load crop dataset from detection data."""
        # Create curriculum detection dataset
        detection_dataset = CurriculumDetectionDataset.from_data_directory(
            root_data_directory=str(self.root_data_directory),
            split=split,
            curriculum_config=self.curriculum_config,
            transform=self.transforms.get("train") if self.transforms else None,  # Use train transforms
            compute_difficulties=self.compute_difficulties,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
        )
        
        # Ensure proper types for keep_classes and discard_classes
        keep_classes = self.single_class_config["keep_classes"]
        discard_classes = self.single_class_config["discard_classes"]
        
        # Convert to proper types if needed
        if keep_classes is not None and not isinstance(keep_classes, list):
            keep_classes = [keep_classes] if keep_classes else None
        if discard_classes is not None and not isinstance(discard_classes, list):
            discard_classes = [discard_classes] if discard_classes else None
        
        # Create crop dataset
        crop_dataset = PatchDataset(
            dataset=detection_dataset,
            crop_size=int(self.crop_config["crop_size"]),
            max_tn_crops=int(self.crop_config["max_tn_crops"]),
            p_draw_annotations=self.crop_config["p_draw_annotations"],
            load_as_single_class=bool(self.single_class_config["load_as_single_class"]),
            background_class_name=str(self.single_class_config["background_class_name"]),
            single_class_name=str(self.single_class_config["single_class_name"]),
            keep_classes=keep_classes,
            discard_classes=discard_classes,
        )
        
        # Apply rebalancing if needed
        if self.resample_func:
            crop_dataset = crop_dataset.apply_rebalance_filter(self.resample_func)
        
        return crop_dataset

    def setup(self, stage: Optional[str] = None):
        """Setup datasets based on dataset type."""
        if stage == "fit":
            if self.dataset_type == "roi":
                # Load ROI datasets
                train_data = self._load_roi_dataset(
                    splits=["train"], 
                    resample_function=self.resample_func
                )
                
                # Check if val split exists
                available_splits = ["train", "val", "test"]
                val_data = {}
                if "val" in available_splits:
                    try:
                        val_data = self._load_roi_dataset(
                            splits=["val"], 
                            resample_function=None
                        )
                    except Exception as e:
                        logger.warning(f"Val split not found, using train split for validation: {e}")
                        val_data = {"val": train_data["train"]}
                
                self.train_dataset = train_data["train"]
                self.val_dataset = val_data.get("val", train_data["train"])  # Fallback to train if val not found
                
            elif self.dataset_type == "crop":
                # Load crop datasets
                self.train_dataset = self._load_crop_dataset("train")
                try:
                    self.val_dataset = self._load_crop_dataset("val")
                except Exception as e:
                    logger.warning(f"Val split not found, using train split for validation: {e}")
                    self.val_dataset = self.train_dataset
            
            # Get class mapping
            if self.train_dataset is not None:
                self.class_mapping = self._get_class_mapping(self.train_dataset)
            if self.val_dataset is not None:
                val_class_mapping = self._get_class_mapping(self.val_dataset)
                
                # For crop datasets, class mappings might differ due to different crops
                # Only assert if both datasets have the same number of classes
                if len(self.class_mapping) == len(val_class_mapping):
                    assert self.class_mapping == val_class_mapping, (
                        "Class mapping mismatch between train and val datasets"
                    )
                else:
                    logger.warning(f"Train dataset has {len(self.class_mapping)} classes, val dataset has {len(val_class_mapping)} classes")
                    # Use the larger class mapping to ensure all classes are covered
                    if len(self.class_mapping) > len(val_class_mapping):
                        logger.info("Using train dataset class mapping")
                    else:
                        logger.info("Using val dataset class mapping")
                        self.class_mapping = val_class_mapping
            
        elif stage == "validate":
            if self.dataset_type == "roi":
                try:
                    val_data = self._load_roi_dataset(splits=["val"])
                    self.val_dataset = val_data["val"]
                except Exception as e:
                    logger.warning(f"Val split not found: {e}")
                    # Load train split as fallback
                    train_data = self._load_roi_dataset(splits=["train"])
                    self.val_dataset = train_data["train"]
            elif self.dataset_type == "crop":
                try:
                    self.val_dataset = self._load_crop_dataset("val")
                except Exception as e:
                    logger.warning(f"Val split not found, using train split: {e}")
                    self.val_dataset = self._load_crop_dataset("train")
            
            if self.val_dataset is not None:
                self.class_mapping = self._get_class_mapping(self.val_dataset)
            
        elif stage == "test":
            if self.dataset_type == "roi":
                try:
                    test_data = self._load_roi_dataset(splits=["test"])
                    self.test_dataset = test_data["test"]
                except Exception as e:
                    logger.warning(f"Test split not found: {e}")
                    # Load train split as fallback
                    train_data = self._load_roi_dataset(splits=["train"])
                    self.test_dataset = train_data["train"]
            elif self.dataset_type == "crop":
                try:
                    self.test_dataset = self._load_crop_dataset("test")
                except Exception as e:
                    logger.warning(f"Test split not found, using train split: {e}")
                    self.test_dataset = self._load_crop_dataset("train")
            
            if self.test_dataset is not None:
                self.class_mapping = self._get_class_mapping(self.test_dataset)
        else:
            raise ValueError(f"Stage {stage} not supported")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not found")
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False if self.is_curriculum_enabled() else True, 
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not found")
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not found")
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
