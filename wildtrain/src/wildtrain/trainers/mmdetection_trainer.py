import os
import json
import torch
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
import mlflow
from dotenv import load_dotenv
import shutil
import mmdet
from mmdet.datasets.coco import CocoDataset
from mmengine.config import Config
from mmengine.registry import RUNNERS, MODELS, DATASETS
from mmengine.runner import Runner


from ..utils.logging import ROOT, get_logger
from .base import ModelTrainer

logger = get_logger(__name__)


def test_mmdet_config(config_file: str):
    mmdet_cfg = Config.fromfile(config_file)

    dataset_cfg = mmdet_cfg.train_dataloader.dataset
    DATASETS.build(dataset_cfg)

    model_cfg = mmdet_cfg.model
    MODELS.build(model_cfg)


class MMDetectionTrainer(ModelTrainer):
    """
    Trainer class for object detection models using MMDetection.

    This class handles the training and evaluation of detection models
    using MMDetection framework with MLflow for experiment tracking.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.mmdet_cfg: Optional[Config] = None
        self.class_mapping: Dict[int, str] = {}
        self.runner: Optional[Runner] = None

    # TODO: debug this function
    def _filter_annotations(
        self,
        coco_data: Dict[str, Any],
        discard_labels: Optional[List[str]] = None,
        keep_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Filter annotations and categories from COCO data and recompute category indices.

        Args:
            coco_data: COCO format dictionary containing 'annotations', 'categories', and 'images'
            discard_labels: List of category names to discard
            keep_labels: List of category names to keep (takes precedence over discard_labels)

        Returns:
            Filtered COCO data with recomputed category indices
        """

        check = (keep_labels is not None) + (discard_labels is not None)
        assert check == 1, "Either keep_labels or discard_labels must be provided"

        if check == 0:
            # No filtering, return original data
            return coco_data

        # Create a copy to avoid modifying the original
        filtered_data = {
            "images": coco_data.get("images", []),
            "annotations": coco_data.get("annotations", []),
            "categories": coco_data.get("categories", []),
        }

        # Determine which categories to keep
        if keep_labels is not None:
            categories_to_keep = [
                cat for cat in filtered_data["categories"] if cat["name"] in keep_labels
            ]
        else:
            categories_to_keep = [
                cat
                for cat in filtered_data["categories"]
                if cat["name"] not in discard_labels
            ]

        # Create mapping from old category IDs to new ones
        old_to_new_id = {}
        for new_id, category in enumerate(categories_to_keep, start=1):
            old_to_new_id[category["id"]] = new_id
            category["id"] = new_id

        # Filter annotations to only include those with kept categories
        kept_category_ids = set(old_to_new_id.keys())
        filtered_annotations = []
        for annotation in filtered_data["annotations"]:
            if annotation["category_id"] in kept_category_ids:
                # Update category_id to new index
                annotation["category_id"] = old_to_new_id[annotation["category_id"]]
                filtered_annotations.append(annotation)

        # Filter images based on whether they contain annotations for kept categories
        # TODO: keep negative samples as well
        images_with_annotations = set()
        for annotation in filtered_annotations:
            images_with_annotations.add(annotation["image_id"])

        filtered_images = []
        for image in filtered_data["images"]:
            if image["id"] in images_with_annotations:
                filtered_images.append(image)

        # Update the filtered data
        filtered_data["categories"] = categories_to_keep
        filtered_data["annotations"] = filtered_annotations
        filtered_data["images"] = filtered_images

        return filtered_data

    def _make_single_class_annotations(
        self,
        train_ann: Dict[str, Any],
        val_ann: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ):
        single_class_name = getattr(
            self.config.dataset, "single_class_name", "wildlife"
        )
        single_class_id = getattr(self.config.dataset, "single_class_id", 0)
        categories = [{"id": single_class_id, "name": single_class_name}]

        # update annotations
        train_ann["categories"] = categories
        val_ann["categories"] = categories
        for ann in train_ann["annotations"]:
            ann["category_id"] = single_class_id
        for ann in val_ann["annotations"]:
            ann["category_id"] = single_class_id

        # update dataset_info
        dataset_info["classes"] = categories

        return train_ann, val_ann, dataset_info

    def _update_categories(
        self,
    ):
        """
        Create copies of annotation files with updated categories in a persistent temp directory.
        Update config to use these new files.
        """

        root = Path(self.config.ROOT_DATASET)
        train_ann_path = root / self.config.dataset.train_ann
        val_ann_path = root / self.config.dataset.val_ann
        dataset_info_path = root / self.config.dataset.dataset_info

        # Load original annotation files
        with open(train_ann_path, "r", encoding="utf-8") as file:
            train_ann = json.load(file)
        with open(val_ann_path, "r", encoding="utf-8") as file:
            val_ann = json.load(file)
        with open(dataset_info_path, "r", encoding="utf-8") as file:
            dataset_info = json.load(file)

        # single-class
        if getattr(self.config.dataset, "load_as_single_class", False):
            train_ann, val_ann, dataset_info = self._make_single_class_annotations(
                train_ann=train_ann, val_ann=val_ann, dataset_info=dataset_info
            )

            # Save modified files
            new_train_ann = train_ann_path.parent / "train_single_class.json"
            new_val_ann = val_ann_path.parent / "val_single_class.json"
            new_dataset_info = (
                dataset_info_path.parent / "dataset_info_single_class.json"
            )
            with open(new_train_ann, "w", encoding="utf-8") as file:
                json.dump(train_ann, file, indent=2)
            with open(new_val_ann, "w", encoding="utf-8") as file:
                json.dump(val_ann, file, indent=2)
            with open(new_dataset_info, "w", encoding="utf-8") as file:
                json.dump(dataset_info, file, indent=2)

            # Update config to use these files
            self.config.dataset.train_ann = os.path.relpath(new_train_ann, root)
            self.config.dataset.val_ann = os.path.relpath(new_val_ann, root)
            self.config.dataset.dataset_info = os.path.relpath(new_dataset_info, root)

    def _setup(self) -> None:
        self.mmdet_cfg = Config.fromfile(self.config.model.config_file)

        # update categories
        self._update_categories()

        # get categories info
        root = Path(self.config.ROOT_DATASET)
        with open(
            root / self.config.dataset.dataset_info, "r", encoding="utf-8"
        ) as file:
            dataset_info = json.load(file)

        self.class_mapping = {
            item["id"]: item["name"] for item in dataset_info["classes"]
        }
        num_classes = len(self.class_mapping)
        classes = tuple(
            self.class_mapping[i] for i in sorted(self.class_mapping.keys())
        )
        self.mmdet_cfg.model.roi_head.bbox_head.num_classes = num_classes

        # freeze backbone
        if self.config.train.freeze_backbone:
            self.mmdet_cfg.model.backbone.frozen_stages = len(
                self.mmdet_cfg.model.backbone.out_indices
            )

        # training runtime
        self.mmdet_cfg.train_cfg.max_epochs = self.config.train.epochs
        self.mmdet_cfg.train_cfg.val_interval = self.config.train.val_interval

        setattr(
            self.mmdet_cfg.optim_wrapper, "optimizer", dict(self.config.train.optimizer)
        )
        setattr(
            self.mmdet_cfg, "param_scheduler", dict(self.config.train.param_scheduler)
        )
        setattr(
            self.mmdet_cfg.default_hooks,
            "checkpoint",
            dict(self.config.train.checkpointer),
        )

        self.mmdet_cfg.visualizer.vis_backends = [
            {
                "type": "MLflowVisBackend",
                "tracking_uri": self.config.mlflow.tracking_uri,
                "save_dir": self.config.work_dir,
                "run_name": self.config.mlflow.run_name,
                "exp_name": self.config.mlflow.experiment_name,
            }
        ]

        # Region proposal
        self.mmdet_cfg.model.train_cfg.rpn_proposal.nms.iou_threshold = 0.6
        self.mmdet_cfg.model.train_cfg.rpn_proposal.max_per_img = 300
        for name in ["pos_iou_thr", "neg_iou_thr", "min_pos_iou"]:
            value = getattr(self.config.train, name)
            if value is not None:
                setattr(self.mmdet_cfg.model.train_cfg.rcnn.assigner, name, value)

        self.mmdet_cfg.model.test_cfg.rpn.nms.iou_threshold = 0.6
        self.mmdet_cfg.model.test_cfg.rpn.max_per_img = 300
        self.mmdet_cfg.model.test_cfg.rcnn.nms.iou_threshold = 0.5
        self.mmdet_cfg.model.test_cfg.rcnn.max_per_img = 300

        # setting dataloaders values
        for name in ["batch_size", "num_workers", "persistent_workers"]:
            value = getattr(self.config.dataloader, name)
            if value is not None:
                setattr(self.mmdet_cfg.train_dataloader, name, value)
                setattr(self.mmdet_cfg.val_dataloader, name, value)

        # setting dataset values
        ## Train loader
        self.mmdet_cfg.train_dataloader.dataset.data_root = self.config.ROOT_DATASET
        self.mmdet_cfg.train_dataloader.dataset.ann_file = self.config.dataset.train_ann
        self.mmdet_cfg.train_dataloader.dataset.data_prefix.img = (
            self.config.dataset.train_images
        )
        self.mmdet_cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt = (
            self.config.dataset.filter_empty_gt.train
        )
        self.mmdet_cfg.train_dataloader.dataset.metainfo.classes = classes

        ## Val loader
        self.mmdet_cfg.val_dataloader.dataset.data_root = self.config.ROOT_DATASET
        self.mmdet_cfg.val_dataloader.dataset.ann_file = self.config.dataset.val_ann
        self.mmdet_cfg.val_dataloader.dataset.test_mode = True
        self.mmdet_cfg.val_dataloader.dataset.data_prefix.img = (
            self.config.dataset.val_images
        )
        self.mmdet_cfg.val_dataloader.dataset.filter_cfg.filter_empty_gt = (
            self.config.dataset.filter_empty_gt.val
        )
        self.mmdet_cfg.val_dataloader.dataset.metainfo.classes = classes

        ## evaluator
        for name in ["type", "ann_file", "metric", "format_only"]:
            value = getattr(self.config.val_evaluator, name)
            if value is not None:
                setattr(self.mmdet_cfg.val_evaluator, name, value)

        if self.config.train.amp == True and torch.cuda.is_available():
            self.mmdet_cfg.optim_wrapper.type = "AmpOptimWrapper"
            self.mmdet_cfg.optim_wrapper.loss_scale = "dynamic"

        if self.config.train.resume == "auto":
            self.mmdet_cfg.resume = True
            self.mmdet_cfg.load_from = None
        elif self.config.train.resume is not None:
            self.mmdet_cfg.resume = True
            self.mmdet_cfg.load_from = self.config.train.resume

        if self.config.work_dir is not None:
            self.mmdet_cfg.work_dir = self.config.work_dir
        else:
            self.mmdet_cfg.work_dir = str(ROOT / "work_dirs" / "mmdet")
        Path(self.mmdet_cfg.work_dir).mkdir(parents=True, exist_ok=True)

    @property
    def num_frozen_params(
        self,
    ):
        if self.runner is None:
            logger.info(
                "cannot compute number of frozen params. Runner is not set. Returning 0"
            )
            return 0

        backbone = self.runner.model.backbone
        count = 0
        for param in backbone.parameters():
            count += param.requires_grad == False

        logger.info(f"Frozen params: {count}")

        return count

    def _clean_up(self):
        pass

    def run(self, debug: bool = False) -> None:
        """
        Run object detection training or evaluation using MMDetection.

        Args:
            debug: If True, run with limited batches for debugging
        """

        self._setup()

        # build the runner from config
        if "runner_type" not in self.mmdet_cfg:
            self.runner = Runner.from_cfg(self.mmdet_cfg)
        else:
            self.runner = RUNNERS.build(self.mmdet_cfg)

        self.runner.train()

        self._clean_up()
