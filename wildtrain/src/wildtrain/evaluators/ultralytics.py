from typing import Any, Dict, Tuple, Union, List, Generator

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import supervision as sv
from pathlib import Path
from .base import BaseEvaluator
from ..models.detector import Detector
from ..utils.io import merge_data_cfg
from ..data.filters.algorithms import FilterDataCfg
from ..data.utils import load_all_detection_datasets


class UltralyticsEvaluator(BaseEvaluator):
    """
    Evaluator for Ultralytics YOLO models.
    Implements evaluation logic using Ultralytics metrics.
    """

    def __init__(self, config: Union[DictConfig, str, dict]):
        super().__init__(config)
        self.model = self._load_model()
        self.dataloader = self._create_dataloader()    

    def _set_data_cfg(self):
        assert (self.config.dataset.data_cfg is not None) ^ (self.config.dataset.root_data_directory is not None), "Either data_cfg or root_data_directory must be provided"
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

        # updating data_cfg
        if self.config.dataset.root_data_directory is not None:
            data_cfg = Path(self.config.results_dir)/"merged_data_cfg.yaml"
            merge_data_cfg(root_data_directory=self.config.dataset.root_data_directory,
                                                        output_path=data_cfg,
                                                        force_merge=self.config.dataset.force_merge)
            self.config.dataset.data_cfg = data_cfg 

    def _load_model(self) -> Any:
        localizer_config = OmegaConf.create({
            'weights': str(self.config.weights.localizer),
            'imgsz': self.config.eval.imgsz,
            'device': self.config.device,
            'conf_thres': self.config.eval.conf,
            'iou_thres': self.config.eval.iou,
            'overlap_metric': 'IOU',
            'task': self.config.eval.task,
            'max_det': self.config.eval.max_det,
        })
        return Detector.from_config(localizer_config=localizer_config,classifier_ckpt=self.config.weights.classifier)

    def _create_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for evaluation using Ultralytics' build_yolo_dataset utility.

        Args:
            data_config: Dictionary from YOLO data YAML (contains names, path, val/test, etc.).
            eval_config: Dictionary of evaluation parameters (imgsz, batch_size, etc.).

        Returns:
            PyTorch DataLoader for the evaluation split.
        """
        # Convert DictConfig to dict properly to preserve all fields
        eval_config = self.config.eval
        split = eval_config.split
        if split is None:
            raise ValueError(f"No 'split' found in eval_config: {eval_config}")
        
        dataset = load_all_detection_datasets(root_data_directory=self.config.dataset.root_data_directory,
                                              split=split)

        dataloader = DataLoader(
            dataset,
            batch_size=eval_config.batch_size,
            shuffle=False,
            num_workers=eval_config.num_workers,
            collate_fn=self._collate_fn,
        )
        return dataloader

    def _collate_fn(self, batch: List[Tuple[str, np.ndarray, sv.Detections]]) -> Dict[str, Any]:
        """
        Custom collate function for YOLO evaluation batches.
        Stacks images into a batch tensor; collects all other fields as lists.
        Discards 'batch_idx'.
        """
        imgs = torch.stack([torch.from_numpy(item[1]) for item in batch], dim=0).float()
        imgs = imgs.permute(0,3,1,2)
        if imgs.min()>= 0. and imgs.max() > 1.:
            imgs = imgs / 255.

        im_files = [item[0] for item in batch]

        # filter images
        selected_im_files = FilterDataCfg.filter_image_paths(image_paths=im_files,
                                                    label_map=self.model.class_mapping,
                                                    keep_classes=self.config.dataset.keep_classes,
                                                    discard_classes=self.config.dataset.discard_classes)
        indices = [im_files.index(im_file) for im_file in selected_im_files]
        annotations = [batch[i][2] for i in indices]
        for i,ann in enumerate(annotations):
            ann.metadata["file_path"] = selected_im_files[i]

        return {
            "img": imgs,
            "annotations": annotations,
        }

    def _run_inference(self) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        for batch in tqdm(self.dataloader, desc="Running inference"):
            predictions = self.model.predict(batch["img"],return_as_dict=False)
            gt = batch["annotations"]
            yield dict(predictions=predictions, ground_truth=gt)
