from typing import Any, Dict, Tuple, List, Generator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import supervision as sv
from pathlib import Path
from .base import BaseEvaluator
from ..shared.models import DetectionEvalConfig
from ..models.detector import Detector
from ..utils.io import merge_data_cfg
from ..data.filters.algorithms import FilterDataCfg
from ..data.utils import load_all_detection_datasets


class UltralyticsEvaluator(BaseEvaluator):
    """
    Evaluator for Ultralytics YOLO models.
    Implements evaluation logic using Ultralytics metrics.
    """

    def __init__(self, config: DetectionEvalConfig,disable_detection_filtering:bool=False):
        super().__init__(config)
        self.model = self._load_model(disable_detection_filtering=disable_detection_filtering)
        self.class_mapping = self.model.class_mapping
        self._gt_labels_to_keep:list[int] = None
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
            print(f"Merged data cfg saved to: {data_cfg}")
            self.config.dataset.data_cfg = data_cfg 

    def _load_model(self,disable_detection_filtering:bool=False) -> Detector:
        localizer_config = self.config.to_yolo_inference_config(disable_detection_filtering=disable_detection_filtering)
        return Detector.from_config(localizer_config=localizer_config,classifier_ckpt=self.config.weights.classifier)
    
    def _set_gt_labels_to_keep(self,gt_classes:list[str]):
        def _is_class_to_keep(class_name:str) -> bool:
            if self.config.dataset.keep_classes is not None and class_name in self.config.dataset.keep_classes:
                return True
            if self.config.dataset.discard_classes is not None and class_name not in self.config.dataset.discard_classes:
                return True
            return False
        #print("gt_classes:",gt_classes)
        #print("keep_classes:",self.config.dataset.keep_classes)
        #print("discard_classes:",self.config.dataset.discard_classes)
        self._gt_labels_to_keep = [i for i,class_name in enumerate(gt_classes) if _is_class_to_keep(class_name)]

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
        self._set_gt_labels_to_keep(dataset.classes)
        #print("gt_labels_to_keep:",self._gt_labels_to_keep)
        
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
        annotations = [b[2] for b in batch]
        selected_im_files = im_files        
        for i,ann in enumerate(annotations):
            ann.metadata["file_path"] = selected_im_files[i]
            annotations[i] = self._filter_annotation(ann)

        return {
            "img": imgs,
            "annotations": annotations,
        }
    
    def _filter_annotation(self, annotation: sv.Detections) -> sv.Detections:
        filtered_annotation = sv.Detections.empty()
        valid_indices = []
        for i,class_id in enumerate(annotation.class_id.tolist()):
            if class_id in self._gt_labels_to_keep:
                valid_indices.append(i)
        filtered_annotation.class_id = annotation.class_id[valid_indices]
        if isinstance(annotation.confidence, np.ndarray):
            filtered_annotation.confidence = annotation.confidence[valid_indices]
        filtered_annotation.xyxy = annotation.xyxy[valid_indices]
        filtered_annotation.metadata = annotation.metadata

        if self.config.dataset.load_as_single_class:
            filtered_annotation.class_id = np.zeros_like(filtered_annotation.class_id,dtype=int)
        
        return filtered_annotation

    def _run_inference(self) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        for batch in tqdm(self.dataloader, desc="Running inference"):
            predictions = self.model.predict(batch["img"],return_as_dict=False)
            gt = batch["annotations"]
            yield dict(predictions=predictions, ground_truth=gt)
