from typing import Any, Dict, Tuple, List, Generator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import supervision as sv
from pathlib import Path
from typing import Any, Dict, List, Generator, Optional
from supervision.metrics import (
    MeanAveragePrecision,
    MeanAverageRecall,
)
import supervision as sv
from supervision.metrics.detection import ConfusionMatrix
from logging import getLogger
import traceback
from .metrics import MyPrecision,MyRecall,MyF1Score
import json
from ..shared.models import DetectionEvalConfig
from ..shared.models import DetectionEvalConfig
from ..models.detector import Detector
from ..utils.io import merge_data_cfg
from ..data.utils import load_all_detection_datasets

logger = getLogger(__name__)


class UltralyticsEvaluator:
    """
    Evaluator for Ultralytics YOLO models.
    Implements evaluation logic using Ultralytics metrics.
    """

    def __init__(self, config: DetectionEvalConfig,disable_detection_filtering:bool=False):
        self.config = config      
        self.metrics = self._get_metrics()
        self._report: Dict[str, float] = dict()
        self.gt_and_preds: List[dict[str, List[sv.Detections]]] = []
        self._gt_labels_to_keep:list[int] = None
        self._pred_labels_to_keep:list[int] = None

        self.model = self._load_model(disable_detection_filtering=disable_detection_filtering)
        self.class_mapping = self.model.class_mapping        
        self.dataloader = self._create_dataloader()
    
    @property
    def labels(self)->List[str]:
        if self.class_mapping is None:
            return None
        return [self.class_mapping[i] for i in sorted(self.class_mapping.keys())]    
    
    def _get_metrics(self,):
        boxes = sv.metrics.core.MetricTarget.BOXES
        average = getattr(sv.metrics.core.AveragingMethod,self.config.metrics.average.upper())
        return dict(
            mAP=MeanAveragePrecision(
                boxes,
                class_agnostic=self.config.metrics.class_agnostic,
                class_mapping=None,
            ),
            mAR=MeanAverageRecall(boxes),
            precision=MyPrecision(boxes, averaging_method=average),
            recall=MyRecall(boxes, averaging_method=average),
            f1=MyF1Score(boxes, averaging_method=average),
        )
        
    def evaluate(
        self,
        debug:bool=False,
        save_path:Optional[str]=None
    ) -> Dict[str, Any]:
        """
        Evaluate model using parameters from config dict passed via kwargs.
        """
        count = 0
        for results in self._run_inference():
            self.gt_and_preds.append(results)
            try:
                self._compute_metrics(results)
            except Exception:
                logger.error(f"Error computing metrics: {traceback.format_exc()}")
                #logger.info(results)
                raise
            count += 1
            if debug and count > 10:
                break

        try:
            self._set_report(self._get_results())
            if save_path:
                self.save_report(save_path)
        except Exception:
            logger.error(f"Error generating report: {traceback.format_exc()}")

        return self._report

    def _compute_metrics(self, results: Dict[str, List[sv.Detections]]) -> None:
        """
        Compute metrics for a batch of results.
        """
        assert len(results["predictions"]) == len(results["ground_truth"]), f"Number of predictions and ground truth must be the same, got {len(results['predictions'])} and {len(results['ground_truth'])}"
        for metric_name, metric in self.metrics.items():
            for pred, gt in zip(results["predictions"], results["ground_truth"]):
                if not isinstance(pred, sv.Detections):
                    raise ValueError(f"Prediction must be a sv.Detections object, got {type(pred)}")
                if not isinstance(gt, sv.Detections):
                    raise ValueError(f"Ground truth must be a sv.Detections object, got {type(gt)}")
                #if pred.is_empty() and gt.is_empty():
                #    continue
                metric.update(pred, gt)               
        
    def get_report(self) -> Dict[str, Any]:
        return self._report
    
    def get_confusion_matrix(self) -> ConfusionMatrix:
        preds = []
        gts = []
        for values in self.gt_and_preds:
            preds.extend(values["predictions"])
            gts.extend(values["ground_truth"])
        #print("labels:",self.labels)
        #for pred, gt in zip(preds, gts):
            #if not gt.is_empty() and gt.class_id.max()>1:
            #    print(gt)
       
        return ConfusionMatrix.from_detections(
            predictions=preds,
            targets=gts,
            classes=self.labels,
            conf_threshold=self.config.eval.conf,
            iou_threshold=self.config.eval.iou,
        )

    def _set_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a summary evaluation report as a pandas DataFrame.
        Includes mAP@50, mAP@75, mAR@1, Precision@50, Recall@50, F1@50.
        Only summary metrics are included, no per-class metrics.
        """

        dfs = {}
        for name, result in results.items():
            df = result.to_pandas()
            for record in df.to_dict(orient='records'):
                dfs.update(record)

        for name in ["f1","precision","recall"]:
            argmax = getattr(results[name],f"{name}_scores").argmax()
            best_score = getattr(results[name],f"{name}_scores")[argmax]
            best_iou = results[name].iou_thresholds[argmax]
            dfs[f'best_{name}'] = {f'{name}_at_{best_iou}': best_score}
            dfs[f"{name}_scores"] = list(zip(results[name].iou_thresholds, getattr(results[name],f"{name}_scores")))

        self._report = dfs

    def _get_results(self) -> Dict[str, Any]:
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def save_report(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self._report, f,indent=2)
        
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
        
        self._gt_labels_to_keep = [i for i,class_name in enumerate(gt_classes) if _is_class_to_keep(class_name)]
        self._pred_labels_to_keep = [i for i,class_name in enumerate(self.class_mapping.values()) if _is_class_to_keep(class_name)]
        
        # update class mapping
        new_class_mapping = {}
        for label,class_name in self.class_mapping.items():
            if _is_class_to_keep(class_name):
                new_class_mapping[label] = class_name
        self.class_mapping = new_class_mapping
        return None

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
    
    def _filter_prediction(self, prediction: sv.Detections) -> sv.Detections:
        filtered_prediction = sv.Detections.empty()
        valid_indices = []
        for i,class_id in enumerate(prediction.class_id.tolist()):
            if class_id in self._pred_labels_to_keep:
                valid_indices.append(i)
        filtered_prediction.class_id = prediction.class_id[valid_indices]
        if isinstance(prediction.confidence, np.ndarray):
            filtered_prediction.confidence = prediction.confidence[valid_indices]
        filtered_prediction.xyxy = prediction.xyxy[valid_indices]
        filtered_prediction.metadata = prediction.metadata
        return filtered_prediction

    def _run_inference(self) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        for batch in tqdm(self.dataloader, desc="Running inference"):
            predictions = self.model.predict(batch["img"],return_as_dict=False)
            gts = batch["annotations"]
            for i in range(len(predictions)):
                predictions[i] = self._filter_prediction(predictions[i])
                gts[i] = self._filter_annotation(gts[i])

            yield dict(predictions=predictions, ground_truth=gts)
