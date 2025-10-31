from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Generator, Optional
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from supervision.metrics import (
    MeanAveragePrecision,
    MeanAverageRecall,
)
import supervision as sv
from copy import deepcopy
from logging import getLogger
import traceback
from .metrics import MyPrecision,MyRecall,MyF1Score
import json

logger = getLogger(__name__)

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    Subclasses must implement the evaluate method.
    """

    def __init__(self, config: Union[DictConfig, str, dict]):
        if isinstance(config, str):
            self.config = DictConfig(OmegaConf.load(config))
        elif isinstance(config, dict):
            self.config = DictConfig(config)
        elif isinstance(config, DictConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        self.metrics = self._get_metrics()
        self.per_image_metrics = deepcopy(self.metrics)
        self.per_image_results = dict()
        self._report: Dict[str, float] = dict()
        
    
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
            try:
                self._compute_metrics(results)
            except Exception:
                logger.error(f"Error computing metrics: {traceback.format_exc()}")
                logger.info(results)
                raise
            count += 1
            if debug and count > 10:
                break

        results = self._get_results()
        try:
            self._set_report(results)
            if save_path:
                self.save_report(save_path)
        except Exception:
            logger.error(f"Error generating report: {traceback.format_exc()}")

        self._reset()
        return self._report

    @abstractmethod
    def _run_inference(self) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        """
        Run inference and return predictions and ground truth.
        """
        pass

    def _compute_metrics(self, results: Dict[str, List[sv.Detections]]) -> None:
        """
        Compute metrics for a batch of results.
        """
        assert len(results["predictions"]) == len(results["ground_truth"]), "Number of predictions and ground truth must be the same"
        for metric_name, metric in self.metrics.items():
            for pred, gt in zip(results["predictions"], results["ground_truth"]):
                if not isinstance(pred, sv.Detections):
                    raise ValueError(f"Prediction must be a sv.Detections object, got {type(pred)}")
                if not isinstance(gt, sv.Detections):
                    raise ValueError(f"Ground truth must be a sv.Detections object, got {type(gt)}")
                #if not pred.is_empty():
                metric.update(pred, gt)               
        
    def _reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.per_image_results = dict()
    
    def get_report(self) -> Dict[str, Any]:
        return self._report

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
