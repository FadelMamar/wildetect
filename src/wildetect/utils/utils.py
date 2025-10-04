import logging
import traceback
from pathlib import Path
from typing import List, Optional, Union

import torch
from torchmetrics.functional.detection import complete_intersection_over_union

logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:
    logger.warning("mlflow not installed")


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    Args:
        bbox1 (List[float]): Bounding box in [x_min, y_min, x_max, y_max] format.
        bbox2 (List[float]): Bounding box in [x_min, y_min, x_max, y_max] format.
    Returns:
        float: IoU value between the two bounding boxes.
    """

    iou = complete_intersection_over_union(
        preds=torch.tensor([bbox1]), target=torch.tensor([bbox2]), aggregate=False
    ).item()

    return iou


def get_experiment_id(name: str):
    """Gets mlflow experiments id

    Args:
        name (str): mlflow experiment name

    Returns:
        str: experiment id
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id
