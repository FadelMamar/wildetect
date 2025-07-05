import logging
import torch
from torchmetrics.functional.detection import complete_intersection_over_union

from typing import List

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

    bbox1 = torch.tensor([bbox1])
    bbox2 = torch.tensor([bbox2])

    iou = complete_intersection_over_union(
        preds=bbox1, target=bbox2, aggregate=False
    ).item()

    return iou



def load_registered_model(
    alias,
    name,
    tag_to_append: str = "",
    mlflow_tracking_url="http://localhost:5000",
    load_unwrapped: bool = False,
):
    
    mlflow.set_tracking_uri(mlflow_tracking_url)

    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}" + tag_to_append
    modelURI = f"models:/{name}/{version}"

    model = mlflow.pyfunc.load_model(modelURI)

    metadata = dict(version=modelversion, modeluri=modelURI)
    metadata.update(model.metadata.metadata)

    if load_unwrapped:
        try:
            model = model.unwrap_python_model().model
        except:
            try:
                model = model.unwrap_python_model().detection_model
            except:
                model = model.unwrap_python_model().classifier

    metadata["detection_model_type"] = "ultralytics"

    return model, metadata


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