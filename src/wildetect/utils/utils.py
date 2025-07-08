import logging
import traceback
from pathlib import Path
from typing import List, Optional, Union

import torch
from torchmetrics.functional.detection import complete_intersection_over_union

from wildetect.core.config import ROOT

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


def load_registered_model(
    alias,
    name,
    tag_to_append: str = "",
    mlflow_tracking_url="http://localhost:5000",
    load_unwrapped: bool = False,
    dwnd_location: Optional[Union[str, Path]] = ROOT / "models",
):
    mlflow.set_tracking_uri(mlflow_tracking_url)

    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}" + tag_to_append
    modelURI = f"models:/{name}/{version}"

    if dwnd_location is None:
        dwnd_location = ROOT / Path(f"models/{name}")
        dwnd_location.mkdir(parents=True, exist_ok=True)
        dwnd_location = dwnd_location / version
        dwnd_location = str(dwnd_location.resolve())
    try:
        model = mlflow.pyfunc.load_model(str(dwnd_location))
    except:
        model = mlflow.pyfunc.load_model(modelURI, dst_path=str(dwnd_location))

    metadata = dict(version=modelversion, modeluri=modelURI)
    try:
        metadata.update(model.metadata.metadata)
    except:
        logger.warning(
            f"No metadata found for model {modelversion}. msg {traceback.format_exc()}"
        )

    if load_unwrapped:
        try:
            model = model.unwrap_python_model().model
            metadata["detection_model_type"] = "ultralytics"
        except:
            try:
                model = model.unwrap_python_model().detection_model
                metadata["detection_model_type"] = "ultralytics"
            except:
                model = model.unwrap_python_model().classifier
                metadata["detection_model_type"] = "classifier"

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
