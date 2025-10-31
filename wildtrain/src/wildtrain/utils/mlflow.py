import mlflow
import traceback
from pathlib import Path
from typing import Optional, Union
from ..utils.logging import get_logger

logger = get_logger(__name__)

def load_registered_model(
    alias,
    name,
    mlflow_tracking_url="http://localhost:5000",
    load_unwrapped: bool = False,
    dwnd_location: Optional[Union[str, Path]] = None,
):
    mlflow.set_tracking_uri(mlflow_tracking_url)

    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}"
    modelURI = f"models:/{name}/{version}"

    if dwnd_location is None:
        dwnd_location = Path(f"models-registry/{name}")
    else:
        dwnd_location = Path(dwnd_location)/name
    
    dwnd_location.mkdir(parents=True, exist_ok=True)
    dwnd_location = dwnd_location / version
    dwnd_location = str(dwnd_location.resolve())

    Path(dwnd_location).mkdir(parents=True, exist_ok=True)

    try:
        model = mlflow.pyfunc.load_model(str(dwnd_location))
    except Exception as e:
        logger.error(f"Error loading model from {dwnd_location}: {e}")
        logger.info(f"Loading model from {modelURI}")
        model = mlflow.pyfunc.load_model(modelURI, dst_path=str(dwnd_location))

    metadata = dict(version=modelversion, modeluri=modelURI,
                    model_path=str(dwnd_location))
    try:
        metadata.update(model.metadata.metadata)
    except:
        logger.warning(
            f"No metadata found for model {modelversion}. msg {traceback.format_exc()}"
        )

    if load_unwrapped:
        model = model.unwrap_python_model().model
    model.metadata = metadata
    return model


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
