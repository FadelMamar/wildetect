from typing import List, Optional
from urllib.parse import unquote

from label_studio_sdk.client import LabelStudio
from label_studio_tools.core.utils.io import get_local_path
from tqdm import tqdm

from ..config import FlightSpecs
from ..data import Detection, DroneImage


class LabelStudioManager:
    def __init__(self, url: str, api_key: str, download_resources: bool = False):
        self.client = LabelStudio(base_url=url, api_key=api_key)
        self.download_resources = download_resources
        self.url = url

    def get_project(self, project_id: int):
        return self.client.projects.get(project_id)

    def get_image_path(self, task_id: int):
        url = unquote(self.client.tasks.get(task_id).data["image"])
        return get_local_path(
            url,
            download_resources=self.download_resources,
            hostname=self.url,
        )

    def get_tasks(self, project_id: int):
        return self.client.tasks.list(project=project_id)

    def get_detections(
        self,
        task_id: int,
        load_as_drone_image: bool = False,
        flight_specs: Optional[FlightSpecs] = None,
    ) -> Optional[dict | DroneImage]:
        annotations = self.client.annotations.list(id=task_id)
        predictions = self.client.predictions.list(task=task_id)
        image_path = self.get_image_path(task_id)

        if len(annotations) == 0 and len(predictions) == 0:
            return None

        if load_as_drone_image:
            annotations = Detection.from_ls(annotations, image_path)
            predictions = Detection.from_ls(predictions, image_path)
            image = DroneImage(image_path=image_path, flight_specs=flight_specs)
            image.set_annotations(annotations, update_gps=True)
            image.set_predictions(predictions, update_gps=True)
            return image

        return dict(
            image_path=image_path, annotations=annotations, predictions=predictions
        )

    def get_all_project_detections(
        self,
        project_id: int,
        load_as_drone_image: bool = False,
        flight_specs: Optional[FlightSpecs] = None,
    ) -> List[dict | DroneImage]:
        tasks = self.get_tasks(project_id)
        detections = []
        for task in tqdm(tasks, desc="Getting Label Studio annotations"):
            dets = self.get_detections(
                task.id,
                load_as_drone_image=load_as_drone_image,
                flight_specs=flight_specs,
            )
            if dets is not None:
                detections.append(dets)
        return detections
