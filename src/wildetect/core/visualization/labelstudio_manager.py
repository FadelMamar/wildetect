import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Optional
from urllib.parse import unquote

from label_studio_sdk.client import LabelStudio
from label_studio_tools.core.utils.io import get_local_path
from tqdm import tqdm

from ..data import Detection

logger = logging.getLogger(__name__)


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

    def get_tasks_paths(self, project_id: int) -> Dict[str, int]:
        tasks = self.get_tasks(project_id)
        return {self.get_image_path(task.id): task.id for task in tasks}

    def get_detections(
        self,
        task_id: int,
    ) -> Optional[dict]:
        annotations = self.client.annotations.list(id=task_id)
        predictions = self.client.predictions.list(task=task_id)
        image_path = self.get_image_path(task_id)

        if len(annotations) == 0 and len(predictions) == 0:
            return None

        return dict(
            image_path=image_path, annotations=annotations, predictions=predictions
        )

    def get_all_project_detections(
        self,
        project_id: int,
    ) -> List[dict]:
        tasks = [task.id for task in self.get_tasks(project_id)]
        detections = []

        logger.info(f"Getting detections for {len(tasks)} tasks...")

        # with ThreadPoolExecutor(max_workers=3) as executor:
        for dets in tqdm(
            map(self.get_detections, tasks),
            total=len(tasks),
            desc="Getting detections from Label Studio",
        ):
            if dets is not None:
                detections.append(dets)
        return detections

    def upload_detections(
        self,
        task_id: int,
        detections: List[Detection],
        model_tag: str,
        from_name: str,
        to_name: str,
        label_type: str,
        img_height: int,
        img_width: int,
    ):
        formatted_pred = [
            detection.to_ls(
                from_name=from_name,
                to_name=to_name,
                label_type=label_type,
                img_height=img_height,
                img_width=img_width,
            )
            for detection in detections
        ]

        max_score = max([pred["score"] for pred in formatted_pred] + [0.0])
        self.client.predictions.create(
            task=task_id,
            score=max_score,
            result=formatted_pred,
            model_version=model_tag,
        )
