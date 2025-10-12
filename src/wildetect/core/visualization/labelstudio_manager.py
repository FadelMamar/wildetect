import logging
from typing import Dict, List, Optional
from urllib.parse import unquote

from label_studio_sdk.client import LabelStudio
from label_studio_tools.core.utils.io import get_local_path
from tqdm import tqdm

from ..data import Detection, DroneImage

logger = logging.getLogger(__name__)


class LabelStudioManager:
    def __init__(
        self,
        url: str,
        api_key: str,
        download_resources: bool = False,
        label_config: Optional[str] = None,
    ):
        self.client = LabelStudio(base_url=url, api_key=api_key)
        self.download_resources = download_resources
        self.url = url

        self.label_config = (
            label_config
            or """<View>
                                    <Image name="image" value="$image"/>
                                    <RectangleLabels name="detections" toName="image">
                                        <Label value="wildlife"/>
                                        <Label value="background"/>
                                    </RectangleLabels>
                                    </View>
                                """
        )
        self.from_name = "detections"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

    def get_project(self, project_id: int):
        return self.client.projects.get(project_id)

    def delete_project(self, project_id: int):
        return self.client.projects.delete(id=project_id)

    def create_local_project(self, project_name: str, path: str) -> int:
        r = self.client.projects.create(
            title=project_name, label_config=self.label_config
        )
        resp_storage = self.client.import_storage.local.create(
            project=r.id,
            path=path,
            use_blob_urls=True,
        )
        self.client.import_storage.local.sync(id=resp_storage.id)
        return r.id

    def get_image_path(self, task):
        url = unquote(task.data["image"])
        return get_local_path(
            url,
            download_resources=self.download_resources,
            hostname=self.url,
        )

    def get_tasks(self, project_id: int):
        return self.client.tasks.list(project=project_id)

    def get_tasks_paths(self, project_id: int) -> Dict[str, int]:
        tasks = self.get_tasks(project_id)
        # return {task.storage_filename: task.id for task in tasks}
        return {self.get_image_path(task): task.id for task in tasks}

    def get_detections(
        self,
        task,
    ) -> Optional[dict]:
        annotations = self.client.annotations.list(id=task.id)
        predictions = self.client.predictions.list(task=task.id)
        image_path = task.storage_filename

        if len(annotations) == 0 and len(predictions) == 0:
            return None

        return dict(
            image_path=image_path, annotations=annotations, predictions=predictions
        )

    def get_all_project_detections(
        self,
        project_id: int,
    ) -> List[dict]:
        tasks = self.get_tasks(project_id)
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

    def upload_drone_images(
        self,
        drone_images: List[DroneImage],
        project_name: str,
        image_dir: str,
        model_tag: str,
    ) -> int:
        project_id = self.create_local_project(
            project_name=project_name, path=image_dir
        )
        image_paths_task_ids = self.get_tasks_paths(project_id)

        def upload_one_image(drone_image: DroneImage):
            try:
                self.upload_detections(
                    image_paths_task_ids.get(drone_image.image_path),
                    detections=drone_image.get_non_empty_predictions(),
                    model_tag=model_tag,
                    img_height=drone_image.height,
                    img_width=drone_image.width,
                )
            except Exception as e:
                pass

        for drone_image in tqdm(
            drone_images, desc="Uploading detections to Label Studio"
        ):
            upload_one_image(drone_image)

        return project_id

    def upload_detections(
        self,
        task_id: int,
        detections: List[Detection],
        model_tag: str,
        img_height: int,
        img_width: int,
    ):
        formatted_pred = [
            detection.to_ls(
                from_name=self.from_name,
                to_name=self.to_name,
                label_type=self.label_type,
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
