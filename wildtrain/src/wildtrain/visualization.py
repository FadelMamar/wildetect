"""
FiftyOne integration for WildDetect.

This module handles dataset creation, visualization, and annotation collection
using FiftyOne for wildlife detection datasets.
"""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, ItemsView, List, Optional, Union

import fiftyone as fo
from dotenv import load_dotenv
from tqdm import tqdm
from omegaconf import OmegaConf
import supervision as sv
import numpy as np
from wildtrain.models.detector import Detector
from wildtrain.models.classifier import GenericClassifier
import torch
from PIL import Image
import torchvision.transforms.v2 as T
import math
from label_studio_sdk.client import LabelStudio
from label_studio_tools.core.utils.io import get_local_path

from urllib.parse import unquote

from .utils.io import read_image

logger = logging.getLogger(__name__)


class Visualizer:

    def __init__(self,
                 fiftyone_dataset_name: Optional[str]=None,
                 label_studio_url: Optional[str]=None,
                 label_studio_api_key: Optional[str]=None,
                 label_studio_project_id: Optional[int] = None,
                 ):

        self.fiftyone_dataset_name = fiftyone_dataset_name
        self.fiftyone_dataset = None
        self.label_studio_client = None
        self.project = None
        self.label_studio_url = None

        assert (fiftyone_dataset_name is None) ^ (label_studio_url is None), "Either fiftyone_dataset_name or label_studio_url must be provided"
        
        if fiftyone_dataset_name is not None:
            if fiftyone_dataset_name not in fo.list_datasets():
                raise ValueError(f"Dataset {fiftyone_dataset_name} not found")
            self.fiftyone_dataset = fo.load_dataset(fiftyone_dataset_name)
        else:
            assert label_studio_url is not None
            assert label_studio_api_key is not None
            assert label_studio_project_id is not None
            self.label_studio_client = LabelStudio(base_url=label_studio_url, api_key=label_studio_api_key)
            self.project = self.label_studio_client.projects.get(id=label_studio_project_id)
            self.label_studio_url = label_studio_url

    
    def _load_as_batch(self,image_paths:List[str])->torch.Tensor:
        batch_images = []
        for image_path in image_paths:
            image = read_image(image_path).convert("RGB")
            image_tensor = T.PILToTensor()(image)
            batch_images.append(image_tensor)
        return torch.stack(batch_images)
        
    def get_fiftyone_samples(self,debug:bool=False,batch_size:int=32):
        samples = list(self.fiftyone_dataset)
        num_samples = len(samples)
        if debug:
            num_samples = 25
        for i in tqdm(range(0, num_samples, batch_size),desc="Adding predictions",total=math.ceil(num_samples/batch_size)):
            batch_samples = samples[i:i+batch_size]
            batch_images = [sample.filepath for sample in batch_samples]
            if len(batch_images) > 0:
                yield self._load_as_batch(batch_images),batch_samples
            else:
                continue
    
    def get_label_studio_samples(self,debug:bool=False,batch_size:int=32):
        tasks = self.label_studio_client.tasks.list(
            project=self.project.id,
        )
        image_paths = []
        task_ids = []
        for i,task in enumerate(tasks):
            img_url = unquote(task.data["image"])
            img_path = get_local_path(
                img_url,
                download_resources=False,
                hostname=self.label_studio_url,
            )
            if not Path(img_path).exists():
                img_path = get_local_path(
                    img_url,
                    download_resources=True,
                    hostname=self.label_studio_url,
                )
            image_paths.append(img_path)
            task_ids.append(task.id)

            if debug and i > 24:
                break
        
        for i in tqdm(range(0, len(image_paths), batch_size),desc="Adding predictions",total=math.ceil(len(image_paths)/batch_size)):
            batch_image_paths = image_paths[i:i+batch_size]
            batch_tensor = self._load_as_batch(batch_image_paths)
            yield batch_tensor,task_ids[i:i+batch_size]
        
    def add_predictions_from_classifier(self, 
                                        model: GenericClassifier, 
                                        prediction_field: str = "classification_predictions", 
                                        batch_size: int = 32,
                                        debug: bool = False
                                        ):

        """Add predictions from a GenericClassifier to all samples in the FiftyOne dataset using batching."""
        
        if self.fiftyone_dataset is None:
            raise ValueError("No dataset found")
        for batch_tensor,batch_samples in self.get_fiftyone_samples(debug=debug,batch_size=batch_size):
            preds = model.predict(batch_tensor)
            for sample, pred in zip(batch_samples, preds):
                if pred is not None:
                    sample[prediction_field] = fo.Classification(label=pred["class"], confidence=pred["score"])
                    sample.save()
        self.fiftyone_dataset.save()
        
    def add_predictions_from_detector(self, 
                                    detector: Detector, 
                                    imgsz: int,
                                    model_tag:Optional[str]=None,
                                    prediction_field: str = "detection_predictions", 
                                    batch_size: int = 32,
                                    debug: bool = False,
                                    from_name:str="label",
                                    to_name:str="image",
                                    label_type:str="rectanglelabels",
                                    ):
        """Add predictions from a Detector to all samples in the FiftyOne dataset using batching."""

        def normalize_and_resize(batch_tensor:torch.Tensor):
            batch_tensor = batch_tensor / 255.
            batch_tensor = T.Resize((imgsz,imgsz),interpolation=T.InterpolationMode.NEAREST)(batch_tensor)
            return batch_tensor


        if self.fiftyone_dataset is not None:
            for batch_tensor,batch_samples in self.get_fiftyone_samples(debug=debug,batch_size=batch_size):
                batch_tensor = normalize_and_resize(batch_tensor)
                B,C,H,W = batch_tensor.shape
                detections_list = detector.predict(batch_tensor,return_as_dict=False)
                for sample, detections in zip(batch_samples, detections_list):
                    if detections is None or len(detections) == 0:
                        continue
                    fo_detections = Detector.to_fiftyone(detections,images_width=W,images_height=H)
                    sample[prediction_field] = fo.Detections(detections=fo_detections)
                    sample.save()
            self.fiftyone_dataset.save()

        else:
            for batch_tensor,task_ids in self.get_label_studio_samples(debug=debug,batch_size=batch_size):
                batch_tensor = normalize_and_resize(batch_tensor)
                B,C,H,W = batch_tensor.shape
                detections_list = detector.predict(batch_tensor,return_as_dict=False)
                for task_id, detection in zip(task_ids, detections_list):
                    if detection is None or len(detection) == 0:
                        continue
                    formatted_pred = Detector.to_label_studio(
                                                from_name=from_name,
                                                to_name=to_name,
                                                label_type=label_type,
                                                img_height=H,
                                                img_width=W,
                                                detection=detection)
                    max_score = max([pred["score"] for pred in formatted_pred] + [0.0])
                    self.label_studio_client.predictions.create(
                        task=task_id,
                        score=max_score,
                        result=formatted_pred,
                        model_version=model_tag,
                    )

