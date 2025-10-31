"""
Feature extraction for image filtering algorithms.

This module provides feature extraction capabilities for clustering
and filtering algorithms in object detection training data selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import timm
import torch.nn as nn
import torchvision.transforms as T


class FeatureExtractor(nn.Module):
    """
    Feature extractor.
    """

    def __init__(
        self,
        backbone: str = "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        backbone_source: str = "timm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
        use_cls_token: bool = True,
    ):
        """
        Initialize the feature extractor.
        Args:
            model_name: timm model name (default: 'timm/vit_small_patch16_224.dino')
            device: Device to run inference on ('cpu', 'cuda',)
        """
        super().__init__()
        if backbone_source != "timm":
            raise ValueError(f"Backbone source must be 'timm', got {backbone_source}")
        self.backbone = backbone
        self.model = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0
            )
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        trfs = nn.Sequential(*[t for t in transform.transforms if isinstance(t, (T.Normalize,T.Resize))])
        self.transform = trfs
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.use_cls_token = use_cls_token
        self.pil_to_tensor = T.PILToTensor()
        self.eval()
    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        return self.model.num_features

    @torch.no_grad()
    def forward(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from a list of images.
        Args:
            image_paths: List of image file paths
        Returns:
            Features as a numpy array
        """
        if isinstance(images, torch.Tensor):
            images = images.float()
            images = self.transform(images).to(self.device)
        else:
            for image in images:
                assert isinstance(image, Image.Image), f"Image must be a PIL Image. Received {type(image)}"
            images = torch.stack([self.pil_to_tensor(image) for image in images],dim=0)
            images = images.float()
            images = self.transform(images).to(self.device)
        
        return self._forward(images)
    
    def _forward(self,images:torch.Tensor) -> torch.Tensor:
        if "vit" in self.backbone and self.use_cls_token: # get CLS token for ViT models
            x = self.model.forward_features(images)[:,0,:]
        else:
            x = self.model(images)
        return x.cpu()
