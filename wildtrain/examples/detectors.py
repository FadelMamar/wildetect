# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:28:12 2025

@author: FADELCO
"""

# import os

import torch
import wildtrain
from wildtrain.models.localizer import UltralyticsLocalizer
from wildtrain.models.classifier import GenericClassifier
from wildtrain.models.detector import Detector
from wildtrain.data import load_image
from PIL import Image

import supervision as sv

# Example label map for classifier
# label_to_class_map = {0: "cat", 1: "dog"}

# Instantiate the localizer (YOLO weights path or model name required)
device = "cpu"
# localizer = UltralyticsLocalizer(weights="D:/workspace/repos/wildtrain/models/best.pt", 
#                                  conf_thres=0.2,
#                                  iou_thres=0.5,
#                                  imgsz=800,
#                                  device=device)

# Instantiate the classifier
classifier = GenericClassifier.load_from_checkpoint("checkpoints/classification/unified_classifier/best.ckpt",
                                                    map_location=device
                                                    )


# classifier = GenericClassifier(label_to_class_map={0:"background",1:"wildlife"},
#                                num_layers=2,
#                                hidden_dim=128                               
#                                )

# classifier = classifier.to_torchscript()

classifier = classifier.to_onnx(output_path="checkpoints/classification/unified_classifier/best.onnx",
                                batch_size=8,
                                report=False)

classifier = GenericClassifier._load_onnx(onnx_path="checkpoints/classification/unified_classifier/best.onnx",
                                                      label_to_class_map={0:"background",1:"wildlife"})

o = classifier.predict(torch.rand(1,3,384,384))

print(o)

# Create the two-stage detector
# classifier = None
# model = Detector(localizer=localizer, classifier=classifier)

# # Dummy input: batch of 2 RGB images, 3x224x224
# path = r"D:\workspace\data\demo-dataset\savmap\images\train\00a033fefe644429a1e0fcffe88f8b39_0_augmented_0_tile_12_832_832.jpg"
# images = load_image(path).unsqueeze(0) / 255.

# # Run detection
# detections = model.predict(images)

# # Print results
# for i, det in enumerate(detections):
#     print(f"Image {i} detections:")
#     print(det)
    
# box_annotator = sv.BoxAnnotator()

# annotated_frame = box_annotator.annotate(
#     scene=Image.open(path).convert("RGB"),
#     detections=detections[0]
# )
