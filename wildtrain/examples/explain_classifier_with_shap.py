import torch
import shap
import matplotlib.pyplot as plt
from wildtrain.models.classifier import GenericClassifier
from wildtrain.data.classification_datamodule import ClassificationDataModule
from wildtrain.explainers.shap import ClassifierSHAPExplainer
import numpy as np

# ---- User configuration ----
CHECKPOINT_PATH = 'D:/workspace/repos/wildtrain/checkpoints/classification/best-v5.ckpt'  # <-- Update this path
DATA_ROOT = 'D:/workspace/data/demo-dataset'  # <-- Update this path
BACKGROUND_SAMPLES = 50
BACKGROUND_LOADER = 'train'
N_EXPLAIN = 5  # Number of images to explain

# ---- Initialize data module ----
data_module = ClassificationDataModule(
    root_data_directory=DATA_ROOT,
)
data_module.setup('fit')

# ---- Initialize SHAP explainer ----
explainer = ClassifierSHAPExplainer(
    checkpoint_path=CHECKPOINT_PATH,
    data_module=data_module,
    background_loader=BACKGROUND_LOADER,
    background_samples=BACKGROUND_SAMPLES,
)

# ---- Sample images to explain ----
val_dataset = data_module.val_dataset
images = []
labels = []
for i in range(N_EXPLAIN):
    img, label = val_dataset[i]
    images.append(img)
    labels.append(label)
images = torch.stack(images)

print(f"Test images: {images.shape}")

# ---- Run SHAP explanation ----
shap_values = explainer.explain(images)

print(f"shap_values: {shap_values}")
print("True labels:", labels) 