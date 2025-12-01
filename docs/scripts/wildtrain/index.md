# WildTrain Scripts Reference

This page documents all batch scripts available in the WildTrain package for model training and evaluation.

## Overview

All scripts are located in `wildtrain/scripts/` directory.

## Quick Reference

| Script | Purpose | Config File |
|--------|---------|-------------|
| [train_classifier.bat](#train-classifier) | Train classification model | `configs/classification/classification_train.yaml` |
| [eval_classifier.bat](#eval-classifier) | Evaluate classifier | `configs/classification/classification_eval.yaml` |
| [eval_detector.bat](#eval-detector) | Evaluate detector | Various detector configs |
| [train_yolo.bat](#train-yolo) | Train YOLO detector | `configs/detection/yolo_configs/yolo.yaml` |
| [train_mmdet.bat](#train-mmdet) | Train MMDetection model | `configs/detection/mmdet_configs/mmdet.yaml` |
| [register_model.bat](#register-model) | Register model to MLflow | `configs/registration/*.yaml` |
| [run_classification_pipeline.bat](#run-classification-pipeline) | Full classification pipeline | `configs/classification/classification_pipeline_config.yaml` |
| [run_detection_pipeline.bat](#run-detection-pipeline) | Full detection pipeline | `configs/detection/yolo_configs/yolo_pipeline_config.yaml` |
| [run_server.bat](#run-server) | Run inference server | `configs/inference.yaml` |
| [visualize_predictions.bat](#visualize-predictions) | Visualize predictions | Various configs |
| [create_dataset.bat](#create-dataset) | Create/prepare dataset | `configs/datapreparation/*.yaml` |

---

## Classification Scripts

### train_classifier.bat

**Purpose**: Train an image classification model using PyTorch Lightning.

**Location**: `wildtrain/scripts/train_classifier.bat`

**Command**:
```batch
uv run wildtrain train classifier -c configs\classification\classification_train.yaml
```

**Configuration**: `wildtrain/configs/classification/classification_train.yaml`

**Key Parameters**:
```yaml
model:
  architecture: "resnet50"  # resnet18, resnet50, efficientnet_b0, etc.
  num_classes: 10
  pretrained: true
  learning_rate: 0.001

data:
  root_data_directory: "D:/data/roi_dataset"
  split: "train"
  batch_size: 32
  num_workers: 4
  image_size: 224

training:
  max_epochs: 100
  accelerator: "gpu"  # gpu, cpu
  devices: 1
  precision: 16  # 16, 32
  gradient_clip_val: 1.0

callbacks:
  early_stopping:
    monitor: "val_loss"
    patience: 10
  model_checkpoint:
    monitor: "val_acc"
    mode: "max"
    save_top_k: 3

mlflow:
  experiment_name: "classification"
  tracking_uri: "http://localhost:5000"
```

**Example Usage**:
```bash
cd wildtrain

# Edit config
notepad configs\classification\classification_train.yaml

# Train
scripts\train_classifier.bat
```

**Output**:
- Trained model checkpoints
- MLflow run with metrics
- Training logs
- Best model weights

---

### eval_classifier.bat

**Purpose**: Evaluate a trained classification model.

**Location**: `wildtrain/scripts/eval_classifier.bat`

**Command**:
```batch
uv run wildtrain eval classifier -c configs\classification\classification_eval.yaml
```

**Configuration**: `wildtrain/configs/classification/classification_eval.yaml`

**Key Parameters**:
```yaml
model:
  checkpoint_path: "checkpoints/best.ckpt"
  # or load from MLflow
  mlflow_model_name: "wildlife_classifier"
  mlflow_model_version: "latest"

data:
  root_data_directory: "D:/data/roi_dataset"
  split: "test"
  batch_size: 64

evaluation:
  save_predictions: true
  generate_confusion_matrix: true
  class_metrics: true
```

**Example Usage**:
```bash
cd wildtrain
scripts\eval_classifier.bat
```

**Output**:
- Evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Per-class metrics
- Predictions file (if enabled)

---

## Detection Scripts

### train_yolo.bat

**Purpose**: Train YOLO object detection model.

**Location**: `wildtrain/scripts/train_yolo.bat` (or similar)

**Command**:
```batch
uv run wildtrain train detector -c configs\detection\yolo_configs\yolo.yaml
```

**Configuration**: `wildtrain/configs/detection/yolo_configs/yolo.yaml`

**Key Parameters**:
```yaml
model:
  framework: "yolo"
  size: "n"  # n, s, m, l, x
  pretrained: true

data:
  data_yaml: "D:/data/wildlife/data.yaml"
  # data.yaml contains:
  # - train: path to train images
  # - val: path to val images
  # - nc: number of classes
  # - names: class names

training:
  epochs: 100
  imgsz: 640
  batch: 16
  optimizer: "AdamW"
  lr0: 0.001
  device: 0  # GPU device
  
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

mlflow:
  experiment_name: "yolo_detection"
```

**Example Usage**:
```bash
cd wildtrain
scripts\train_yolo.bat
```

**YOLO Data Format**:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

**data.yaml**:
```yaml
train: ./images/train
val: ./images/val
nc: 3
names: ['elephant', 'giraffe', 'zebra']
```

---

### train_mmdet.bat

**Purpose**: Train model using MMDetection framework.

**Location**: `wildtrain/scripts/train_mmdet.bat` (or similar)

**Command**:
```batch
uv run wildtrain train detector -c configs\detection\mmdet_configs\mmdet.yaml
```

**Configuration**: `wildtrain/configs/detection/mmdet_configs/`

**Supported Models**:
- Faster R-CNN
- YOLO variants
- ATSS
- FCOS
- RetinaNet

**Example Config**:
```yaml
model:
  framework: "mmdet"
  config_file: "configs/detection/mmdet_configs/faster_rcnn.py"
  checkpoint: null  # or pretrained weights

data:
  data_root: "D:/data/coco_format"
  ann_file_train: "train.json"
  ann_file_val: "val.json"

training:
  work_dir: "work_dirs/faster_rcnn"
  max_epochs: 12
  batch_size: 2
  num_workers: 2
```

---

### eval_detector.bat

**Purpose**: Evaluate object detection model.

**Location**: `wildtrain/scripts/eval_detector.bat`

**Command**:
```batch
uv run wildtrain eval detector -c configs\detection\yolo_configs\yolo_eval.yaml
```

**Example Usage**:
```bash
cd wildtrain
scripts\eval_detector.bat
```

**Output**:
- mAP (mean Average Precision)
- mAP@50, mAP@75
- Per-class AP
- Precision-Recall curves
- Detection visualizations

---

## Pipeline Scripts

### run_classification_pipeline.bat

**Purpose**: Run complete classification training pipeline.

**Location**: `wildtrain/scripts/run_classification_pipeline.bat`

**Configuration**: `wildtrain/configs/classification/classification_pipeline_config.yaml`

**Pipeline Steps**:
1. Data validation
2. Model initialization
3. Training
4. Evaluation
5. Model registration
6. Export for deployment

**Example Config**:
```yaml
pipeline:
  name: "wildlife_classification_v1"
  
data_preparation:
  validate_data: true
  augmentation: true

training:
  config_file: "configs/classification/classification_train.yaml"
  
evaluation:
  config_file: "configs/classification/classification_eval.yaml"
  
registration:
  register_to_mlflow: true
  model_name: "wildlife_classifier"
  stage: "Staging"

export:
  export_onnx: true
  export_torchscript: true
```

---

### run_detection_pipeline.bat

**Purpose**: Run complete detection training pipeline.

**Location**: `wildtrain/scripts/run_detection_pipeline.bat`

**Configuration**: `wildtrain/configs/detection/yolo_configs/yolo_pipeline_config.yaml`

**Pipeline Includes**:
- Data preparation
- Training
- Validation
- Hyperparameter tuning (optional)
- Model registration
- Export

---

## Model Management Scripts

### register_model.bat

**Purpose**: Register trained model to MLflow model registry.

**Location**: `wildtrain/scripts/register_model.bat`

**Command**:
```batch
uv run wildtrain register <model_type> <config_file>
```

**Example Configs**:

#### Classifier Registration
```yaml
# configs/registration/classifier_registration_example.yaml
model_path: "checkpoints/best.ckpt"
model_name: "wildlife_classifier"
model_type: "classifier"

description: "ResNet50 classifier for wildlife ROI"

tags:
  architecture: "resnet50"
  dataset: "wildlife_roi_v1"
  num_classes: "10"
  training_date: "2024-01-15"
  accuracy: "0.95"

aliases:
  - "production"
  - "latest"
```

#### Detector Registration
```yaml
# configs/registration/detector_registration_example.yaml
model_path: "runs/detect/train/weights/best.pt"
model_name: "wildlife_detector"
model_type: "detector"

description: "YOLO11n detector for aerial wildlife"

tags:
  framework: "yolo"
  version: "11n"
  dataset: "wildlife_aerial_v2"
  map50: "0.89"
  training_date: "2024-01-15"
```

**Example Usage**:
```bash
cd wildtrain
scripts\register_model.bat
```

---

## Inference Scripts

### run_server.bat

**Purpose**: Run inference server for model deployment.

**Location**: `wildtrain/scripts/run_server.bat`

**Command**:
```batch
uv run wildtrain serve -c configs\inference.yaml
```

**Configuration**: `wildtrain/configs/inference.yaml`

**Key Parameters**:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 2

model:
  mlflow_model_name: "wildlife_detector"
  mlflow_model_alias: "production"
  device: "cuda"

inference:
  batch_size: 8
  confidence_threshold: 0.5
  nms_threshold: 0.45
```

**Example Usage**:
```bash
cd wildtrain
scripts\run_server.bat
```

**API Access**:
- Endpoint: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

**Example Request**:
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

predictions = response.json()
```

---

## Utility Scripts

### visualize_predictions.bat

**Purpose**: Visualize model predictions on images.

**Location**: `wildtrain/scripts/visualize_predictions.bat`

**Configuration**: Various visualization configs

**Features**:
- Draw bounding boxes
- Show confidence scores
- Save annotated images
- Generate prediction gallery

**Example Usage**:
```bash
cd wildtrain
scripts\visualize_predictions.bat
```

---

### create_dataset.bat

**Purpose**: Create or prepare dataset for training.

**Location**: `wildtrain/scripts/create_dataset.bat`

**Configuration**: `wildtrain/configs/datapreparation/`

**Capabilities**:
- Convert formats
- Split train/val/test
- Apply transformations
- Validate dataset

---

## Common Workflows

### Training Workflow (Classification)

```bash
cd wildtrain

# 1. Prepare data (if needed)
scripts\create_dataset.bat

# 2. Train model
scripts\train_classifier.bat

# 3. Evaluate
scripts\eval_classifier.bat

# 4. Register to MLflow
scripts\register_model.bat
```

### Training Workflow (Detection)

```bash
cd wildtrain

# 1. Train YOLO
scripts\train_yolo.bat

# 2. Evaluate
scripts\eval_detector.bat

# 3. Visualize predictions
scripts\visualize_predictions.bat

# 4. Register model
scripts\register_model.bat
```

### Hyperparameter Tuning

```python
# Use Optuna for HPO
cd wildtrain
uv run wildtrain tune classifier -c configs/classification/classification_sweep.yaml
```

### Model Deployment

```bash
# 1. Register model
scripts\register_model.bat

# 2. Start inference server
scripts\run_server.bat

# 3. Test API
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"
```

---

## Configuration Examples

### Complete Training Config

```yaml
# configs/classification/classification_train.yaml
model:
  architecture: "resnet50"
  num_classes: 10
  pretrained: true
  learning_rate: 0.001
  weight_decay: 0.0001
  dropout: 0.5

data:
  root_data_directory: "D:/data/roi_dataset"
  split: "train"
  batch_size: 32
  num_workers: 4
  image_size: 224
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

augmentation:
  random_flip: 0.5
  random_rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
  random_crop: true

training:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1
  precision: 16
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  log_every_n_steps: 10

optimizer:
  type: "Adam"
  lr: 0.001
  betas: [0.9, 0.999]

scheduler:
  type: "CosineAnnealingLR"
  T_max: 100
  eta_min: 0.00001

callbacks:
  early_stopping:
    monitor: "val_loss"
    patience: 10
    mode: "min"
  model_checkpoint:
    monitor: "val_acc"
    mode: "max"
    save_top_k: 3
    filename: "epoch{epoch:02d}-acc{val_acc:.4f}"

mlflow:
  experiment_name: "wildlife_classification"
  tracking_uri: "http://localhost:5000"
  log_models: true
```

---

## Troubleshooting

### Training Crashes

**Issue**: Training crashes with CUDA out of memory

**Solutions**:
1. Reduce batch size
2. Use mixed precision (`precision: 16`)
3. Reduce image size
4. Enable gradient accumulation

### MLflow Connection Error

**Issue**: Can't connect to MLflow tracking server

**Solutions**:
1. Start MLflow server: `mlflow server --port 5000`
2. Check `MLFLOW_TRACKING_URI` environment variable
3. Verify network connection
4. Check firewall settings

### Data Loading Slow

**Issue**: Data loading is bottleneck

**Solutions**:
1. Increase `num_workers` (max 4 on Windows)
2. Enable `pin_memory: true`
3. Use SSD for data storage
4. Reduce data preprocessing complexity

### Model Won't Load

**Issue**: Can't load trained model

**Solutions**:
1. Check checkpoint path is correct
2. Verify model architecture matches
3. Check for version compatibility
4. Try loading with `strict=False`

---

## Next Steps

- [WildTrain Configuration Reference](../../configs/wildtrain/index.md)
- [Model Training Tutorial](../../tutorials/model-training.md)
- [WildTrain CLI Reference](../../api-reference/wildtrain-cli.md)
- [MLflow Integration Guide](../../architecture/wildtrain.md#model-registration)

