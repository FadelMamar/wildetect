# WildTrain Configuration Reference

Documentation for all WildTrain configuration files used in model training.

## Configuration Structure

WildTrain uses Hydra for hierarchical configuration management.

```
configs/
├── classification/          # Classification configs
│   ├── classification_train.yaml
│   ├── classification_eval.yaml
│   ├── classification_sweep.yaml
│   └── classification_pipeline_config.yaml
├── detection/              # Detection configs
│   ├── yolo_configs/
│   │   ├── yolo.yaml
│   │   ├── yolo_eval.yaml
│   │   └── data/demo.yaml
│   └── mmdet_configs/
│       ├── mmdet.yaml
│       └── [various model configs]
├── datapreparation/        # Data prep configs
│   ├── import-config-example.yaml
│   └── savmap.yaml
├── registration/           # Model registration
│   ├── classifier_registration_example.yaml
│   └── detector_registration_example.yaml
├── main.yaml              # Main config
└── inference.yaml         # Inference config
```

---

## Classification Configs

### classification_train.yaml

**Purpose**: Configure classification model training

```yaml
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

training:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1
  precision: 16
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

### classification_eval.yaml

```yaml
model:
  checkpoint_path: "checkpoints/best.ckpt"
  mlflow_model_name: "classifier"

data:
  root_data_directory: "D:/data/roi_dataset"
  split: "test"
  batch_size: 64

evaluation:
  save_predictions: true
  generate_confusion_matrix: true
```

---

## Detection Configs

### YOLO Configuration

**File**: `configs/detection/yolo_configs/yolo.yaml`

```yaml
model:
  framework: "yolo"
  size: "n"  # n, s, m, l, x
  pretrained: true

data:
  data_yaml: "D:/data/wildlife/data.yaml"

training:
  epochs: 100
  imgsz: 640
  batch: 16
  optimizer: "AdamW"
  lr0: 0.001
  device: 0

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

mlflow:
  experiment_name: "yolo_detection"
```

**YOLO Data YAML**:
```yaml
# data/wildlife/data.yaml
train: ./images/train
val: ./images/val
nc: 3
names: ['elephant', 'giraffe', 'zebra']
```

### MMDetection Configuration

**File**: `configs/detection/mmdet_configs/mmdet.yaml`

```yaml
model:
  framework: "mmdet"
  config_file: "configs/detection/mmdet_configs/faster_rcnn.py"

data:
  data_root: "D:/data/coco_format"
  ann_file_train: "train.json"
  ann_file_val: "val.json"

training:
  work_dir: "work_dirs/faster_rcnn"
  max_epochs: 12
```

---

## Registration Configs

### Classifier Registration

**File**: `configs/registration/classifier_registration_example.yaml`

```yaml
model_path: "checkpoints/best.ckpt"
model_name: "wildlife_classifier"
model_type: "classifier"

description: "ResNet50 classifier for wildlife ROI"

tags:
  architecture: "resnet50"
  dataset: "wildlife_roi_v1"
  accuracy: "0.95"

aliases:
  - "production"
```

### Detector Registration

**File**: `configs/registration/detector_registration_example.yaml`

```yaml
model_path: "runs/detect/train/weights/best.pt"
model_name: "wildlife_detector"
model_type: "detector"

description: "YOLO11n for aerial wildlife"

tags:
  framework: "yolo"
  map50: "0.89"

aliases:
  - "production"
```

---

## Inference Configuration

**File**: `configs/inference.yaml`

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

---

## Main Configuration

**File**: `configs/main.yaml`

```yaml
defaults:
  - model: yolo
  - data: detection
  - training: default

experiment_name: "wildlife_detection"
seed: 42
```

---

## Hyperparameter Tuning

### classification_sweep.yaml

```yaml
method: bayes  # grid, random, bayes
metric:
  name: val_acc
  goal: maximize

parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [16, 32, 64]
  architecture:
    values: ["resnet18", "resnet50"]
```

---

## Best Practices

1. **Use MLflow** for experiment tracking
2. **Save checkpoints** frequently
3. **Enable early stopping** to prevent overfitting
4. **Use mixed precision** (precision: 16) for faster training
5. **Version your configs** with git

---

## Next Steps

- [WildTrain Scripts](../../scripts/wildtrain/index.md)
- [Training Tutorial](../../tutorials/model-training.md)
- [WildTrain CLI](../../api-reference/wildtrain-cli.md)

