# Model Training Tutorial

Learn how to train detection and classification models using WildTrain.

## Prerequisites

- WildTrain installed
- Prepared dataset (see [Dataset Preparation](dataset-preparation.md))
- MLflow server running

## Training a YOLO Detector

### Step 1: Prepare Data

Ensure YOLO format:

```
data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml`:
```yaml
train: ./images/train
val: ./images/val
nc: 3
names: ['elephant', 'giraffe', 'zebra']
```

### Step 2: Configure Training

Create `configs/yolo_train.yaml`:

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
  device: 0

mlflow:
  experiment_name: "yolo_wildlife"
```

### Step 3: Train

```bash
cd wildtrain
scripts\train_yolo.bat

# Or with CLI
wildtrain train detector -c configs/yolo_train.yaml
```

### Step 4: Evaluate

```bash
wildtrain eval detector -c configs/yolo_eval.yaml
```

### Step 5: Register Model

```bash
wildtrain register detector configs/registration/detector_registration.yaml
```

## Training a Classifier

### Step 1: Prepare ROI Dataset

Use WilData to create ROI dataset from detections.

### Step 2: Configure

`configs/classification_train.yaml`:

```yaml
model:
  architecture: "resnet50"
  num_classes: 10
  pretrained: true
  learning_rate: 0.001

data:
  root_data_directory: "D:/data/roi_dataset"
  batch_size: 32

training:
  max_epochs: 100
  accelerator: "gpu"
```

### Step 3: Train

```bash
cd wildtrain
scripts\train_classifier.bat
```

## Monitor with MLflow

```bash
# Start MLflow
scripts\launch_mlflow.bat

# Access at http://localhost:5000
```

View:
- Training metrics
- Model performance
- Hyperparameters
- Artifacts

## Complete Python Example

```python
from wildtrain import Trainer

# Configure
config = Trainer.load_config("configs/yolo_train.yaml")

# Train
trainer = Trainer(config)
model = trainer.train()

# Evaluate
metrics = trainer.evaluate()

# Register
trainer.register_model(
    model_name="wildlife_detector",
    tags={"dataset": "wildlife_v1"}
)
```

---

**Next Steps:**
- [End-to-End Detection](end-to-end-detection.md)
- [WildTrain Scripts](../scripts/wildtrain/index.md)

