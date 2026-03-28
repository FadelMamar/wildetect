# Model Training Tutorial

Learn how to train detection and classification models using WildTrain.

## Prerequisites

- WildTrain installed (`uv pip install -e .` in the `wildtrain/` directory)
- Prepared dataset (see [Dataset Preparation](dataset-preparation.md))
- MLflow server running (optional, for experiment tracking)

## Training a YOLO Detector

### Step 1: Prepare Data

Ensure your dataset is in YOLO format:

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

Create or edit a YOLO training config (e.g., `configs/detection/yolo_configs/yolo.yaml`). The config follows the [Ultralytics format](https://docs.ultralytics.com/usage/cfg/):

```yaml
model: yolo12n.pt        # Pretrained model
data: configs/detection/yolo_configs/data/wildlife.yaml
epochs: 100
imgsz: 800
batch: 16
device: 0                # GPU index
lr0: 0.001
lrf: 0.01
optimizer: AdamW
weight_decay: 0.0005
```

### Step 3: Train

```bash
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml
```

Or use the provided script:

```bash
wildtrain\scripts\train_detector.bat
```

### Step 4: Evaluate

```bash
wildtrain evaluate detector -c configs/detection/yolo_configs/yolo_eval.yaml
```

You can also run direct Ultralytics evaluation:

```bash
wildtrain evaluate yolo-model -c configs/detection/yolo_configs/eval_config.yaml
```

### Step 5: Register Model to MLflow

```bash
wildtrain register detector configs/registration/detector_registration_example.yaml
```

See [Registration Config Reference](../configs/wildtrain/registration.md) for config details.

---

## Training a Classifier

### Step 1: Prepare ROI Dataset

Use WilData to create an ROI dataset from detection annotations:

```bash
wildata create-roi-dataset -c configs/roi-create-config.yaml
```

This generates a classification-ready directory structure:

```
data/<dataset_name>/roi/<split>/<class_name>/<images>
```

### Step 2: Compute Dataset Statistics

```bash
wildtrain dataset stats D:/data/roi_dataset --split train -o stats.json
```

Use the computed mean and std in your training config.

### Step 3: Configure Training

Create or edit `configs/classification/classification_train.yaml`:

```yaml
dataset:
  root_data_directory: D:/data
  dataset_type: "roi"
  input_size: 384
  batch_size: 64
  rebalance: true
  stats:
    mean: [0.554, 0.469, 0.348]   # From step 2
    std: [0.203, 0.173, 0.144]
  transforms:
    train:
      - name: Resize
        params: { size: ${dataset.input_size} }
      - name: RandomHorizontalFlip
        params: { p: 0.5 }
    val:
      - name: Resize
        params: { size: ${dataset.input_size} }
  single_class:
    enable: true
    background_class_name: "background"
    single_class_name: "wildlife"
    discard_classes: ["rocks", "vegetation", "other"]

model:
  backbone: timm/vit_base_patch14_reg4_dinov2.lvd142m
  pretrained: true
  backbone_source: timm
  dropout: 0.2
  freeze_backbone: true
  hidden_dim: 128
  num_layers: 2

mlflow:
  experiment_name: wildtrain_classification
  run_name: "baseline_dinov2"
  log_model: true

train:
  epochs: 20
  lr: 1e-3
  weight_decay: 1e-3
  precision: bf16-mixed
  accelerator: auto
  num_workers: 4

checkpoint:
  monitor: val_f1score
  save_top_k: 1
  mode: max
  patience: 10
  dirpath: checkpoints/classification
  filename: "best"
```

See [Classification Training Config Reference](../configs/wildtrain/classification-train.md) for all fields.

### Step 4: Train

```bash
wildtrain train classifier -c configs/classification/classification_train.yaml
```

### Step 5: Evaluate

```bash
wildtrain evaluate classifier -c configs/classification/classification_eval.yaml
```

### Step 6: Register Model to MLflow

```bash
wildtrain register classifier -c configs/registration/classifier_registration_example.yaml
```

---

## Monitor with MLflow

Start the MLflow tracking server:

```bash
# Using the provided script
scripts\launch_mlflow.bat

# Or directly
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlartifacts --port 5000
```

Access the MLflow UI at [http://localhost:5000](http://localhost:5000) to view:

- Training metrics (loss, accuracy, F1)
- Hyperparameters
- Model artifacts and checkpoints
- Model registry

## Hyperparameter Sweep

Run an Optuna-based hyperparameter sweep:

```bash
# Classification sweep
wildtrain pipeline classification -c configs/classification/classification_pipeline_config.yaml

# Detection sweep
wildtrain pipeline detection -c configs/detection/detection_sweep.yaml
```

## Validate Configs Before Training

Check your config for errors before starting a long training run:

```bash
wildtrain config validate configs/classification/classification_train.yaml --type classification
```

---

**Next Steps:**

- [End-to-End Detection](end-to-end-detection.md) — Use trained models for detection
- [Classification Training Config](../configs/wildtrain/classification-train.md) — All config fields
- [Detection Training Config](../configs/wildtrain/detection-train.md) — YOLO config details
- [WildTrain CLI Reference](../api-reference/wildtrain-cli.md) — Full CLI documentation
