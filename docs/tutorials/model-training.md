# Model Training Tutorial

Learn how to train detection and classification models using WildTrain.

## Prerequisites

- WildTrain installed (`uv pip install -e .` in the `wildtrain/` directory)
- Prepared dataset (see [Dataset Preparation](dataset-preparation.md))
- MLflow server running (optional - see [Step 0: Start MLflow](#step-0-start-mlflow-optional))

## Step 0: Start MLflow (Optional)

Before you begin training, it is recommended to launch the MLflow tracking server to monitor your experiments in real-time.

```bash
cd wildetect
scripts\launch_mlflow.bat
```

Access at: [http://localhost:5000](http://localhost:5000)

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

Edit `configs/detection/yolo_configs/yolo.yaml`. WildTrain validates that file as a `DetectionConfig` (dataset, model, train, curriculum, pretraining, mlflow). It is not a flat Ultralytics CLI yaml, and it is not an MMDetection config.

Keep `train.single_cls: true`. That trains the class-agnostic localizer the ROI classifier and the census app expect. Multi-class YOLO is not the supported detector.

### Step 3: Train

```bash
wildtrain run -c configs/detection/yolo_configs/yolo.yaml
```

Or use the provided script:

```bash
wildtrain\scripts\train_detector.bat
```

### Step 4: Evaluate

```bash
wildtrain evaluate -c configs/detection/yolo_configs/yolo_eval.yaml
```

`configs/detection/yolo_configs/eval_config.yaml` is the same eval document. `wildtrain evaluate yolo-model` has been removed.

### Step 5: Register Model to MLflow

```bash
wildtrain register -c configs/registration/detector_registration_example.yaml
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
task: classification
name: baseline_dinov2
tracking:
  mlflow:
    experiment_name: wildtrain_classification
    run_name: baseline_dinov2
    log_model: true
data:
  format: roi
  root: ${oc.env:DATA_ROOT,.}
  input_size: 384
  batch_size: 64
  rebalance: true
  label_policy:
    mode: collapse
    positive_name: wildlife
    background_name: background
    discard: ["rocks", "vegetation", "other"]
  stats:
    mean: [0.554, 0.469, 0.348]
    std: [0.203, 0.173, 0.144]
  transforms:
    train:
      - name: Resize
        params: { size: ${data.input_size} }
      - name: RandomHorizontalFlip
        params: { p: 0.5 }
    val:
      - name: Resize
        params: { size: ${data.input_size} }
model:
  backend: timm
  backbone: timm/vit_base_patch14_reg4_dinov2.lvd142m
  pretrained: true
  dropout: 0.2
  freeze_backbone: true
train:
  epochs: 20
  lr: 1e-3
  weight_decay: 1e-3
  precision: bf16-mixed
  accelerator: auto
checkpoint:
  monitor: val_f1score
  mode: max
  patience: 10
  dirpath: checkpoints/classification
  filename: best
```

See [Classification Training Config Reference](../configs/wildtrain/classification-train.md) for all fields.

### Step 4: Train

```bash
wildtrain run -c configs/classification/classification_train.yaml
```

### Step 5: Evaluate

```bash
wildtrain evaluate -c configs/classification/classification_eval.yaml
```

### Step 6: Register Model to MLflow

```bash
wildtrain register -c configs/registration/classifier_registration_example.yaml
```

---

### Check the Results in MLflow

Once your training is complete, access the MLflow UI at [http://localhost:5000](http://localhost:5000) to view:

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
