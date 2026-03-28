# WildTrain Detection Training Configuration

Detailed reference for detection training, evaluation, and sweep YAML configuration files.

## Overview

WildTrain supports YOLO-based object detection via [Ultralytics](https://docs.ultralytics.com/). Training configs define the model, data, training hyperparameters, and MLflow tracking.

**Usage:**

```bash
# Train
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml

# Evaluate
wildtrain evaluate detector -c configs/detection/yolo_configs/yolo_eval.yaml

# Hyperparameter sweep
wildtrain pipeline detection -c configs/detection/detection_sweep.yaml
```

---

## Training Config (`yolo.yaml`)

The YOLO training config follows the [Ultralytics configuration format](https://docs.ultralytics.com/usage/cfg/) with additional WildTrain fields.

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Path to YOLO model config or pretrained weights |
| `data` | `str` | Path to YOLO data YAML file |
| `epochs` | `int` | Number of training epochs |
| `imgsz` | `int` | Input image size |
| `batch` | `int` | Batch size |
| `device` | `str/int` | Device: `0` (GPU 0), `cpu`, etc. |
| `lr0` | `float` | Initial learning rate |
| `lrf` | `float` | Final learning rate factor |
| `optimizer` | `str` | Optimizer: `SGD`, `Adam`, `AdamW` |
| `weight_decay` | `float` | Weight decay for regularization |

### Data YAML Format

The `data` field points to a standard YOLO data config:

```yaml
train: ./images/train
val: ./images/val
nc: 3
names: ['elephant', 'giraffe', 'zebra']
```

---

## Evaluation Config (`yolo_eval.yaml`)

Used with `wildtrain evaluate detector -c CONFIG`.

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Path to trained YOLO weights |
| `data` | `str` | Path to YOLO data YAML |
| `imgsz` | `int` | Input image size |
| `batch` | `int` | Batch size |
| `device` | `str/int` | Device for evaluation |

---

## Hyperparameter Sweep Config

Used with `wildtrain pipeline detection -c CONFIG` for Optuna-based hyperparameter optimization.

### Sweep Structure

```yaml
# Base config used as template
base_config: configs/detection/yolo_configs/yolo.yaml

# Search space
parameters:
  model: null          # Model variants (optional)
  train:
    lr0: [0.0001, 0.001, 0.01]
    lrf: [0.01, 0.1, 0.5]
    batch: [8, 16, 32]
    epochs: [10, 20, 30]
    imgsz: [640, 1280]
    optimizer: [SGD, AdamW, Adam]
    weight_decay: [0.0001, 0.0005, 0.001]
```

### Sweep Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_config` | `str` | — | Path to base training config |
| `sweep_name` | `str` | — | Name for the sweep experiment |
| `n_trials` | `int` | `20` | Number of Optuna trials |
| `seed` | `int` | `42` | Random seed |
| `objective` | `str` | `map_50` | Metric to optimize: `precision`, `recall`, `f1_score`, `map`, `map_50`, `map_50_95`, `fitness` |
| `direction` | `str` | `maximize` | Optimization direction: `minimize` or `maximize` |
| `timeout` | `int/null` | `null` | Maximum optimization time in seconds |

### Output Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output.directory` | `str` | `results/sweeps/{name}` | Results output directory |
| `output.save_results` | `bool` | `true` | Save trial results |
| `output.save_plots` | `bool` | `true` | Save optimization plots |
| `output.format` | `str` | `both` | Output format: `json`, `csv`, or `both` |
| `output.include_optimization_history` | `bool` | `true` | Include full optimization history |

---

## Complete Training Example

```yaml
# Model
model: yolo12n.pt
data: configs/detection/yolo_configs/data/wildlife.yaml

# Training
epochs: 100
imgsz: 800
batch: 16
device: 0

# Learning rate
lr0: 0.001
lrf: 0.01
optimizer: AdamW
weight_decay: 0.0005

# Augmentation (built into Ultralytics)
augment: true
mosaic: 1.0
mixup: 0.0
```

---

**See also:**

- [WildTrain CLI Reference](../../api-reference/wildtrain-cli.md) — `train detector`, `evaluate detector` commands
- [YOLO Config Guide](yolo-config-guide.md)
- [Model Training Tutorial](../../tutorials/model-training.md)
