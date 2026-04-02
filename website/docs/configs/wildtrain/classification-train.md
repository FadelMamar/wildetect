# WildTrain Classification Training Configuration

Detailed reference for the classification training YAML configuration file.

## Overview

The classification training config controls the full classifier training pipeline: dataset loading, model architecture, training hyperparameters, MLflow tracking, checkpointing, and curriculum learning.

**Usage:**

```bash
wildtrain train classifier -c configs/classification/classification_train.yaml
```

---

## Configuration Sections

### `dataset` — Data Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `root_data_directory` | `str` | — | Path to the root data directory |
| `dataset_type` | `str` | `roi` | Dataset type: `roi` (pre-computed crops) or `crop` (dynamic extraction) |
| `input_size` | `int` | `384` | Input image size for the model |
| `batch_size` | `int` | `64` | Training batch size |
| `rebalance` | `bool` | `true` | Rebalance class distribution via oversampling |

#### Dataset Statistics

Used for image normalization:

```yaml
dataset:
  stats:
    mean: [0.554, 0.469, 0.348]
    std: [0.203, 0.173, 0.144]
```

Use `wildtrain dataset stats DATA_DIR` to compute these values.

#### Transforms

Define per-split torchvision transforms:

```yaml
dataset:
  transforms:
    train:
      - name: Resize
        params:
          size: ${dataset.input_size}
      - name: RandomHorizontalFlip
        params:
          p: 0.5
      - name: ColorJitter
        params:
          brightness: 0.1
          contrast: 0.0
          saturation: 0.0
      - name: RandomRotation
        params:
          degrees: 45
    val:
      - name: Resize
        params:
          size: ${dataset.input_size}
```

#### Single-Class Mode

Merge all species into a binary classifier (wildlife vs. background):

```yaml
dataset:
  single_class:
    enable: true
    background_class_name: "background"
    single_class_name: "wildlife"
    keep_classes: null          # List of classes to keep (null = all)
    discard_classes: ["rocks", "vegetation", "other"]
```

#### Crop Dataset Parameters

Used when `dataset_type: "crop"`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `crop_size` | `int` | `${dataset.input_size}` | Crop size for dynamic extraction |
| `max_tn_crops` | `int` | `1` | Max true-negative crops per image |
| `p_draw_annotations` | `float` | `0.2` | Probability of drawing annotations on crops |
| `compute_difficulties` | `bool` | `true` | Compute sample difficulty scores |
| `preserve_aspect_ratio` | `bool` | `true` | Preserve aspect ratio during cropping |

#### Curriculum Learning

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `curriculum_config.enabled` | `bool` | `false` | Enable curriculum learning |
| `curriculum_config.type` | `str` | `difficulty` | Curriculum type |
| `curriculum_config.difficulty_strategy` | `str` | `linear` | Difficulty progression strategy |
| `curriculum_config.start_difficulty` | `float` | `0.0` | Starting difficulty |
| `curriculum_config.end_difficulty` | `float` | `1.0` | Ending difficulty |
| `curriculum_config.warmup_epochs` | `int` | `0` | Warmup epochs before curriculum starts |

---

### `model` — Model Architecture

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone` | `str` | — | Backbone model identifier (e.g., `timm/vit_base_patch14_reg4_dinov2.lvd142m`) |
| `pretrained` | `bool` | `true` | Use pretrained weights |
| `backbone_source` | `str` | `timm` | Source library: `timm` |
| `dropout` | `float` | `0.2` | Dropout rate |
| `freeze_backbone` | `bool` | `true` | Freeze backbone weights (train head only) |
| `input_size` | `int` | `${dataset.input_size}` | Model input size |
| `mean` | `list` | `${dataset.stats.mean}` | Normalization mean |
| `std` | `list` | `${dataset.stats.std}` | Normalization std |
| `hidden_dim` | `int` | `128` | Hidden layer dimension |
| `num_layers` | `int` | `2` | Number of classification head layers |
| `weights` | `str` | `None` | Path to pretrained checkpoint for warm-start |

---

### `train` — Training Hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `${dataset.batch_size}` | Training batch size |
| `epochs` | `int` | `20` | Number of training epochs |
| `lr` | `float` | `1e-3` | Learning rate |
| `label_smoothing` | `float` | `0.0` | Label smoothing factor |
| `weight_decay` | `float` | `1e-3` | Weight decay |
| `lrf` | `float` | `1e-2` | Final learning rate factor |
| `precision` | `str` | `bf16-mixed` | Training precision: `bf16-mixed`, `16-mixed`, `32` |
| `accelerator` | `str` | `auto` | PyTorch Lightning accelerator: `auto`, `gpu`, `cpu` |
| `num_workers` | `int` | `4` | DataLoader workers |
| `val_check_interval` | `int` | `2` | Validation check interval (epochs) |

---

### `mlflow` — Experiment Tracking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_name` | `str` | — | MLflow experiment name |
| `run_name` | `str` | — | MLflow run name |
| `log_model` | `bool` | `true` | Log model artifacts to MLflow |

---

### `checkpoint` — Model Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `monitor` | `str` | `val_f1score` | Metric to monitor for checkpointing |
| `save_top_k` | `int` | `1` | Number of top checkpoints to keep |
| `mode` | `str` | `max` | Optimization direction: `max` or `min` |
| `save_last` | `bool` | `true` | Always save last epoch checkpoint |
| `dirpath` | `str` | `checkpoints/classification` | Checkpoint save directory |
| `patience` | `int` | `10` | Early stopping patience |
| `save_weights_only` | `bool` | `true` | Save only model weights (not optimizer state) |
| `filename` | `str` | `best` | Checkpoint filename pattern |
| `min_delta` | `float` | `0.001` | Minimum improvement for early stopping |

---

## Complete Example

```yaml
dataset:
  root_data_directory: D:/data
  dataset_type: "roi"
  input_size: 384
  batch_size: 64
  rebalance: true
  stats:
    mean: [0.554, 0.469, 0.348]
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
    discard_classes: ["rocks", "vegetation"]

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

---

**See also:**

- [WildTrain CLI Reference](../../api-reference/wildtrain-cli.md) — `train classifier` command
- [Model Training Tutorial](../../tutorials/model-training.md)
