# WildTrain

WildTrain trains the two models the census app uses:

- a class-agnostic Ultralytics YOLO localizer (`task: detection`, `train.single_cls: true`)
- an ROI image classifier on crops from those boxes (`task: classification`, `data.format: roi`)

An experiment is one YAML document. OmegaConf resolves interpolations such as `${data.input_size}` before validation. `wildtrain run -c` and `wildtrain sweep -c` dispatch through the harness registry. `POST /experiments` is the only training route.

Flat training YAML, `wildtrain train classifier`, `wildtrain train detector`, `base_config` sweep files, and `POST /training/classifier` / `POST /training/detector` are gone. Loading an old file raises an error that names the replacement fields.

## Layout

```
configs/          # experiment YAML, plus eval, registration, and visualization
src/wildtrain/
  harness/        # experiment schema, registry, sweep, and pipeline
  trainers/       # ClassifierTrainer and UltralyticsDetectionTrainer
  models/         # GenericClassifier, two-stage Detector, MLflow registration
  cli/            # Typer commands
```

## Train and sweep

From `wildtrain/` after `uv sync`:

```bash
wildtrain run -c configs/classification/classification_train.yaml
wildtrain run -c configs/detection/yolo_configs/yolo.yaml
wildtrain sweep -c configs/classification/classification_sweep.yaml
wildtrain sweep -c configs/detection/detection_sweep.yaml
```

Set `DATA_ROOT` for the classification dataset root and `DETECTION_DATA_CFG` for the YOLO data yaml. Those files are not required to exist until training starts.

`wildtrain evaluate` and `wildtrain register` still use their own YAML. The `eval` block on an experiment is recorded and is not the eval command. `tracking.dvc.enabled` is recorded; the shared DVC tracker is not wired yet.

## Classification

```yaml
task: classification
name: base_dinov2
seed: 41
tracking:
  mlflow:
    tracking_uri: http://127.0.0.1:5000
    experiment_name: wildtrain_classification
    run_name: base_dinov2
    log_model: true
  dvc:
    enabled: false
data:
  format: roi                  # roi is the product format; crop is optional
  root: ${oc.env:DATA_ROOT,.}
  batch_size: 64
  num_workers: 4
  input_size: 384
  label_policy:                # omitted means identity, not collapse
    mode: collapse             # identity | collapse
    positive_name: wildlife
    background_name: background
    discard: [rocks, vegetation, other, "termite mound"]
  stats:
    mean: [0.554, 0.469, 0.348]
    std: [0.203, 0.173, 0.144]
  transforms:
    train:
      - {name: Resize, params: {size: ${data.input_size}}}
      - {name: RandomHorizontalFlip, params: {p: 0.5}}
    val:
      - {name: Resize, params: {size: ${data.input_size}}}
model:
  backend: timm
  backbone: timm/vit_base_patch14_reg4_dinov2.lvd142m
  pretrained: true
  dropout: 0.2
  freeze_backbone: true
  weights: null
train:
  epochs: 20
  lr: 0.001
  weight_decay: 0.001
  lrf: 0.01
  accelerator: auto
  precision: "32"
checkpoint:
  monitor: val_f1score
  mode: max
  patience: 10
  dirpath: checkpoints/classification
  filename: best
eval:
  split: val
  metrics: [accuracy, precision, recall, f1, auroc]
```

`label_policy.positive_name` is the collapsed class. `background_name` is the background class. `mode: identity` keeps species. The shipped file sets `collapse`.

## Detection

```yaml
task: detection
name: yolo12s-d0
project: wildai-localization
seed: 41
tracking:
  mlflow:
    tracking_uri: http://127.0.0.1:5000
    experiment_name: wildai-localization
    run_name: yolo12s-d0
  dvc:
    enabled: false
data:
  format: yolo
  data_cfg: ${oc.env:DETECTION_DATA_CFG,configs/detection/yolo_configs/data/demo.yaml}
  discard: [rocks, vegetation, other, "termite mound"]
model:
  backend: ultralytics          # ultralytics | custom_yolo
  weights: yolo12s.pt
  architecture: null            # set for a yaml architecture such as yolov8n.yaml
train:
  epochs: 25
  batch_size: 32
  imgsz: 800
  single_cls: true              # default; false keeps dataset class ids
  lr0: 0.0001
  lrf: 0.1
  optimizer: AdamW
  device: cuda
eval:
  split: val
  metrics: [metrics/mAP50(B), metrics/mAP50-95(B)]
  conf: 0.1
  iou: 0.6
```

`model.backend: custom_yolo` selects the custom fork. The default path does not import it. `train.batch_size` is the public name of the Ultralytics batch.

## Sweep

Add a `sweep` block to the same document. `parameters` maps a dotted field to a list of choices.

```yaml
sweep:
  n_trials: 20
  direction: maximize
  seed: 42
  objective: metrics/mAP50(B)
  parameters:
    train.lr0: [0.0001, 0.001]
    train.batch_size: [8, 16, 32]
    model.weights: [yolo12n.pt, yolo12s.pt]
```

A detection `objective` must match an Ultralytics metrics key exactly. It is not translated. Common keys from this Ultralytics build:

- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `fitness`

`fitness` uses the fitness value stored on the training result. A classification objective of `f1_score` uses the validation F1 stored on that result. A name such as `map_50` is looked up as written and will not score unless that exact key is present.

## Evaluate and register

```bash
wildtrain evaluate detector -c configs/detection/yolo_configs/yolo_eval.yaml
wildtrain evaluate classifier -c configs/classification/classification_eval.yaml
wildtrain register detector configs/registration/detector_registration_example.yaml
```

Registration still writes `batch`, `imgsz`, `task`, `num_classes`, `model_type`, and `cls_export_format`. Census reads `batch`, then tile size from `tilesize` or `imgsz`.
