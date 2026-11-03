# WildTrain Model Registration Configuration

Reference for classifier and detector registration YAML configuration files.

## Overview

Registration configs control how trained models are registered to the [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html), including weight export format, MLflow tracking URI, and model metadata.

**Usage:**

```bash
wildtrain register -c configs/registration/classifier_registration_example.yaml
wildtrain register -c configs/registration/detector_registration_example.yaml
```

The old `processing` / `localizer.yolo` files are rejected. Registered metadata stays `batch`, `imgsz`, `task`, `num_classes`, `model_type`, and `cls_export_format`.

---

## Classifier registration

| Field | Description |
|-------|-------------|
| `task` | `classification` |
| `name` | MLflow registered model name |
| `weights` | Classifier checkpoint. `${oc.env:CLASSIFIER_WEIGHTS,...}` in the shipped file |
| `batch` | Batch size written to metadata and used for export |
| `mlflow_tracking_uri` | Tracking URI. `${oc.env:MLFLOW_TRACKING_URI,...}` |
| `export_format` | `torchscript` or `onnx` |

```yaml
task: classification
name: classifier
weights: ${oc.env:CLASSIFIER_WEIGHTS,checkpoints/classification/best.ckpt}
batch: 8
mlflow_tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,http://127.0.0.1:5000}
export_format: torchscript
```

---

## Detector registration

A localizer-only file registers the single-class detector. Set `classifier_weights` for the two-stage localizer plus ROI classifier. `export_format: pt` keeps the YOLO `.pt` file. The classifier is exported as torchscript unless `classifier_export_format` is set.

| Field | Description |
|-------|-------------|
| `task` | `detection` |
| `name` | MLflow registered model name |
| `batch` | Batch size written as `batch` |
| `localizer.weights` | YOLO weights. `${oc.env:LOCALIZER_WEIGHTS,...}` |
| `localizer.imgsz` | Tile size written as `imgsz` |
| `classifier_weights` | Optional ROI classifier checkpoint |

```yaml
task: detection
name: detector
batch: 8
mlflow_tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,http://127.0.0.1:5000}
export_format: pt
localizer:
  weights: ${oc.env:LOCALIZER_WEIGHTS,models/best.pt}
  imgsz: 800
  device: cpu
classifier_weights: ${oc.env:CLASSIFIER_WEIGHTS,checkpoints/classification/best.ckpt}
```

---

## Inference Server Config

The inference server config (`configs/inference.yaml`) controls the LitServe model serving:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `port` | `int` | `4141` | Server port |
| `workers_per_device` | `int` | `1` | Number of workers per GPU |
| `mlflow_registry_name` | `str` | — | MLflow model registry name |
| `mlflow_alias` | `str` | — | Model version alias |
| `mlflow_local_dir` | `str` | — | Local directory for model download |
| `mlflow_tracking_uri` | `str` | — | MLflow tracking server URI |

### Example

```yaml
port: 4141
workers_per_device: 1
mlflow_registry_name: detector
mlflow_alias: production
mlflow_local_dir: models-registry
mlflow_tracking_uri: http://localhost:5000
```

**Usage:**

```bash
wildtrain run-server -c configs/inference.yaml
```

---

**See also:**

- [WildTrain CLI Reference](../../api-reference/wildtrain-cli.md) — `register` and `run-server` commands
- [Detector Registration Config (WildDetect)](../wildetect/detector_registration.md)
