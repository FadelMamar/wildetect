# WildTrain Model Registration Configuration

Reference for classifier and detector registration YAML configuration files.

## Overview

Registration configs control how trained models are registered to the [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html), including weight export format, MLflow tracking URI, and model metadata.

**Usage:**

```bash
# Register classifier
wildtrain register classifier -c configs/registration/classifier_registration_example.yaml

# Register detector
wildtrain register detector configs/registration/detector_registration_example.yaml
```

---

## Classifier Registration

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `weights` | `str` | — | Path to the classifier checkpoint file (`.ckpt`) |
| `processing.name` | `str` | `classifier` | Model name in MLflow registry |
| `processing.batch_size` | `int` | `8` | Batch size for inference (used in export) |
| `processing.mlflow_tracking_uri` | `str` | `http://localhost:5000` | MLflow tracking server URI |
| `processing.export_format` | `str` | `torchscript` | Export format: `torchscript` or `onnx` |
| `processing.dynamic` | `bool` | `true` | Use dynamic axes for ONNX export |

### Example

```yaml
weights: checkpoints/classification/best.ckpt

processing:
  name: "classifier"
  batch_size: 8
  mlflow_tracking_uri: "http://localhost:5000"
  export_format: "torchscript"
  dynamic: true
```

---

## Detector Registration

The detector registration config supports a two-model architecture (localizer + classifier).

### Configuration Fields

#### Classifier Section

| Field | Type | Description |
|-------|------|-------------|
| `classifier.weights` | `str` | Path to classifier checkpoint |
| `classifier.processing.batch_size` | `int` | Batch size for export |
| `classifier.processing.export_format` | `str` | Export format: `torchscript` or `onnx` |

#### Localizer Section

| Field | Type | Description |
|-------|------|-------------|
| `localizer.yolo.weights` | `str` | Path to YOLO weights file (`.pt`) |
| `localizer.yolo.imgsz` | `int` | Input image size |
| `localizer.yolo.device` | `str` | Device: `cuda` or `cpu` |
| `localizer.yolo.conf_thres` | `float` | Confidence threshold |
| `localizer.yolo.iou_thres` | `float` | NMS IoU threshold |
| `localizer.yolo.max_det` | `int` | Maximum detections per image |
| `localizer.yolo.overlap_metric` | `str` | Overlap metric: `IOU` |
| `localizer.yolo.task` | `str` | YOLO task: `detect` or `obb` |
| `localizer.processing.export_format` | `str` | Export format: `pt` |
| `localizer.processing.batch_size` | `int` | Batch size for export |
| `localizer.processing.dynamic` | `bool` | Dynamic axes (ONNX) |

#### Processing Section

| Field | Type | Description |
|-------|------|-------------|
| `processing.name` | `str` | Model name in MLflow registry |
| `processing.mlflow_tracking_uri` | `str` | MLflow tracking server URI |

### Example

```yaml
classifier:
  weights: "checkpoints/classification/best.ckpt"
  processing:
    batch_size: 8
    export_format: "torchscript"

localizer:
  yolo:
    weights: "runs/detect/train/weights/best.pt"
    imgsz: 800
    device: "cuda"
    conf_thres: 0.1
    iou_thres: 0.3
    max_det: 300
    overlap_metric: "IOU"
    task: "detect"
  processing:
    export_format: "pt"
    batch_size: 32
    dynamic: false

processing:
  name: "detector"
  mlflow_tracking_uri: "http://localhost:5000"
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
