# Detector Registration Configuration

> **Location**: `config/detector_registration.yaml`

**Purpose**: Configuration file for registering trained detector and classifier models to the MLflow model registry. This file defines model paths, metadata, tags, and aliases for model versioning.

## Configuration Structure

### Complete Parameter Reference

```yaml
classifier:
  weights: D:\PhD\workspace\wildetect\models-registry\detector\4\artifacts\best-v6.ckpt
  processing:
    batch_size: 8  # used for onnx export
    export_format: "torchscript"  # torchscript or onnx

localizer:
  yolo:
    weights: D:\PhD\workspace\wildetect\models-registry\detector\4\artifacts\best.pt
    imgsz: 800
    device: "cpu"
    conf_thres: 0.1
    iou_thres: 0.3
    max_det: 300
    overlap_metric: "IOU"
    task: "detect"
  mmdet: null
  processing:
    export_format: "pt"
    batch_size: 128
    dynamic: false

processing:
  name: "detector"
  mlflow_tracking_uri: "http://localhost:5000"
```

### Parameter Descriptions

#### `classifier`
Configuration for registering a classifier model.

- **`weights`** (string): Path to the classifier model weights file (`.ckpt` format)
- **`processing.batch_size`** (int): Batch size used during model export/conversion
- **`processing.export_format`** (string): Target export format. Options: `torchscript`, `onnx`

#### `localizer`
Configuration for registering a detector/localizer model. Supports YOLO and MMDetection frameworks.

- **`yolo.weights`** (string): Path to YOLO model weights file (`.pt` format)
- **`yolo.imgsz`** (int): Input image size the model expects (pixels)
- **`yolo.device`** (string): Device used for model export/testing. Options: `cpu`, `cuda`
- **`yolo.conf_thres`** (float): Default confidence threshold (0.0-1.0)
- **`yolo.iou_thres`** (float): Default IoU threshold for NMS (0.0-1.0)
- **`yolo.max_det`** (int): Maximum detections per image
- **`yolo.overlap_metric`** (string): Overlap metric used. Options: `IOU`, `DIOU`, `CIOU`
- **`yolo.task`** (string): Model task type. Options: `detect`, `segment`, `classify`
- **`mmdet`** (null or dict): MMDetection configuration (set to `null` if not using MMDetection)
- **`processing.export_format`** (string): Export format for the model. Options: `pt`, `torchscript`, `onnx`
- **`processing.batch_size`** (int): Batch size for processing
- **`processing.dynamic`** (bool): Whether to use dynamic batch sizing

#### `processing`
General processing and MLflow configuration.

- **`name`** (string): Model name in MLflow registry
- **`mlflow_tracking_uri`** (string): MLflow tracking server URI

---

## Example Configurations

### Register YOLO Detector

```yaml
localizer:
  yolo:
    weights: models/best.pt
    imgsz: 800
    device: "cuda"
    conf_thres: 0.25
    iou_thres: 0.45
    max_det: 300
    overlap_metric: "IOU"
    task: "detect"
  mmdet: null
  processing:
    export_format: "pt"
    batch_size: 32
    dynamic: false

processing:
  name: "wildlife_detector_v2"
  mlflow_tracking_uri: "http://localhost:5000"
```

### Register Classifier

```yaml
classifier:
  weights: models/classifier_best.ckpt
  processing:
    batch_size: 8
    export_format: "torchscript"

processing:
  name: "wildlife_classifier"
  mlflow_tracking_uri: "http://localhost:5000"
```

### Register Both Models

```yaml
classifier:
  weights: models/classifier.ckpt
  processing:
    batch_size: 8
    export_format: "torchscript"

localizer:
  yolo:
    weights: models/detector.pt
    imgsz: 800
    device: "cuda"
    conf_thres: 0.25
    iou_thres: 0.45
    max_det: 300
    overlap_metric: "IOU"
    task: "detect"
  processing:
    export_format: "pt"
    batch_size: 32

processing:
  name: "detector"
  mlflow_tracking_uri: "http://localhost:5000"
```

---

## Best Practices

1. **Model Paths**: Use absolute paths or paths relative to the project root
2. **Model Naming**: Use descriptive names in `processing.name` that indicate model version or purpose
3. **Export Formats**: 
   - Use `torchscript` for PyTorch deployment
   - Use `onnx` for cross-platform deployment
   - Use `pt` for PyTorch-only environments
4. **MLflow URI**: Ensure MLflow server is running before registration
5. **Model Testing**: Test model loading before registration to catch errors early
6. **Version Control**: Tag models with meaningful aliases (e.g., "production", "latest", "v1.0")

---

## Troubleshooting

### Model File Not Found

**Issue**: Cannot find model weights file

**Solutions**:
1. Verify the path in `weights` field is correct
2. Check file permissions
3. Use absolute paths if relative paths fail
4. Ensure model file exists before registration

### MLflow Connection Error

**Issue**: Cannot connect to MLflow tracking server

**Solutions**:
1. Start MLflow server: `scripts/launch_mlflow.bat`
2. Verify `mlflow_tracking_uri` is correct
3. Check network connectivity
4. Verify MLflow server is accessible from your machine

### Model Registration Fails

**Issue**: Registration process fails with error

**Solutions**:
1. Check model file format is correct
2. Verify model can be loaded independently
3. Check MLflow server logs for detailed error messages
4. Ensure sufficient disk space for model artifacts
5. Verify MLflow database is accessible

### Export Format Not Supported

**Issue**: Export format conversion fails

**Solutions**:
1. Verify model framework supports the export format
2. Check required dependencies are installed (ONNX, TorchScript)
3. Try different export format
4. Ensure model is in correct state (eval mode, etc.)

---

## Related Documentation

- [Configuration Overview](index.md)
- [Model Registration Script](../../scripts/wildetect/register_model.md)
- [MLflow Setup](../../scripts/wildetect/launch_mlflow.md)
- [Main Config](config.md)

