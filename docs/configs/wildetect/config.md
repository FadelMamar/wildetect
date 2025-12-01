# Main Configuration

> **Location**: `config/config.yaml`

**Purpose**: Main application configuration file that defines model paths, processing settings, and MLflow tracking configuration for WildDetect.

## Configuration Structure

### Complete Parameter Reference

```yaml
classifier:
  weights: config/classifier.ckpt
  processing:
    batch_size: 8
    export_format: torchscript
localizer:
  yolo:
    weights: D:\workspace\repos\wildetect\config\localizer.pt
    imgsz: 800
    device: cpu
    conf_thres: 0.1
    iou_thres: 0.3
    max_det: 300
    overlap_metric: IOU
    task: detect
  mmdet: null
  processing:
    export_format: pt
    batch_size: 32
    dynamic: false
processing:
  name: detector
  mlflow_tracking_uri: http://localhost:5000
```

### Parameter Descriptions

#### `classifier`
Configuration for the classifier model used in the detection pipeline.

- **`weights`** (string): Path to the classifier model weights file (`.ckpt` format)
- **`processing.batch_size`** (int): Batch size used for classifier inference or export
- **`processing.export_format`** (string): Export format for the classifier model. Options: `torchscript`, `onnx`

#### `localizer`
Configuration for the localizer (detector) model. Supports both YOLO and MMDetection frameworks.

- **`yolo.weights`** (string): Path to YOLO model weights file (`.pt` format)
- **`yolo.imgsz`** (int): Input image size for YOLO inference (pixels)
- **`yolo.device`** (string): Device to run inference on. Options: `cpu`, `cuda`, `auto`
- **`yolo.conf_thres`** (float): Confidence threshold for detections (0.0-1.0)
- **`yolo.iou_thres`** (float): IoU threshold for Non-Maximum Suppression (0.0-1.0)
- **`yolo.max_det`** (int): Maximum number of detections per image
- **`yolo.overlap_metric`** (string): Metric used for overlap calculation. Options: `IOU`, `DIOU`, `CIOU`
- **`yolo.task`** (string): Task type. Options: `detect`, `segment`, `classify`
- **`mmdet`** (null or dict): MMDetection model configuration (currently not used, set to `null`)
- **`processing.export_format`** (string): Export format for localizer. Options: `pt`, `torchscript`, `onnx`
- **`processing.batch_size`** (int): Batch size for localizer processing
- **`processing.dynamic`** (bool): Whether to use dynamic batch sizing

#### `processing`
General processing configuration.

- **`name`** (string): Name identifier for the processing pipeline
- **`mlflow_tracking_uri`** (string): MLflow tracking server URI for model registry and experiment tracking

---

## Example Configurations

### Basic Configuration

```yaml
classifier:
  weights: config/classifier.ckpt
  processing:
    batch_size: 8
    export_format: torchscript

localizer:
  yolo:
    weights: config/localizer.pt
    imgsz: 800
    device: cuda
    conf_thres: 0.25
    iou_thres: 0.45
    max_det: 300
    overlap_metric: IOU
    task: detect
  mmdet: null
  processing:
    export_format: pt
    batch_size: 32
    dynamic: false

processing:
  name: detector
  mlflow_tracking_uri: http://localhost:5000
```

### CPU-Only Configuration

```yaml
localizer:
  yolo:
    device: cpu
    conf_thres: 0.1
    iou_thres: 0.3
    max_det: 200
  processing:
    batch_size: 16  # Reduced for CPU
```

---

## Best Practices

1. **Model Paths**: Use relative paths when possible, or absolute paths that are consistent across environments
2. **Device Selection**: Set `device: "auto"` to let the system choose the best available device
3. **MLflow URI**: Use environment variables for MLflow tracking URI in production:
   ```yaml
   processing:
     mlflow_tracking_uri: ${MLFLOW_TRACKING_URI}
   ```
4. **Batch Sizes**: Adjust batch sizes based on available GPU memory. Start with smaller values and increase if memory allows
5. **Confidence Thresholds**: Lower `conf_thres` values (0.1-0.2) for initial detection, then filter in post-processing

---

## Troubleshooting

### Model Not Found

**Issue**: Error loading model weights

**Solutions**:
1. Verify the path to model weights is correct
2. Check that model files exist at the specified location
3. Ensure file permissions allow reading

### MLflow Connection Failed

**Issue**: Cannot connect to MLflow tracking server

**Solutions**:
1. Verify MLflow server is running: `scripts/launch_mlflow.bat`
2. Check `mlflow_tracking_uri` is correct
3. Verify network connectivity to MLflow server
4. Check firewall settings

### CUDA Out of Memory

**Issue**: GPU memory errors during inference

**Solutions**:
1. Reduce `batch_size` in processing settings
2. Set `device: "cpu"` to use CPU instead
3. Reduce `imgsz` for smaller input images
4. Lower `max_det` to limit number of detections

---

## Related Documentation

- [Configuration Overview](../index.md)
- [Detection Config](detection.md)
- [Census Config](census.md)
- [Scripts Reference](../../scripts/wildetect/index.md)

