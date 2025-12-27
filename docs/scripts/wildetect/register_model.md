# Model Registration Script

> **Location**: `scripts/register_model.bat`

**Purpose**: Register a trained detector or classifier model to the MLflow model registry. This makes the model available for use in detection and census pipelines.

## Usage

```batch
scripts\register_model.bat
```

The script automatically:
1. Changes to the project root directory
2. Runs the model registration command

## Command Executed

```batch
uv run wildtrain register detector config/detector_registration.yaml
```

Note: This script registers a **detector** model. For classifier registration, modify the script or use the CLI directly.

## Configuration

**Config File**: `config/detector_registration.yaml`

See [Detector Registration Configuration](../../configs/wildetect/detector_registration.md) for complete parameter reference.

## Prerequisites

1. **MLflow Server Running**: MLflow server must be running
   ```batch
   scripts\launch_mlflow.bat
   ```

2. **Model Files Available**: Model weights files must exist at paths specified in config

3. **Configuration**: `config/detector_registration.yaml` must be properly configured

4. **Environment**: `.env` file should have `MLFLOW_TRACKING_URI` set

## Example Workflow

### 1. Configure Registration

Edit `config/detector_registration.yaml`:

```yaml
localizer:
  yolo:
    weights: "models/best.pt"
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
  name: "wildlife_detector_v2"
  mlflow_tracking_uri: "http://localhost:5000"
```

### 2. Start MLflow

```batch
scripts\launch_mlflow.bat
```

### 3. Register Model

```batch
scripts\register_model.bat
```

### 4. Verify Registration

- Open MLflow UI: `http://localhost:5000`
- Navigate to Models section
- Verify model appears with correct name and version

## Output

The registration script:

- **Registers Model**: Adds model to MLflow model registry
- **Creates Version**: Assigns a new version number
- **Stores Artifacts**: Saves model files and metadata
- **Sets Metadata**: Stores tags, descriptions, and configuration

## Model Usage After Registration

Once registered, use the model in detection/census configs:

```yaml
# In config/detection.yaml or config/census.yaml
model:
  mlflow_model_name: "wildlife_detector_v2"  # Match name from registration
  mlflow_model_alias: "latest"  # or "production"
  device: "cuda"
```

## Registering Classifier

To register a classifier instead of detector, modify the script or use CLI directly:

```batch
uv run wildtrain register classifier config/detector_registration.yaml
```

Ensure config has `classifier` section configured.

## Troubleshooting

### MLflow Not Running

**Issue**: Cannot connect to MLflow server

**Solutions**:
1. Start MLflow: `scripts\launch_mlflow.bat`
2. Verify `MLFLOW_TRACKING_URI` in `.env` matches server URL
3. Check MLflow server is accessible: `curl http://localhost:5000`

### Model File Not Found

**Issue**: Cannot find model weights file

**Solutions**:
1. Verify `weights` path in config is correct
2. Use absolute paths instead of relative paths
3. Check file permissions
4. Ensure model file exists before registration

### Registration Fails

**Issue**: Registration process errors

**Solutions**:
1. Check model file format is correct (`.pt` for YOLO, `.ckpt` for classifier)
2. Verify model can be loaded independently
3. Check MLflow server logs for detailed errors
4. Ensure sufficient disk space for artifacts
5. Verify MLflow database is accessible

### Model Name Already Exists

**Issue**: Model name conflicts with existing model

**Solutions**:
1. Use different model name in `processing.name`
2. Or let MLflow create new version automatically
3. Check existing models in MLflow UI

### Export Format Not Supported

**Issue**: Model export/conversion fails

**Solutions**:
1. Verify model framework supports the export format
2. Check required dependencies are installed
3. Try different export format
4. Ensure model is in correct state (eval mode)

## Related Documentation

- [Detector Registration Configuration](../../configs/wildetect/detector_registration.md)
- [MLflow Launch Script](launch_mlflow.md)
- [Main Config](../../configs/wildetect/config.md)
- [WildTrain CLI](../../api-reference/wildtrain-cli.md)

