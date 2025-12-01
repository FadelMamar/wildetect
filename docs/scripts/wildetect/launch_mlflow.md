# MLflow Launch Script

> **Location**: `scripts/launch_mlflow.bat`

**Purpose**: Launch the MLflow tracking server UI for model registry management, experiment tracking, and model versioning.

## Usage

```batch
scripts\launch_mlflow.bat
```

The script automatically:
1. Changes to the project root directory
2. Launches MLflow tracking server

## Command Executed

```batch
uv run mlflow server --backend-store-uri runs/mlflow --host 0.0.0.0 --port 5000
```

## Access

Once launched, MLflow UI will be available at:
- **URL**: `http://localhost:5000`
- Opens automatically in your default web browser

## Features

MLflow provides:

- **Model Registry**: View and manage registered models
- **Experiment Tracking**: View training runs and metrics
- **Model Versioning**: Track model versions and aliases
- **Model Comparison**: Compare different model versions
- **Artifacts**: View model artifacts and files
- **Metadata**: View model tags, descriptions, and metadata

## Prerequisites

1. **MLflow Installed**: MLflow should be installed (included in dependencies)
2. **Port Availability**: Port 5000 should be available
3. **Storage**: `runs/mlflow` directory will be created for backend storage

## Example Workflow

### 1. Launch MLflow

```batch
scripts\launch_mlflow.bat
```

### 2. Access UI

Open browser to `http://localhost:5000` (usually opens automatically)

### 3. View Models and Experiments

- Browse registered models
- View experiment runs
- Compare model versions
- Download model artifacts

## Environment Variables

Set in `.env` file:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
```

This URI is used by detection and census scripts to load models.

## Integration with WildDetect

MLflow is used throughout WildDetect for:

- **Model Loading**: Detection and census scripts load models from MLflow
- **Model Registration**: Register trained models via `register_model.bat`
- **Version Management**: Use model aliases (e.g., "production", "latest")

### Example Model Configuration

```yaml
# In config/detection.yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"  # Loads production version
  device: "cuda"
```

## Stopping MLflow

Press `Ctrl+C` in the terminal window to stop the MLflow server.

## Troubleshooting

### Port Already in Use

**Issue**: Port 5000 is already occupied

**Solutions**:
1. Close other MLflow instances
2. Kill process using port 5000: `netstat -ano | findstr :5000`
3. Use different port: `--port 5001` (update `.env` accordingly)

### Cannot Connect to MLflow

**Issue**: Detection scripts cannot connect to MLflow

**Solutions**:
1. Verify MLflow server is running
2. Check `MLFLOW_TRACKING_URI` in `.env` matches server URL
3. Ensure server is accessible: `curl http://localhost:5000`
4. Check firewall settings

### Models Not Visible

**Issue**: Registered models don't appear in UI

**Solutions**:
1. Verify models were registered successfully
2. Check backend storage directory exists: `runs/mlflow`
3. Refresh browser page
4. Check model name and version in registry
5. Verify backend storage is accessible

### Backend Storage Issues

**Issue**: Errors related to backend storage

**Solutions**:
1. Verify `runs/mlflow` directory exists or can be created
2. Check file permissions for storage directory
3. Ensure sufficient disk space
4. Check for corrupted storage files

## Related Documentation

- [Model Registration Script](register_model.md)
- [Detection Script](run_detection.md)
- [Detector Registration Config](../../configs/wildetect/detector_registration.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

