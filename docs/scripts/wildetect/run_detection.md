# Detection Script

> **Location**: `scripts/run_detection.bat`

**Purpose**: Run wildlife detection on images using a trained model. This script executes the main detection pipeline configured in `config/detection.yaml`.

## Usage

```batch
scripts\run_detection.bat
```

The script automatically:
1. Changes to the project root directory
2. Loads environment variables from `.env` file
3. Runs the detection command with the configuration file

## Command Executed

```batch
uv run --env-file .env wildetect detection detect -c config/detection.yaml
```

## Configuration

**Config File**: `config/detection.yaml`

See [Detection Configuration](../../configs/wildetect/detection.md) for complete parameter reference.

## Prerequisites

1. **Environment Setup**:
   - `.env` file exists in project root
   - Environment variables configured (MLflow URI, etc.)

2. **Model Availability**:
   - Model registered in MLflow (or model path specified)
   - MLflow server running (if using MLflow models)

3. **Input Images**:
   - Images specified in `config/detection.yaml`
   - Images accessible at specified paths

4. **Dependencies**:
   - All Python dependencies installed via `uv sync`

## Example Workflow

### 1. Configure Detection

Edit `config/detection.yaml`:

```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

image_dir: "D:/survey_images/2024/"

processing:
  batch_size: 32
  tile_size: 800
  pipeline_type: "raster"

output:
  directory: "results/detections"
```

### 2. Start MLflow (if using MLflow models)

```batch
scripts\launch_mlflow.bat
```

### 3. Run Detection

```batch
scripts\run_detection.bat
```

### 4. View Results

Results will be saved to the directory specified in `output.directory`. You can also view in FiftyOne:

```batch
scripts\launch_fiftyone.bat
```

## Output

The detection script generates:

- **Detection Results**: JSON and CSV files with detection coordinates, classes, and confidence scores
- **Visualizations**: Images with bounding boxes overlaid (if enabled)
- **FiftyOne Dataset**: Dataset uploaded to FiftyOne (if `output.dataset_name` is set)
- **Logs**: Processing logs saved to log file or console

## Common Use Cases

### Single Image Detection

```yaml
# In config/detection.yaml
image_paths:
  - "D:/images/single_image.tif"
```

### Directory of Images

```yaml
# In config/detection.yaml
image_dir: "D:/survey_images/"
```

### Large Raster (GeoTIFF)

```yaml
# In config/detection.yaml
image_paths:
  - "D:/orthomosaics/large_ortho.tif"

processing:
  pipeline_type: "raster"

flight_specs:
  gsd: 2.38  # Required for raster
```

### With GPS Update

```yaml
# In config/detection.yaml
exif_gps_update:
  image_folder: "D:/images/"
  csv_path: "D:/gps_data.csv"
  filename_col: "filename"
  lat_col: "latitude"
  lon_col: "longitude"
```

## Troubleshooting

### Script Exits Immediately

**Issue**: Script closes without running

**Solutions**:
1. Check Python environment is set up: `uv sync`
2. Verify `.env` file exists
3. Run from project root directory
4. Check script file has correct line endings (Windows CRLF)

### Model Not Found

**Issue**: Cannot load model from MLflow

**Solutions**:
1. Ensure MLflow server is running: `scripts\launch_mlflow.bat`
2. Verify model name and alias in config match MLflow registry
3. Check `MLFLOW_TRACKING_URI` in `.env` file
4. List available models: `mlflow models list`

### Images Not Found

**Issue**: Cannot find input images

**Solutions**:
1. Verify image paths in config are correct
2. Use absolute paths instead of relative paths
3. Check file permissions
4. Ensure images exist at specified locations

### CUDA Out of Memory

**Issue**: GPU memory errors

**Solutions**:
1. Reduce `batch_size` in config (try 16, 8, or 4)
2. Reduce `tile_size` (try 640 or 512)
3. Set `device: "cpu"` to use CPU
4. Close other GPU applications

### Detection Takes Too Long

**Issue**: Processing is very slow

**Solutions**:
1. Increase `batch_size` if memory allows
2. Use `pipeline_type: "mt"` for multi-threading
3. Use GPU (`device: "cuda"`) instead of CPU
4. Reduce `tile_size` for faster processing
5. Process fewer images at once

## Related Documentation

- [Detection Configuration](../../configs/wildetect/detection.md)
- [Census Script](run_census.md)
- [End-to-End Detection Tutorial](../../tutorials/end-to-end-detection.md)
- [CLI Reference](../../api-reference/wildetect-cli.md)

