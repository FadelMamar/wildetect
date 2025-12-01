# WildDetect Scripts Reference

This page documents all batch scripts available in the WildDetect package. These scripts provide convenient ways to run common operations on Windows.

## Overview

All scripts are located in the `scripts/` directory and should be run from the project root.

## Quick Reference

| Script | Purpose | Config File |
|--------|---------|-------------|
| [run_detection.bat](#run-detection) | Run wildlife detection | `config/detection.yaml` |
| [run_census.bat](#run-census) | Run census campaign | `config/census.yaml` |
| [launch_ui.bat](#launch-ui) | Launch Streamlit UI | None |
| [launch_fiftyone.bat](#launch-fiftyone) | Launch FiftyOne viewer | None |
| [launch_labelstudio.bat](#launch-labelstudio) | Launch Label Studio | `.env` |
| [launch_mlflow.bat](#launch-mlflow) | Launch MLflow UI | None |
| [launch_inference_server.bat](#launch-inference-server) | Launch inference API | None |
| [register_model.bat](#register-model) | Register model to MLflow | `config/detector_registration.yaml` |
| [extract_gps.bat](#extract-gps) | Extract GPS from images | `config/extract-gps.yaml` |
| [profile_census.bat](#profile-census) | Profile census performance | `config/census.yaml` |
| [run_integration_tests.bat](#run-integration-tests) | Run integration tests | None |
| [load_env.bat](#load-env) | Load environment variables | `.env` |

---

## Detection Scripts

### run_detection.bat

**Purpose**: Run wildlife detection on images using a trained model.

**Location**: `scripts/run_detection.bat`

**Command**:
```batch
uv run --env-file .env wildetect detection detect -c config/detection.yaml
```

**Configuration**: `config/detection.yaml`

**Key Parameters**:
- `model.mlflow_model_name`: Model name in MLflow registry
- `model.mlflow_model_alias`: Model version/alias (e.g., "production")
- `model.device`: Device to use ("cuda", "cpu", "auto")
- `image_paths`: List of image paths to process
- `image_dir`: Directory containing images
- `processing.batch_size`: Batch size for inference
- `processing.tile_size`: Tile size for large images
- `processing.pipeline_type`: Pipeline strategy ("raster", "multithreaded", "simple")
- `output.directory`: Output directory for results

**Example Usage**:
```bash
# Edit config/detection.yaml first
cd wildetect
scripts\run_detection.bat
```

**Example Config**:
```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

image_dir: "D:/images/survey_2024/"

processing:
  batch_size: 32
  tile_size: 800
  pipeline_type: "raster"

output:
  directory: "results/detections"
```

**Output**:
- Detection results: JSON and CSV files
- Visualizations (if enabled)
- FiftyOne dataset (if configured)

---

### run_census.bat

**Purpose**: Run a complete wildlife census campaign with statistics and reports.

**Location**: `scripts/run_census.bat`

**Command**:
```batch
uv run --env-file .env --no-sync wildetect detection census -c config/census.yaml
```

**Configuration**: `config/census.yaml`

**Key Parameters**:
- `campaign.name`: Census campaign name
- `campaign.target_species`: List of target species
- `model`: Model configuration (same as detection)
- `flight_specs.flight_height`: Flight altitude in meters
- `flight_specs.gsd`: Ground Sample Distance (cm/px)
- `analysis`: Analysis options (density, hotspots, maps)
- `output.generate_pdf_report`: Generate PDF report

**Example Usage**:
```bash
cd wildetect
scripts\run_census.bat
```

**Example Config**:
```yaml
campaign:
  name: "Summer_2024_Survey"
  target_species: ["elephant", "giraffe", "zebra"]

model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"

flight_specs:
  flight_height: 120.0
  gsd: 2.38

output:
  directory: "census_results"
  generate_pdf_report: true
```

**Output**:
- Census statistics (counts, density)
- Geographic analysis
- Visualizations and maps
- PDF report (if enabled)

---

### profile_census.bat

**Purpose**: Profile census performance with detailed timing and memory analysis.

**Location**: `scripts/profile_census.bat`

**Command**:
```batch
uv run wildetect detection census -c config/census.yaml --profile --gpu-profile --line-profile
```

**Flags**:
- `--profile`: Enable general profiling
- `--gpu-profile`: Profile GPU memory usage
- `--line-profile`: Line-by-line profiling

**Example Usage**:
```bash
cd wildetect
scripts\profile_census.bat
```

**Output**:
- Profiling reports
- Performance metrics
- Memory usage statistics
- Bottleneck identification

---

## Visualization Scripts

### extract_gps.bat

**Purpose**: Extract GPS coordinates from image EXIF data and create a summary.

**Location**: `scripts/extract_gps.bat`

**Command**:
```batch
uv run wildetect visualization extract-gps-coordinates -c config/extract-gps.yaml
```

**Configuration**: `config/extract-gps.yaml`

**Key Parameters**:
- `image_directory`: Directory containing images with EXIF data
- `output_file`: Output CSV file for GPS coordinates
- `recursive`: Search subdirectories

**Example Usage**:
```bash
cd wildetect
scripts\extract_gps.bat
```

**Example Config**:
```yaml
image_directory: "D:/images/survey/"
output_file: "gps_coordinates.csv"
recursive: true
```

**Output**:
- CSV file with GPS coordinates
- Summary statistics
- Coverage map data

---

## UI and Service Scripts

### launch_ui.bat

**Purpose**: Launch the WildDetect Streamlit web interface.

**Location**: `scripts/launch_ui.bat`

**Command**:
```batch
uv run wildetect services ui
```

**Features**:
- Interactive detection interface
- Configuration editor
- Results visualization
- Real-time processing

**Example Usage**:
```bash
cd wildetect
scripts\launch_ui.bat
```

**Access**: Opens browser at `http://localhost:8501`

---

### launch_fiftyone.bat

**Purpose**: Launch FiftyOne app for interactive dataset visualization.

**Location**: `scripts/launch_fiftyone.bat`

**Command**:
```batch
uv run --no-sync --env-file .env fiftyone app launch
```

**Prerequisites**:
- FiftyOne installed (`pip install fiftyone`)
- Dataset loaded in FiftyOne

**Example Usage**:
```bash
cd wildetect
scripts\launch_fiftyone.bat
```

**Access**: Opens browser at `http://localhost:5151`

**Features**:
- Interactive dataset viewer
- Detection visualization
- Filtering and querying
- Export capabilities

---

### launch_labelstudio.bat

**Purpose**: Launch Label Studio for data annotation.

**Location**: `scripts/launch_labelstudio.bat`

**Command**:
```batch
# Activates separate venv and starts Label Studio
label-studio start -p 8080
```

**Prerequisites**:
- Label Studio venv configured
- Label Studio installed in venv

**Example Usage**:
```bash
cd wildetect/scripts
launch_labelstudio.bat
```

**Access**: Opens browser at `http://localhost:8080`

**Configuration**:
- Set `LABEL_STUDIO_API_KEY` in `.env`
- Configure project settings in Label Studio UI

---

### launch_mlflow.bat

**Purpose**: Launch MLflow tracking server UI.

**Location**: `scripts/launch_mlflow.bat`

**Command**:
```batch
uv run mlflow server --backend-store-uri runs/mlflow --host 0.0.0.0 --port 5000
```

**Example Usage**:
```bash
cd wildetect
scripts\launch_mlflow.bat
```

**Access**: Opens browser at `http://localhost:5000`

**Features**:
- View experiments and runs
- Compare models
- Model registry management
- Metrics and artifacts

**Environment Variables**:
```bash
# In .env
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

### launch_inference_server.bat

**Purpose**: Launch FastAPI inference server for remote detection.

**Location**: `scripts/launch_inference_server.bat`

**Command**:
```batch
uv run wildetect services inference-server --port 4141 --workers 2
```

**Example Usage**:
```bash
cd wildetect
scripts\launch_inference_server.bat
```

**Access**:
- API: `http://localhost:4141`
- Docs: `http://localhost:4141/docs`

**API Endpoints**:
```bash
# Health check
GET /health

# Run detection
POST /predict
Content-Type: multipart/form-data
{
  "file": <image_file>,
  "confidence": 0.5
}

# Batch detection
POST /predict/batch
```

**Example Request**:
```python
import requests

# Single image
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:4141/predict",
        files={"file": f},
        data={"confidence": 0.5}
    )

detections = response.json()
```

---

## Model Management Scripts

### register_model.bat

**Purpose**: Register a trained model to MLflow model registry.

**Location**: `scripts/register_model.bat`

**Command**:
```batch
uv run wildtrain register detector config/detector_registration.yaml
```

**Configuration**: `config/detector_registration.yaml`

**Key Parameters**:
- `model_path`: Path to model weights
- `model_name`: Name for model registry
- `model_type`: Model type ("detector" or "classifier")
- `description`: Model description
- `tags`: Metadata tags

**Example Usage**:
```bash
cd wildetect
scripts\register_model.bat
```

**Example Config**:
```yaml
model_path: "models/best.pt"
model_name: "wildlife_detector"
model_type: "detector"
description: "YOLO model trained on aerial wildlife images"
tags:
  framework: "yolo"
  dataset: "wildlife_v1"
  training_date: "2024-01-15"
```

**Output**:
- Model registered in MLflow
- Model version assigned
- Artifacts logged

---

## Testing Scripts

### run_integration_tests.bat

**Purpose**: Run integration tests for detection pipeline.

**Location**: `scripts/run_integration_tests.bat`

**Command**:
```batch
# Detection pipeline tests
uv run pytest tests/test_detection_pipeline.py::TestDetectionPipeline::test_detection_pipeline_with_real_images -v

# Data loading tests
uv run pytest tests/test_data_loading.py -v
```

**Example Usage**:
```bash
cd wildetect
scripts\run_integration_tests.bat
```

**Tests Covered**:
- Detection pipeline with real images
- Data loading and preprocessing
- Model loading and inference
- Result formatting and export

**Requirements**:
- Test data in `tests/data/`
- Test model available
- pytest installed

---

## Utility Scripts

### load_env.bat

**Purpose**: Load environment variables from `.env` file.

**Location**: `scripts/load_env.bat`

**Usage**: Called automatically by other scripts

**Environment Variables**:
```bash
# Example .env file
MLFLOW_TRACKING_URI=http://localhost:5000
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key
DATA_ROOT=D:/data/
CUDA_VISIBLE_DEVICES=0
```

---

## Common Workflows

### Detection Workflow

```bash
# 1. Start MLflow
scripts\launch_mlflow.bat

# 2. Run detection
scripts\run_detection.bat

# 3. View results
scripts\launch_fiftyone.bat
```

### Census Workflow

```bash
# 1. Configure census
# Edit config/census.yaml

# 2. Run census
scripts\run_census.bat

# 3. View results
# Open census_results/report.pdf
```

### Model Training and Registration

```bash
# 1. Train model (in wildtrain)
cd wildtrain
scripts\train_classifier.bat

# 2. Register model
cd ..
scripts\register_model.bat

# 3. View in MLflow
scripts\launch_mlflow.bat
```

---

## Troubleshooting

### Script Won't Run

**Issue**: Script exits immediately or shows error

**Solutions**:
1. Check Python environment is activated
2. Run `uv sync` to install dependencies
3. Verify `.env` file exists and is configured
4. Check paths in configuration files

### Model Loading Error

**Issue**: Can't load model from MLflow

**Solutions**:
1. Ensure MLflow server is running (`launch_mlflow.bat`)
2. Check model name and alias in config
3. Verify `MLFLOW_TRACKING_URI` in `.env`
4. Use `mlflow models list` to see available models

### GPU Not Detected

**Issue**: Detection runs on CPU despite having GPU

**Solutions**:
1. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Set `device: "cuda"` in config file
3. Check `CUDA_VISIBLE_DEVICES` environment variable
4. Reinstall PyTorch with CUDA support

### Out of Memory

**Issue**: CUDA out of memory or system RAM exhausted

**Solutions**:
1. Reduce `batch_size` in config
2. Reduce `tile_size` for raster detection
3. Close other applications
4. Use `pipeline_type: "simple"` for lower memory usage

---

## Next Steps

- [Configuration Reference](../../configs/wildetect/index.md)
- [CLI Reference](../../api-reference/wildetect-cli.md)
- [Tutorials](../../tutorials/end-to-end-detection.md)

