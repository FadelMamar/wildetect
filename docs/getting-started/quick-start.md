# Quick Start Guide

Get up and running with WildDetect in minutes! This guide shows you the fastest path to running your first wildlife detection.

## Prerequisites

Before starting, ensure you have:

- ✅ Installed all packages ([Installation Guide](installation.md))
- ✅ Activated your Python environment
- ✅ Some aerial images to process
- ✅ A pre-trained model (or use our example)

## Quick Start: Detection

### 1. Using the CLI

The simplest way to run detection:

```bash
wildetect detect /path/to/images --model model.pt --output results/
```

### 2. Using a Script (Windows)

Edit the configuration file, then run:

```bash
cd wildetect
scripts\run_detection.bat
```

### 3. Using Python

```python
from wildetect.core.detection import DetectionPipeline

# Initialize pipeline
pipeline = DetectionPipeline(
    model_path="model.pt",
    device="cuda"  # or "cpu"
)

# Run detection
results = pipeline.detect_batch("/path/to/images")

# Save results
pipeline.save_results(results, "results/detections.json")
```

## Quick Start: Census Campaign

Run a complete census analysis:

```bash
wildetect census campaign_2024 /path/to/images \
    --model model.pt \
    --output campaign_results/ \
    --species "elephant,giraffe,zebra"
```

This will:
- ✅ Detect all animals in your images
- ✅ Generate population statistics
- ✅ Create geographic visualizations
- ✅ Export reports in JSON and CSV

## Quick Start: Data Management

### Import a Dataset

```bash
# Import COCO format
wildata import-dataset annotations.json \
    --format coco \
    --name my_dataset

# Import YOLO format
wildata import-dataset data.yaml \
    --format yolo \
    --name my_dataset
```

### Visualize Data

```bash
# Launch FiftyOne viewer
wildetect fiftyone --action launch --dataset my_dataset

# Or use the script
scripts\launch_fiftyone.bat
```

## Quick Start: Model Training

### Train a Classifier

```bash
cd wildtrain
wildtrain train classifier -c configs/classification/classification_train.yaml
```

### Train a Detector (YOLO)

```bash
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml
```

## Using the Web UI

Each package has a Streamlit-based web interface:

### WildDetect UI

```bash
wildetect ui
# Or: scripts\launch_ui.bat
```

Features:
- Run detections interactively
- Configure detection parameters
- View results in real-time
- Export to various formats

### WilData UI

```bash
cd wildata
streamlit run src/wildata/ui.py
# Or: launch_ui.bat
```

Features:
- Import and export datasets
- Create ROI datasets
- Update GPS metadata
- Visualize data

### WildTrain UI

```bash
cd wildtrain
streamlit run src/wildtrain/ui.py
# Or: launch_ui.bat
```

Features:
- Configure training runs
- Monitor training progress
- Evaluate models
- Register models to MLflow

## Configuration Files

All operations can be configured via YAML files:

### Detection Config Example

Edit `config/detection.yaml`:

```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

processing:
  batch_size: 32
  tile_size: 800
  overlap_ratio: 0.2
  pipeline_type: "raster"

output:
  directory: "results"
  dataset_name: "my_detections"
```

### Dataset Import Config Example

Edit `wildata/configs/import-config-example.yaml`:

```yaml
source_path: "annotations.json"
source_format: "coco"
dataset_name: "my_dataset"
root: "data"
split_name: "train"

transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    min_visibility: 0.7
```

## Common Workflows

### Workflow 1: Detection on New Images

```bash
# 1. Run detection
wildetect detect images/ --model model.pt --output results/

# 2. View results
wildetect fiftyone --action launch

# 3. Export results
wildetect analyze results/detections.json --output analysis/
```

### Workflow 2: Prepare Training Data

```bash
# 1. Import annotations
wildata import-dataset annotations.json --format coco --name train_data

# 2. Apply transformations
wildata import-dataset annotations.json \
    --format coco \
    --name augmented_data \
    --enable-tiling \
    --enable-augmentation

# 3. Export for training
wildata export-dataset augmented_data --format yolo
```

### Workflow 3: Train and Deploy Model

```bash
# 1. Train model
cd wildtrain
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml

# 2. Evaluate model
scripts\eval_detector.bat

# 3. Register to MLflow
scripts\register_model.bat

# 4. Use for detection
cd ..
wildetect detect images/ --model-name my_detector --output results/
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Label Studio (optional)
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key

# Model Storage
MODEL_REGISTRY_PATH=models/

# Data Storage
DATA_ROOT=D:/data/

# GPU Settings
CUDA_VISIBLE_DEVICES=0
```

## Launching Services

### MLflow UI

Track experiments and manage models:

```bash
scripts\launch_mlflow.bat
# Access at http://localhost:5000
```

### Label Studio

Annotate images:

```bash
scripts\launch_labelstudio.bat
# Access at http://localhost:8080
```

### WilData API

REST API for data operations:

```bash
cd wildata
scripts\launch_api.bat
# Access at http://localhost:8441
# Docs at http://localhost:8441/docs
```

### Inference Server

Deploy model as API:

```bash
scripts\launch_inference_server.bat
# Access at http://localhost:4141
```

## Quick Reference

### Detection Commands

```bash
# Basic detection
wildetect detect images/ --model model.pt

# With tiling for large images
wildetect detect large_image.tif --model model.pt --tile-size 800

# Census with statistics
wildetect census campaign images/ --model model.pt

# Analyze results
wildetect analyze results.json
```

### Data Commands

```bash
# Import
wildata import-dataset source --format coco --name dataset

# List datasets
wildata dataset list

# Export
wildata dataset export dataset --format yolo

# Create ROI dataset
wildata create-roi annotations.json --format coco
```

### Training Commands

```bash
# Train classifier
wildtrain train classifier -c config.yaml

# Train detector
wildtrain train detector -c config.yaml

# Evaluate
wildtrain eval classifier -c config.yaml

# Register model
wildtrain register model_path --name my_model
```

## Getting Help

### Command Help

Every command has a `--help` flag:

```bash
wildetect --help
wildetect detect --help
wildata import-dataset --help
wildtrain train --help
```

### Package Information

```bash
# System info
wildetect info

# Check installation
python -c "import wildetect; print(wildetect.__version__)"
```

## Next Steps

Now that you've run your first commands:

1. 📖 **Deep Dive**: Follow the [End-to-End Detection Tutorial](../tutorials/end-to-end-detection.md)
2. 🏗️ **Understand Architecture**: Read the [Architecture Overview](../architecture/overview.md)
3. 🔧 **Configure**: Explore [Configuration Files](../configs/wildetect/index.md)
4. 📚 **Learn More**: Check out all [Tutorials](../tutorials/end-to-end-detection.md)

---

**Questions?** Check the [Troubleshooting Guide](../troubleshooting.md) or reach out via GitHub Issues.

