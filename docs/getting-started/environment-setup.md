# Environment Setup

This guide covers setting up your environment for working with the WildDetect monorepo, including configuration files, environment variables, and external services.

## Directory Structure

Create the following directory structure for your project:

```
your-project/
├── wildetect/          # Main package (cloned repo)
├── data/              # Data storage root
│   ├── raw/           # Original data
│   ├── processed/     # Processed datasets
│   └── exports/       # Exported datasets
├── models/            # Trained models
│   ├── detectors/
│   └── classifiers/
├── results/           # Detection results
│   ├── detections/
│   ├── census/
│   └── visualizations/
└── mlruns/            # MLflow experiment tracking
```

## Environment Variables

### Create .env File

Create a `.env` file in the root directory of each package:

#### WildDetect `.env`

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=wilddetect
MODEL_REGISTRY_PATH=models/

# Data Paths
DATA_ROOT=D:/data/
RESULTS_ROOT=D:/results/

# Label Studio (Optional)
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key_here
LABEL_STUDIO_PROJECT_ID=1

# FiftyOne (Optional)
FIFTYONE_DATABASE_DIR=D:/fiftyone/
FIFTYONE_DEFAULT_DATASET_DIR=D:/data/fiftyone/

# Inference Server
INFERENCE_SERVER_HOST=0.0.0.0
INFERENCE_SERVER_PORT=4141

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/wildetect.log
```

#### WilData `.env`

```bash
# API Configuration
WILDATA_API_HOST=0.0.0.0
WILDATA_API_PORT=8441
WILDATA_API_DEBUG=false

# Data Storage
DATA_ROOT=D:/data/
DVC_REMOTE_URL=s3://my-bucket/datasets  # or local path

# DVC Configuration
DVC_CACHE_DIR=D:/.dvc/cache/

# Label Studio Integration
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key_here

# Processing
MAX_WORKERS=4
BATCH_SIZE=32
```

#### WildTrain `.env`

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=wildtrain

# Training Paths
DATA_ROOT=D:/data/
MODEL_OUTPUT_DIR=D:/models/
CHECKPOINT_DIR=D:/checkpoints/

# Hyperparameter Tuning
OPTUNA_STORAGE=sqlite:///optuna.db
N_TRIALS=50

# Distributed Training (Optional)
MASTER_ADDR=localhost
MASTER_PORT=12355
WORLD_SIZE=1
RANK=0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1  # Multiple GPUs
```

### Loading Environment Variables

The packages automatically load `.env` files when using scripts:

```bash
# Scripts automatically load .env
scripts\run_detection.bat

# Or manually in Python
from dotenv import load_dotenv
load_dotenv()
```

## External Services Setup

### MLflow Tracking Server

MLflow is used for experiment tracking and model registry.

#### 1. Start MLflow Server

```bash
# Launch using script
scripts\launch_mlflow.bat

# Or manually
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

#### 2. Access MLflow UI

Open browser to: `http://localhost:5000`

#### 3. Configure in Code

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")
```

### Label Studio (Optional)

For data annotation and labeling.

#### 1. Install Label Studio

```bash
uv pip install label-studio
```

#### 2. Start Server

```bash
# Launch using script
scripts\launch_labelstudio.bat

# Or manually
label-studio start --port 8080
```

#### 3. Create Project

1. Navigate to `http://localhost:8080`
2. Create new project
3. Upload images
4. Configure labeling interface (use provided XML configs)

#### 4. Get API Key

1. Go to Account & Settings
2. Copy your API token
3. Add to `.env` file

### FiftyOne (Dataset Visualization)

Interactive dataset viewer and analyzer.

#### 1. Install FiftyOne

```bash
uv pip install fiftyone
```

#### 2. Launch Viewer

```bash
# Launch using script
scripts\launch_fiftyone.bat

# Or using CLI
wildetect fiftyone --action launch --dataset my_dataset
```

#### 3. Configure Database

```bash
# Set database directory
fiftyone config database_dir D:/fiftyone/db

# Set default dataset directory
fiftyone config default_dataset_dir D:/data/fiftyone
```

### DVC (Data Version Control)

For versioning large datasets.

#### 1. Initialize DVC

```bash
cd wildata
scripts\dvc-setup.bat

# Or manually
dvc init
dvc remote add -d myremote s3://my-bucket/datasets
```

#### 2. Configure Remote Storage

=== "Local Storage"
    ```bash
    dvc remote add -d local D:/dvc-storage
    ```

=== "AWS S3"
    ```bash
    dvc remote add -d s3remote s3://my-bucket/datasets
    
    # Set credentials
    dvc remote modify s3remote access_key_id YOUR_KEY
    dvc remote modify s3remote secret_access_key YOUR_SECRET
    ```

=== "Google Cloud Storage"
    ```bash
    dvc remote add -d gcs gs://my-bucket/datasets
    
    # Set credentials
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
    ```

=== "Azure Blob"
    ```bash
    dvc remote add -d azure azure://container/path
    
    # Set credentials
    export AZURE_STORAGE_CONNECTION_STRING=your_connection_string
    ```

#### 3. Track Data

```bash
# Add data to DVC
dvc add data/raw/

# Commit changes
git add data/raw.dvc .gitignore
git commit -m "Add raw data"

# Push to remote
dvc push
```

## Configuration Files

### WildDetect Configurations

Location: `config/`

#### detection.yaml

Main detection configuration. Edit based on your needs:

```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

processing:
  batch_size: 32
  tile_size: 800
  overlap_ratio: 0.2
```

See [Detection Config Reference](../configs/wildetect/detection.md) for all options.

#### census.yaml

Census campaign configuration:

```yaml
campaign:
  name: "Summer_2024"
  target_species: ["elephant", "giraffe", "zebra"]

flight_specs:
  flight_height: 120.0
  gsd: 2.38
```

See [Census Config Reference](../configs/wildetect/census.md).

### WilData Configurations

Location: `wildata/configs/`

#### import-config-example.yaml

Dataset import configuration:

```yaml
source_path: "annotations.json"
source_format: "coco"
dataset_name: "my_dataset"

transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
```

See [Import Config Reference](../configs/wildata/import-config-example.md).

### WildTrain Configurations

Location: `wildtrain/configs/`

#### Training Config

For model training:

```yaml
# configs/classification/classification_train.yaml
model:
  architecture: "resnet50"
  num_classes: 10

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

See [Training Configs](../configs/wildtrain/classification.md).

## GPU Configuration

### CUDA Setup

#### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

#### Set GPU Device

In `.env`:
```bash
CUDA_VISIBLE_DEVICES=0  # Use first GPU
CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs
```

In config files:
```yaml
device: "cuda"  # Use default GPU
device: "cuda:0"  # Specific GPU
device: "cpu"  # Force CPU
```

#### Memory Management

For large models or images:

```bash
# In .env
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or in Python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### CPU-Only Setup

If you don't have a GPU:

1. Install CPU-only PyTorch (see [Installation](installation.md))
2. Set `device: "cpu"` in all config files
3. Reduce batch sizes for memory efficiency

## Testing Your Setup

### Run System Info

```bash
wildetect info
```

This will display:
- Python version
- Package versions
- CUDA availability
- GPU information
- Memory available

### Test Detection

```bash
# Test with a single image
wildetect detect test_image.jpg --model model.pt --output test_results/
```

### Test Data Import

```bash
# Test data import
wildata import-dataset test_annotations.json --format coco --name test_dataset
```

### Test Training

```bash
# Test training setup
cd wildtrain
wildtrain train classifier -c configs/classification/classification_train.yaml --dry-run
```

## IDE Setup

### VSCode Configuration

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests",
    "-v"
  ],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

### Ruff Configuration

The project uses ruff for linting. Configuration is in `pyproject.toml`:

```bash
# Run ruff on all files
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/
```

## Directory Permissions (Windows)

Ensure you have write permissions for:
- Data directories
- Model directories
- Results directories
- Log directories

Run PowerShell as Administrator if needed:

```powershell
# Grant full control to current user
icacls "D:\data" /grant %USERNAME%:F /t
```

## Troubleshooting

### Common Issues

??? question "MLflow server won't start"
    
    Check if port 5000 is already in use:
    ```bash
    netstat -ano | findstr :5000
    ```
    
    Use a different port:
    ```bash
    mlflow server --port 5001
    ```

??? question "DVC push fails"
    
    Verify remote credentials:
    ```bash
    dvc remote list
    dvc remote modify --local myremote access_key_id YOUR_KEY
    ```

??? question "Out of memory errors"
    
    Reduce batch size and tile size:
    ```yaml
    processing:
      batch_size: 16  # Reduced
      tile_size: 640  # Reduced
    ```

??? question "Import errors"
    
    Verify virtual environment is activated:
    ```bash
    which python  # Should point to .venv
    ```

## Next Steps

Now that your environment is set up:

1. ✅ Test your setup with the commands above
2. 📚 Follow the [Quick Start Guide](quick-start.md)
3. 🎯 Try an [End-to-End Detection](../tutorials/end-to-end-detection.md) tutorial

---

**Environment ready?** Head to the [Quick Start Guide](quick-start.md) to run your first detection!

