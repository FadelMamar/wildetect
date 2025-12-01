# WilData Scripts Reference

This page documents all batch scripts available in the WilData package for data management operations.

## Overview

All scripts are located in `wildata/scripts/` directory.

## Quick Reference

| Script | Purpose | Config File |
|--------|---------|-------------|
| [import-dataset-example.bat](#import-dataset-example) | Import single dataset | `configs/import-config-example.yaml` |
| [bulk-import-dataset.bat](#bulk-import-dataset) | Bulk import datasets | `configs/bulk-import-*.yaml` |
| [create-roi-dataset.bat](#create-roi-dataset) | Create ROI dataset | `configs/roi-create-config.yaml` |
| [bulk-roi-create-config.bat](#bulk-roi-create) | Bulk create ROI datasets | `configs/bulk-roi-create-config.yaml` |
| [update-gps-example.bat](#update-gps-example) | Update GPS from CSV | `configs/gps-update-config-example.yaml` |
| [visualize_data.bat](#visualize-data) | Visualize dataset | None |
| [dvc-setup.bat](#dvc-setup) | Setup DVC | None |
| [launch_api.bat](#launch-api) | Launch REST API | `.env` |
| [running_tests.bat](#running-tests) | Run tests | None |

---

## Data Import Scripts

### import-dataset-example.bat

**Purpose**: Import a single dataset from COCO, YOLO, or Label Studio format.

**Location**: `wildata/scripts/import-dataset-example.bat`

**Command**:
```batch
uv run wildata import-dataset --config configs\import-config-example.yaml
```

**Configuration**: `wildata/configs/import-config-example.yaml`

**Key Parameters**:
```yaml
source_path: "path/to/annotations.json"
source_format: "coco"  # coco, yolo, ls
dataset_name: "my_dataset"

root: "data"
split_name: "train"  # train, val, test

transformations:
  enable_bbox_clipping: true
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    
roi_config:
  roi_box_size: 384
  random_roi_count: 2
```

**Example Usage**:
```bash
cd wildata

# Edit config file first
notepad configs\import-config-example.yaml

# Run import
scripts\import-dataset-example.bat
```

**Output**:
- Master format dataset in `data/datasets/`
- Processed images (tiled if enabled)
- ROI dataset (if configured)

---

### bulk-import-dataset.bat

**Purpose**: Import multiple datasets in batch mode.

**Location**: `wildata/scripts/bulk-import-dataset.bat`

**Command**:
```batch
uv run wildata bulk-import-datasets --config configs\bulk-import-config-example.yaml -n 2
```

**Configuration**: `wildata/configs/bulk-import-train.yaml` or `bulk-import-val.yaml`

**Parameters**:
- `-n 2`: Number of parallel workers (uses threading on Windows)
- `--config`: Path to bulk import config

**Example Config**:
```yaml
# configs/bulk-import-train.yaml
source_paths:
  - "D:/annotations/dataset1.json"
  - "D:/annotations/dataset2.json"
  - "D:/annotations/dataset3.json"

source_format: "coco"
root: "D:/data"
split_name: "train"

# Shared transformation settings
transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
```

**Example Usage**:
```bash
cd wildata
scripts\bulk-import-dataset.bat
```

**Features**:
- Parallel processing (thread-based)
- Progress tracking
- Error handling per dataset
- Summary report

---

## ROI Dataset Scripts

### create-roi-dataset.bat

**Purpose**: Create Region of Interest (ROI) classification dataset from detection annotations.

**Location**: `wildata/scripts/create-roi-dataset.bat`

**Command**:
```batch
uv run wildata create-roi-dataset --config configs\roi-create-config.yaml
```

**Configuration**: `wildata/configs/roi-create-config.yaml`

**Key Parameters**:
```yaml
source_path: "annotations.json"
source_format: "coco"
dataset_name: "roi_dataset"

roi_config:
  roi_box_size: 128        # Size of extracted ROI
  min_roi_size: 32         # Min object size to extract
  random_roi_count: 10     # Background samples per image
  background_class: "background"
  save_format: "jpg"
  quality: 95
```

**Use Cases**:
- Hard sample mining
- Error analysis
- Training classification models
- Creating balanced datasets

**Example Usage**:
```bash
cd wildata
scripts\create-roi-dataset.bat
```

**Output**:
- ROI image crops
- Classification labels
- Class mapping JSON
- Statistics file

---

### bulk-roi-create.bat

**Purpose**: Create multiple ROI datasets in batch.

**Location**: Script not shown, but referenced in configs

**Configuration**: `wildata/configs/bulk-roi-create-config.yaml`

**Example Config**:
```yaml
source_paths:
  - "dataset1.json"
  - "dataset2.json"

source_format: "coco"
split_name: "val"

roi_config:
  roi_box_size: 128
  random_roi_count: 5
```

---

## GPS Management Scripts

### update-gps-example.bat

**Purpose**: Update image EXIF GPS data from CSV file.

**Location**: `wildata/scripts/update-gps-example.bat`

**Command**:
```batch
uv run wildata update-gps-from-csv --config configs\gps-update-config-example.yaml
```

**Configuration**: `wildata/configs/gps-update-config-example.yaml`

**Key Parameters**:
```yaml
image_folder: "path/to/images"
csv_path: "gps_coordinates.csv"
output_dir: "output/images"

skip_rows: 0
filename_col: "filename"
lat_col: "latitude"
lon_col: "longitude"
alt_col: "altitude"
```

**CSV Format**:
```csv
filename,latitude,longitude,altitude
image1.jpg,40.7128,-74.0060,10.5
image2.jpg,40.7589,-73.9851,15.2
```

**Example Usage**:
```bash
cd wildata

# Prepare CSV with GPS data
# Edit config
notepad configs\gps-update-config-example.yaml

# Run update
scripts\update-gps-example.bat
```

**Output**:
- Images with updated EXIF GPS
- Summary report
- Error log (if any)

---

## Visualization Scripts

### visualize_data.bat

**Purpose**: Launch FiftyOne visualization for datasets.

**Location**: `wildata/scripts/visualize_data.bat`

**Command**:
```batch
uv run wildata visualize-dataset --dataset my_dataset --split train
```

**Example Usage**:
```bash
cd wildata

# Visualize training set
uv run wildata visualize-dataset --dataset my_dataset --split train

# Or use script
scripts\visualize_data.bat
```

**Features**:
- Interactive dataset viewer
- Annotation visualization
- Filtering and search
- Statistics display

---

## DVC Scripts

### dvc-setup.bat

**Purpose**: Initialize and configure DVC for data versioning.

**Location**: `wildata/scripts/dvc-setup.bat`

**Command**:
```batch
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote <storage_path>
```

**Storage Options**:

=== "Local Storage"
    ```batch
    dvc remote add -d local D:\dvc-storage
    ```

=== "AWS S3"
    ```batch
    dvc remote add -d s3remote s3://bucket/path
    dvc remote modify s3remote access_key_id YOUR_KEY
    dvc remote modify s3remote secret_access_key YOUR_SECRET
    ```

=== "Google Cloud"
    ```batch
    dvc remote add -d gcs gs://bucket/path
    set GOOGLE_APPLICATION_CREDENTIALS=path\to\credentials.json
    ```

**Example Usage**:
```bash
cd wildata
scripts\dvc-setup.bat

# Track data
dvc add data\datasets\my_dataset

# Commit DVC file
git add data\datasets\my_dataset.dvc
git commit -m "Add dataset"

# Push to remote
dvc push
```

**DVC Workflow**:
```bash
# On another machine
git pull
dvc pull  # Downloads data
```

---

## API Scripts

### launch_api.bat

**Purpose**: Launch WilData REST API server.

**Location**: `wildata/scripts/launch_api.bat`

**Command**:
```batch
uv run python -m wildata.api.main
```

**Default Port**: 8441

**Example Usage**:
```bash
cd wildata
scripts\launch_api.bat
```

**Access**:
- API: `http://localhost:8441`
- Docs: `http://localhost:8441/docs`
- Redoc: `http://localhost:8441/redoc`

**API Endpoints**:

#### Import Dataset
```bash
POST /api/v1/datasets/import
Content-Type: application/json

{
  "source_path": "/path/to/data.json",
  "source_format": "coco",
  "dataset_name": "my_dataset",
  "root": "data"
}
```

#### List Datasets
```bash
GET /api/v1/datasets?root=data
```

#### Create ROI Dataset
```bash
POST /api/v1/roi/create
Content-Type: application/json

{
  "source_path": "/path/to/data.json",
  "source_format": "coco",
  "dataset_name": "roi_dataset",
  "roi_config": {
    "roi_box_size": 128,
    "random_roi_count": 10
  }
}
```

#### Job Status
```bash
GET /api/v1/jobs/{job_id}
```

**Environment Variables**:
```bash
# In .env
WILDATA_API_HOST=0.0.0.0
WILDATA_API_PORT=8441
WILDATA_API_DEBUG=false
```

**Full API Documentation**: See [WilData API Reference](../../api-reference/wildata-api.md)

---

## Testing Scripts

### running_tests.bat

**Purpose**: Run WilData test suite.

**Location**: `wildata/scripts/running_tests.bat`

**Command**:
```batch
uv run pytest tests/ -v
```

**Example Usage**:
```bash
cd wildata
scripts\running_tests.bat
```

**Test Categories**:
- Format adapter tests
- Transformation tests
- Validation tests
- API tests
- Integration tests

**Run Specific Tests**:
```bash
# Test imports
uv run pytest tests/test_coco_import.py -v

# Test transformations
uv run pytest tests/test_transformations.py -v

# Test API
uv run pytest tests/api/ -v

# With coverage
uv run pytest --cov=wildata tests/
```

---

## Common Workflows

### Dataset Preparation Workflow

```bash
# 1. Import dataset
cd wildata
scripts\import-dataset-example.bat

# 2. Visualize
scripts\visualize_data.bat

# 3. Export for training
uv run wildata dataset export my_dataset --format yolo
```

### ROI Extraction Workflow

```bash
# 1. Import detection dataset
scripts\import-dataset-example.bat

# 2. Create ROI dataset
scripts\create-roi-dataset.bat

# 3. Visualize ROI dataset
uv run wildata visualize-dataset --dataset roi_dataset
```

### GPS Management Workflow

```bash
# 1. Extract GPS from images
# (using WildDetect extract_gps.bat)

# 2. Update GPS if needed
cd wildata
scripts\update-gps-example.bat

# 3. Verify GPS data
# Check EXIF data in images
```

### DVC Workflow

```bash
# Setup (once)
cd wildata
scripts\dvc-setup.bat

# After each dataset import
dvc add data\datasets\new_dataset
git add data\datasets\new_dataset.dvc
git commit -m "Add new dataset"
dvc push

# On other machines
git pull
dvc pull
```

---

## Configuration Examples

### Complete Import Config

```yaml
# configs/import-config-example.yaml
source_path: "D:/annotations/dataset.json"
source_format: "coco"
dataset_name: "wildlife_train"

root: "D:/data"
split_name: "train"
processing_mode: "batch"

# Label Studio integration
ls_xml_config: "configs/label_studio_config.xml"
ls_parse_config: false

# ROI extraction
disable_roi: false
roi_config:
  random_roi_count: 2
  roi_box_size: 384
  min_roi_size: 32
  background_class: "background"
  sample_background: true

# Transformations
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5
    skip_invalid: false
  
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    min_visibility: 0.7
    max_negative_tiles_in_negative_image: 2
    dark_threshold: 0.7
  
  enable_augmentation: false
  augmentation:
    rotation_range: [-45, 45]
    probability: 1.0
    num_transforms: 2
```

---

## Troubleshooting

### Import Fails

**Issue**: Dataset import fails with validation errors

**Solutions**:
1. Check source file format is correct
2. Verify all image paths are valid
3. Check bbox coordinates are within image bounds
4. Use `--verbose` flag for detailed errors

### DVC Push Fails

**Issue**: Can't push data to remote

**Solutions**:
1. Verify remote credentials
2. Check network connection
3. Verify remote storage path exists
4. Use `dvc remote list` to check configuration

### API Won't Start

**Issue**: API server fails to start

**Solutions**:
1. Check port 8441 is not in use
2. Verify `.env` file configuration
3. Check all dependencies installed
4. Look at error logs

### Out of Memory

**Issue**: Import fails with memory error

**Solutions**:
1. Use `processing_mode: "streaming"`
2. Reduce number of parallel workers
3. Process datasets one at a time
4. Disable transformations temporarily

---

## Next Steps

- [WilData Configuration Reference](../../configs/wildata/index.md)
- [WilData CLI Reference](../../api-reference/wildata-cli.md)
- [Dataset Preparation Tutorial](../../tutorials/dataset-preparation.md)
- [WilData API Documentation](../../api-reference/wildata-api.md)

