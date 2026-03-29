# WildData CLI Usage Guide

The WildData CLI provides a command-line interface for managing datasets in the WildData pipeline. It uses Pydantic for comprehensive validation and supports both command-line arguments and YAML configuration files.

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -e .
```

## Available Commands

### 1. Import Dataset

Import a dataset from various formats (COCO, YOLO) into the WildData pipeline.

#### Basic Usage

```bash
# Import a COCO dataset
wildata import-dataset path/to/dataset.json --format coco --name my_dataset

# Import a YOLO dataset
wildata import-dataset path/to/yolo/directory --format yolo --name my_dataset
```

#### Advanced Options

```bash
wildata import-dataset path/to/dataset.json \
    --format coco \
    --name my_dataset \
    --root data \
    --split train \
    --mode batch \
    --track-dvc \
    --bbox-tolerance 5 \
    --verbose
```

#### Using Configuration File

Create a YAML configuration file (see `scripts/import-config-example.yaml` for an example):

```yaml
source_path: "path/to/your/dataset.json"
source_format: "coco"
dataset_name: "my_dataset"
root: "data"
split_name: "train"
processing_mode: "batch"
track_with_dvc: false
bbox_tolerance: 5
```

Then use it:

```bash
wildata import-dataset --config import-config.yaml
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_path` | str | ✅ | - | Path to source dataset |
| `--format, -f` | str | ✅ | - | Source format (coco/yolo) |
| `--name, -n` | str | ✅ | - | Dataset name |
| `--root, -r` | str | ❌ | "data" | Root directory for data storage |
| `--split, -s` | str | ❌ | "train" | Split name (train/val/test) |
| `--mode, -m` | str | ❌ | "batch" | Processing mode (streaming/batch) |
| `--track-dvc` | bool | ❌ | False | Track dataset with DVC |
| `--bbox-tolerance` | int | ❌ | 5 | Bbox validation tolerance |
| `--dotenv` | str | ❌ | None | Path to .env file |
| `--ls-config` | str | ❌ | None | Label Studio XML config path |
| `--parse-ls-config` | bool | ❌ | False | Parse Label Studio config |
| **Transformation Pipeline Options** | | | | |
| `--enable-bbox-clipping` | bool | ❌ | True | Enable bbox clipping |
| `--bbox-clipping-tolerance` | int | ❌ | 5 | Bbox clipping tolerance |
| `--skip-invalid-bbox` | bool | ❌ | False | Skip invalid bboxes |
| `--enable-augmentation` | bool | ❌ | False | Enable data augmentation |
| `--aug-prob` | float | ❌ | 1.0 | Augmentation probability |
| `--num-augs` | int | ❌ | 2 | Number of augmentations per image |
| `--enable-tiling` | bool | ❌ | False | Enable image tiling |
| `--tile-size` | int | ❌ | 512 | Tile size |
| `--tile-stride` | int | ❌ | 416 | Tile stride |
| `--min-visibility` | float | ❌ | 0.1 | Minimum visibility ratio |
| `--config, -c` | str | ❌ | None | Path to YAML config file |
| `--verbose, -v` | bool | ❌ | False | Verbose output |

### 2. List Datasets

List all available datasets in the pipeline.

```bash
wildata list-datasets
wildata list-datasets --root data --verbose
```

### 3. Export Dataset

Export a dataset to a specific format.

```bash
wildata export-dataset my_dataset --format coco --output path/to/export
wildata export-dataset my_dataset --format yolo --output path/to/export
```

### 4. Version

Show version information.

```bash
wildata version
```

## Configuration File Format

The CLI supports YAML configuration files for complex import operations. Here's the complete schema:

```yaml
# Required parameters
source_path: "path/to/your/dataset.json"
source_format: "coco"  # or "yolo"
dataset_name: "my_dataset"

# Pipeline configuration
root: "data"
split_name: "train"  # train, val, or test
enable_dvc: true

# Processing options
processing_mode: "batch"  # streaming or batch
track_with_dvc: false
bbox_tolerance: 5

# Label Studio options (optional)
dotenv_path: null  # "path/to/.env"
ls_xml_config: null  # "path/to/label_studio_config.xml"
ls_parse_config: false

# ROI configuration (optional)
roi_config:
  random_roi_count: 10
  roi_box_size: 128
  min_roi_size: 32
  dark_threshold: 0.5
  background_class: "background"
  save_format: "jpg"
  quality: 95

# Transformation pipeline configuration (optional)
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5
    skip_invalid: false
  
  enable_augmentation: false
  augmentation:
    rotation_range: [-45, 45]
    probability: 1.0
    brightness_range: [-0.2, 0.4]
    scale: [1.0, 2.0]
    translate: [-0.1, 0.2]
    shear: [-5, 5]
    contrast_range: [-0.2, 0.4]
    noise_std: [0.01, 0.1]
    seed: 41
    num_transforms: 2
  
  enable_tiling: false
  tiling:
    tile_size: 512
    stride: 416
    min_visibility: 0.1
    max_negative_tiles_in_negative_image: 3
    negative_positive_ratio: 1.0
    dark_threshold: 0.5

## Validation

The CLI uses Pydantic for comprehensive validation:

- **Source format**: Must be either "coco" or "yolo"
- **Split name**: Must be one of "train", "val", or "test"
- **Processing mode**: Must be either "streaming" or "batch"
- **File paths**: Must exist on the filesystem
- **ROI configuration**: Validates ranges and formats

## Error Handling

The CLI provides clear error messages for validation failures:

```bash
❌ Configuration validation error:
   source_format: source_format must be either "coco" or "yolo"
   source_path: Source path does not exist: /nonexistent/path
```

## Examples

### Import COCO Dataset with ROI Configuration

```bash
wildata import-dataset annotations.json \
    --format coco \
    --name satellite_images \
    --root satellite_data \
    --split train \
    --verbose
```

### Import YOLO Dataset with DVC Tracking

```bash
wildata import-dataset yolo_dataset \
    --format yolo \
    --name object_detection \
    --track-dvc \
    --mode streaming \
    --verbose
```

### Import with Data Augmentation

```bash
wildata import-dataset annotations.json \
    --format coco \
    --name augmented_dataset \
    --enable-augmentation \
    --aug-prob 0.8 \
    --num-augs 3 \
    --verbose
```

### Import with Image Tiling

```bash
wildata import-dataset large_images.json \
    --format coco \
    --name tiled_dataset \
    --enable-tiling \
    --tile-size 512 \
    --tile-stride 256 \
    --min-visibility 0.3 \
    --verbose
```

### Import with Custom Bbox Clipping

```bash
wildata import-dataset annotations.json \
    --format coco \
    --name clipped_dataset \
    --enable-bbox-clipping \
    --bbox-clipping-tolerance 10 \
    --skip-invalid-bbox \
    --verbose
```

### Import with Multiple Transformations

```bash
wildata import-dataset dataset.json \
    --format coco \
    --name processed_dataset \
    --enable-bbox-clipping \
    --enable-augmentation \
    --enable-tiling \
    --tile-size 256 \
    --aug-prob 0.7 \
    --num-augs 2 \
    --verbose
```

### Using Configuration File

```bash
# Create config file
cat > import_config.yaml << EOF
source_path: "data/annotations.json"
source_format: "coco"
dataset_name: "production_dataset"
root: "production_data"
split_name: "train"
processing_mode: "batch"
track_with_dvc: true
bbox_tolerance: 10
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5
    skip_invalid: false
  enable_augmentation: true
  augmentation:
    probability: 0.8
    num_transforms: 3
  enable_tiling: false
EOF

# Import using config
wildata import-dataset --config import_config.yaml --verbose
```

### 8. Update GPS from CSV

Update EXIF GPS data for images using coordinates from a CSV file.

#### Basic Usage

```bash
# Using command line arguments
wildata update-gps-from-csv \
    --image-folder "path/to/images" \
    --csv "path/to/gps_coordinates.csv" \
    --output "path/to/output"

# Using configuration file
wildata update-gps-from-csv --config configs/gps-update-config-example.yaml
```

#### Advanced Options

```bash
wildata update-gps-from-csv \
    --image-folder "path/to/images" \
    --csv "path/to/gps_coordinates.csv" \
    --output "path/to/output" \
    --skip-rows 1 \
    --filename-col "image_name" \
    --lat-col "lat" \
    --lon-col "lon" \
    --alt-col "elevation" \
    --verbose
```

#### Using Configuration File

Create a YAML configuration file (see `configs/gps-update-config-example.yaml` for an example):

```yaml
image_folder: "path/to/images"
csv_path: "path/to/gps_coordinates.csv"
output_dir: "path/to/output"
skip_rows: 0
filename_col: "filename"
lat_col: "latitude"
lon_col: "longitude"
alt_col: "altitude"
```

#### CSV Format

The CSV file should contain the following columns:
- `filename`: Name of the image file (e.g., "image1.jpg")
- `latitude`: Latitude coordinate in decimal degrees
- `longitude`: Longitude coordinate in decimal degrees
- `altitude`: Altitude in meters (optional)

Example CSV:
```csv
filename,latitude,longitude,altitude
image1.jpg,40.7128,-74.0060,10.5
image2.jpg,40.7589,-73.9851,15.2
image3.jpg,40.7505,-73.9934,8.1
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--config, -c` | str | ❌ | None | Path to YAML config file |
| `--image-folder, -i` | str | ✅ | - | Path to folder containing images |
| `--csv` | str | ✅ | - | Path to CSV file with GPS coordinates |
| `--output, -o` | str | ✅ | - | Output directory for updated images |
| `--skip-rows` | int | ❌ | 0 | Number of rows to skip in CSV |
| `--filename-col` | str | ❌ | "filename" | CSV column name for filenames |
| `--lat-col` | str | ❌ | "latitude" | CSV column name for latitude |
| `--lon-col` | str | ❌ | "longitude" | CSV column name for longitude |
| `--alt-col` | str | ❌ | "altitude" | CSV column name for altitude |
| `--verbose, -v` | bool | ❌ | False | Verbose output |

## Testing

Run the test script to verify CLI functionality:

```bash
python scripts/test-cli.py
```

This will test:
- Configuration validation
- YAML file loading/saving
- ROI configuration
- Error handling 