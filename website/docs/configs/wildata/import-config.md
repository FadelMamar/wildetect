# WilData Import Configuration

Detailed reference for the `import-dataset` and `bulk-import-datasets` YAML configuration files.

## Overview

Import configuration controls how datasets are ingested into the WilData pipeline, including source format parsing, data transformations (tiling, augmentation, bbox clipping), ROI extraction, and DVC tracking.

**Usage:**

```bash
# Single dataset import
wildata import-dataset -c configs/import-config-example.yaml

# Bulk import (all files in a directory)
wildata bulk-import-datasets -c configs/bulk-import-train.yaml
```

---

## Single Import Config

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | `str` | Path to source annotation file (JSON for COCO/LS, YAML for YOLO) |
| `source_format` | `str` | Source format: `coco`, `yolo`, or `ls` (Label Studio) |
| `dataset_name` | `str` | Name for the imported dataset |

### Pipeline Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `root` | `str` | `data` | Root directory for data storage |
| `split_name` | `str` | `train` | Dataset split: `train`, `val`, or `test` |
| `processing_mode` | `str` | `batch` | Processing mode: `streaming` or `batch` |
| `track_with_dvc` | `bool` | `false` | Enable DVC version tracking |
| `enable_dvc` | `bool` | `false` | Enable DVC integration |
| `bbox_tolerance` | `int` | `5` | Tolerance for bounding box validation (pixels) |

### Label Studio Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dotenv_path` | `str` | `None` | Path to `.env` file with Label Studio credentials |
| `ls_xml_config` | `str` | `None` | Path to Label Studio XML labeling config |
| `ls_parse_config` | `bool` | `false` | Parse Label Studio config dynamically (requires LS running) |

### ROI Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disable_roi` | `bool` | `false` | Disable ROI extraction during import |
| `roi_config.random_roi_count` | `int` | `2` | Number of random background ROIs per image |
| `roi_config.roi_box_size` | `int` | `384` | ROI crop size in pixels |
| `roi_config.min_roi_size` | `int` | `32` | Minimum ROI size (smaller objects skipped) |
| `roi_config.dark_threshold` | `float` | `0.7` | Dark pixel threshold for filtering |
| `roi_config.background_class` | `str` | `background` | Name for the background class |
| `roi_config.save_format` | `str` | `jpg` | Output format for ROI crops |
| `roi_config.quality` | `int` | `95` | JPEG quality for saved crops |
| `roi_config.sample_background` | `bool` | `true` | Whether to sample background ROIs |

---

## Transformation Pipeline

The `transformations` section controls data processing applied during import.

### Bounding Box Clipping

Clips bounding boxes that extend outside image boundaries.

```yaml
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5         # Pixel tolerance for clipping
    skip_invalid: false  # Skip invalid bboxes instead of clipping
```

### Data Augmentation

Generates augmented copies of images with annotations.

```yaml
transformations:
  enable_augmentation: true
  augmentation:
    rotation_range: [-45, 45]      # Rotation range in degrees
    probability: 1.0               # Probability of applying augmentation
    brightness_range: [-0.2, 0.4]  # Brightness adjustment range
    scale: [1.0, 2.0]             # Scale range
    translate: [-0.1, 0.2]        # Translation range
    shear: [-5, 5]                # Shear range in degrees
    contrast_range: [-0.2, 0.4]   # Contrast adjustment range
    noise_std: [0.01, 0.1]        # Gaussian noise standard deviation range
    seed: 41                      # Random seed for reproducibility
    num_transforms: 2             # Number of augmentations per image
```

### Image Tiling

Splits large images into smaller tiles for training.

```yaml
transformations:
  enable_tiling: true
  tiling:
    tile_size: 800                          # Tile size in pixels
    stride: 640                             # Stride between tiles
    min_visibility: 0.7                     # Minimum bbox visibility ratio in tile
    max_negative_tiles_in_negative_image: 2 # Max empty tiles per negative image
    negative_positive_ratio: 5.0            # Ratio of negative to positive tiles
    dark_threshold: 0.7                     # Dark pixel threshold for filtering
```

---

## Bulk Import Config

For `bulk-import-datasets`, the config uses **`source_paths`** (list of directories) instead of `source_path`:

```yaml
source_paths:
  - D:/annotations/train_files/
  - D:/annotations/additional_files/

source_format: "ls"
root: D:/data
split_name: train
# ... same fields as single import
```

Each file in the directories is imported as a separate dataset. Dataset names are derived from filenames.

---

## Complete Example

```yaml
# Required
source_path: D:/annotations/project_export.json
source_format: "ls"
dataset_name: "wildlife_survey_2024"

# Pipeline
root: D:/data
split_name: "train"
processing_mode: "batch"
track_with_dvc: false
bbox_tolerance: 5

# Label Studio
dotenv_path: ".env"
ls_xml_config: "configs/label_studio_config.xml"
ls_parse_config: false

# ROI extraction
disable_roi: false
roi_config:
  random_roi_count: 2
  roi_box_size: 384
  min_roi_size: 32
  dark_threshold: 0.7
  background_class: "background"
  save_format: "jpg"
  quality: 95
  sample_background: true

# Transformations
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5
    skip_invalid: false

  enable_augmentation: false
  augmentation:
    rotation_range: [-45, 45]
    probability: 1.0
    num_transforms: 2

  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    min_visibility: 0.7
    max_negative_tiles_in_negative_image: 2
    negative_positive_ratio: 5.0
    dark_threshold: 0.7
```

---

**See also:**

- [WilData CLI Reference](../../api-reference/wildata-cli.md)
- [ROI Config Reference](roi-config.md)
- [Dataset Preparation Tutorial](../../tutorials/dataset-preparation.md)
