# Dataset Preparation Tutorial

Learn how to prepare datasets for training using WilData.

## Overview

This tutorial covers importing, transforming, and exporting datasets for wildlife detection training. WilData supports COCO, YOLO, and Label Studio formats, with built-in tiling, augmentation, and ROI extraction.

## Prerequisites

- WilData installed (`uv pip install -e .` in the `wildata/` directory)
- Annotated images in COCO, YOLO, or Label Studio format

## Step 1: Import Dataset

### Option A: Using a Config File (Recommended)

Create an import config YAML:

```yaml
# my_import_config.yaml
source_path: "D:/annotations/dataset.json"
source_format: "coco"        # coco, yolo, or ls (Label Studio)
dataset_name: "wildlife_train"
root: "D:/data"
split_name: "train"

# Enable tiling for large aerial images
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
```

Run the import:

```bash
wildata import-dataset -c my_import_config.yaml
```

### Option B: Direct CLI Arguments

```bash
wildata import-dataset annotations.json \
    --format coco \
    --name wildlife_train \
    --enable-tiling \
    --tile-size 800 \
    --tile-stride 640
```

See [Import Config Reference](../configs/wildata/import-config.md) for all available options.

## Step 2: Apply Transformations

Transformations are applied during import. Configure them in the `transformations` section of your config file.

### Tiling for Large Images

Essential for aerial imagery where images can be very large:

```yaml
transformations:
  enable_tiling: true
  tiling:
    tile_size: 800                          # Tile size in pixels
    stride: 640                             # Stride (set < tile_size for overlap)
    min_visibility: 0.7                     # Min bbox visibility ratio in tile
    max_negative_tiles_in_negative_image: 2 # Limit empty tiles
    negative_positive_ratio: 5.0            # Ratio of negative to positive tiles
```

### Bounding Box Clipping

Fixes annotations that extend outside image boundaries:

```yaml
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5         # Pixels of tolerance
    skip_invalid: false  # false = clip, true = discard
```

### Data Augmentation

Generate augmented copies of training images:

```yaml
transformations:
  enable_augmentation: true
  augmentation:
    rotation_range: [-45, 45]
    probability: 1.0
    brightness_range: [-0.2, 0.4]
    num_transforms: 2   # Number of augmented copies per image
```

## Step 3: Create ROI Dataset

For classification training, extract ROI (Region of Interest) crops from detection annotations:

```bash
wildata create-roi-dataset -c configs/roi-create-config.yaml
```

Example ROI config:

```yaml
source_path: configs/yolo_data.yaml
source_format: yolo
dataset_name: wildlife_roi
root: D:/data
split_name: val

roi_config:
  roi_box_size: 384           # Crop size in pixels
  min_roi_size: 32            # Minimum detection size
  random_roi_count: 1         # Background crops per image
  background_class: "background"
  save_format: "jpg"
  quality: 95
```

See [ROI Config Reference](../configs/wildata/roi-config.md) for all options.

## Step 4: Bulk Import (Multiple Datasets)

For importing many annotation files at once:

```bash
wildata bulk-import-datasets -c configs/bulk-import-train.yaml -n 4
```

Where `-n 4` specifies 4 parallel workers. The config uses `source_paths` (list of directories) instead of `source_path`.

## Step 5: Visualize

Launch FiftyOne to inspect your dataset:

```bash
# Visualize detection dataset
wildata visualize-detection my_dataset --root D:/data --split train

# Visualize classification (ROI) dataset
wildata visualize-classification my_roi_dataset --root D:/data/roi --split train
```

## Step 6: List Datasets

Check what datasets are available:

```bash
wildata list-datasets --root D:/data -v
```

---

**Next Steps:**

- [Model Training Tutorial](model-training.md) — Train models on your prepared data
- [Import Config Reference](../configs/wildata/import-config.md) — All import options
- [WilData CLI Reference](../api-reference/wildata-cli.md) — Full CLI documentation
