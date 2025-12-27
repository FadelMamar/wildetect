# Dataset Preparation Tutorial

Learn how to prepare datasets for training using WilData.

## Overview

This tutorial covers importing, transforming, and exporting datasets for wildlife detection training.

## Prerequisites

- WilData installed
- Annotated images (COCO, YOLO, or Label Studio format)

## Step 1: Import Dataset

### Option A: Using Config File

Create `config.yaml`:

```yaml
source_path: "D:/annotations/dataset.json"
source_format: "coco"
dataset_name: "wildlife_train"
root: "D:/data"
split_name: "train"

transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    min_visibility: 0.7
```

Run import:

```bash
cd wildata
wildata import-dataset --config config.yaml
```

### Option B: Direct CLI

```bash
wildata import-dataset annotations.json \
    --format coco \
    --name wildlife_train \
    --enable-tiling \
    --tile-size 800
```

## Step 2: Apply Transformations

### Tiling for Large Images

```yaml
transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    min_visibility: 0.7
```

### Bbox Clipping

```yaml
transformations:
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5
    skip_invalid: false
```

## Step 3: Create ROI Dataset

For classification training:

```bash
cd wildata
scripts\create-roi-dataset.bat

# Or with CLI
wildata create-roi-dataset --config configs/roi-create-config.yaml
```

## Step 4: Visualize

```bash
# Launch FiftyOne
wildata visualize-dataset --dataset wildlife_train --split train
```

## Step 5: Export for Training

```bash
# Export to YOLO format
wildata dataset export wildlife_train --format yolo --output exports/yolo

# Export to COCO
wildata dataset export wildlife_train --format coco --output exports/coco
```

## Complete Example

```python
from wildata import DataPipeline

# Initialize
pipeline = DataPipeline("data")

# Import with transformations
result = pipeline.import_dataset(
    source_path="annotations.json",
    source_format="coco",
    dataset_name="wildlife_train",
    transformations={
        "enable_tiling": True,
        "tiling": {
            "tile_size": 800,
            "stride": 640
        }
    }
)

# Export for training
pipeline.export_dataset("wildlife_train", "yolo")
```

---

**Next Steps:**
- [Model Training Tutorial](model-training.md)
- [WilData Scripts Reference](../scripts/wildata/index.md)

