# WilData ROI Configuration

Reference for the `create-roi-dataset` and `bulk-create-roi-datasets` YAML configuration files.

## Overview

ROI (Region of Interest) configs control how classification datasets are generated from detection annotations. The process extracts image crops around each annotated bounding box and generates random background crops from unannotated regions.

**Usage:**

```bash
# Single ROI dataset
wildata create-roi-dataset -c configs/roi-create-config.yaml

# Bulk ROI creation
wildata bulk-create-roi-datasets -c configs/bulk-roi-create-config.yaml
```

---

## Single ROI Config

### Source Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | `str` | Path to source annotation file (YOLO data.yaml or COCO/LS JSON) |
| `source_format` | `str` | Source format: `coco`, `yolo`, or `ls` |
| `dataset_name` | `str` | Name for the ROI dataset |
| `root` | `str` | Root directory where data is stored |
| `split_name` | `str` | Dataset split: `train`, `val`, or `test` |

### Label Studio Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ls_xml_config` | `str` | `None` | Path to Label Studio XML config |
| `ls_parse_config` | `bool` | `true` | Parse LS config dynamically |
| `bbox_tolerance` | `int` | `5` | Bbox validation tolerance |
| `draw_original_bboxes` | `bool` | `false` | Draw original bboxes on ROI crops |

### ROI Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `roi_config.random_roi_count` | `int` | `1` | Number of random background ROIs per image |
| `roi_config.roi_box_size` | `int` | `384` | Size of extracted ROI crops in pixels |
| `roi_config.min_roi_size` | `int` | `32` | Minimum detection size for ROI extraction |
| `roi_config.dark_threshold` | `float` | `0.5` | Dark pixel ratio threshold for filtering crops |
| `roi_config.background_class` | `str` | `background` | Name for background class label |
| `roi_config.save_format` | `str` | `jpg` | Image format for saved crops |
| `roi_config.quality` | `int` | `95` | JPEG quality for saved crops |

---

## Bulk ROI Config

For `bulk-create-roi-datasets`, uses **`source_paths`** (list of directories):

```yaml
source_paths:
  - D:/annotations/batch1/
  - D:/annotations/batch2/

source_format: ls
root: D:/data
split_name: val
# ... same roi_config fields
```

---

## Complete Example

```yaml
source_path: configs/yolo_data.yaml
source_format: yolo
dataset_name: wildlife_roi
root: D:/data
split_name: val
bbox_tolerance: 5
draw_original_bboxes: false
ls_xml_config: null
ls_parse_config: true

roi_config:
  random_roi_count: 1
  roi_box_size: 384
  min_roi_size: 32
  dark_threshold: 0.5
  background_class: "background"
  save_format: "jpg"
  quality: 95
```

### Output Structure

The command generates a classification-ready directory structure:

```
data/<dataset_name>/roi/
├── <split>/
│   ├── <class_name>/
│   │   ├── image_001_roi_0.jpg
│   │   ├── image_001_roi_1.jpg
│   │   └── ...
│   ├── background/
│   │   ├── image_001_bg_0.jpg
│   │   └── ...
│   └── ...
```

---

**See also:**

- [WilData CLI Reference](../../api-reference/wildata-cli.md)
- [Import Config Reference](import-config.md)
- [Dataset Preparation Tutorial](../../tutorials/dataset-preparation.md)
