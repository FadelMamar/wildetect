# WilData Configuration Reference

Documentation for all WilData configuration files used in data management operations.

## Configuration Files

| File | Purpose |
|------|---------|
| [import-config-example.yaml](#import-config-exampleyaml) | Dataset import configuration |
| [bulk-import-*.yaml](#bulk-import-configs) | Bulk import configurations |
| [roi-create-config.yaml](#roi-create-configyaml) | ROI dataset creation |
| [bulk-roi-create-config.yaml](#bulk-roi-create-configyaml) | Bulk ROI creation |
| [gps-update-config-example.yaml](#gps-update-config-exampleyaml) | GPS metadata update |
| [label_studio_config.xml](#label-studio-configxml) | Label Studio interface |

---

## import-config-example.yaml

**Purpose**: Configure dataset import with transformations

**Location**: `wildata/configs/import-config-example.yaml`

### Complete Configuration

```yaml
# Required: Source Information
source_path: "D:/annotations/dataset.json"
source_format: "coco"  # coco, yolo, ls (Label Studio)
dataset_name: "my_dataset"

# Pipeline Configuration
root: "D:/data"
split_name: "train"  # train, val, test
enable_dvc: false
processing_mode: "batch"  # batch, streaming
track_with_dvc: false
bbox_tolerance: 5

# Label Studio Options (for ls format)
dotenv_path: ".env"
ls_xml_config: "configs/label_studio_config.xml"
ls_parse_config: false

# ROI Configuration
disable_roi: false
roi_config:
  random_roi_count: 2              # Background samples per image
  roi_box_size: 384                # ROI size (pixels)
  min_roi_size: 32                 # Minimum object size
  dark_threshold: 0.7              # Dark image threshold
  background_class: "background"
  save_format: "jpg"               # jpg, png
  quality: 95                      # JPEG quality
  sample_background: true          # Sample background regions

# Transformation Pipeline
transformations:
  # Bbox Clipping
  enable_bbox_clipping: true
  bbox_clipping:
    tolerance: 5                   # Pixels outside image allowed
    skip_invalid: false            # Skip invalid bboxes
  
  # Data Augmentation
  enable_augmentation: false
  augmentation:
    rotation_range: [-45, 45]     # Rotation degrees
    probability: 1.0               # Augmentation probability
    brightness_range: [-0.2, 0.4]
    scale: [1.0, 2.0]
    translate: [-0.1, 0.2]
    shear: [-5, 5]
    contrast_range: [-0.2, 0.4]
    noise_std: [0.01, 0.1]
    seed: 41
    num_transforms: 2              # Augmentations per image
  
  # Image Tiling
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640                    # Tile stride
    min_visibility: 0.7            # Min object visibility
    max_negative_tiles_in_negative_image: 2
    negative_positive_ratio: 1.0
    dark_threshold: 0.7
```

---

## bulk-import Configs

**Purpose**: Configure batch import of multiple datasets

**Files**:
- `bulk-import-train.yaml`
- `bulk-import-val.yaml`
- `bulk-import-config-example.yaml`

### Configuration Format

```yaml
source_paths:
  - "D:/annotations/dataset1.json"
  - "D:/annotations/dataset2.json"
  - "D:/annotations/dataset3.json"

source_format: "coco"
root: "D:/data"
split_name: "train"

# Shared settings (same as import-config)
processing_mode: "batch"
bbox_tolerance: 5

transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
```

---

## roi-create-config.yaml

**Purpose**: Configure ROI dataset creation

**Location**: `wildata/configs/roi-create-config.yaml`

### Configuration

```yaml
source_path: "annotations.json"
source_format: "coco"
dataset_name: "roi_dataset"

root: "data"
split_name: "val"  # Usually val or test
bbox_tolerance: 5

roi_config:
  roi_box_size: 128              # ROI crop size
  min_roi_size: 32               # Min object size to extract
  random_roi_count: 10           # Background samples per image
  dark_threshold: 0.7
  background_class: "background"
  save_format: "jpg"
  quality: 95
  padding: 10                    # Padding around object
  sample_background: true
  
  # Advanced options
  aspect_ratio_range: [0.5, 2.0]  # Valid aspect ratios
  min_object_area: 32             # Min area (pixels²)
  
ls_xml_config: null
ls_parse_config: false
draw_original_bboxes: false
```

---

## bulk-roi-create-config.yaml

**Purpose**: Bulk ROI dataset creation

### Configuration

```yaml
source_paths:
  - "dataset1.json"
  - "dataset2.json"

source_format: "coco"
split_name: "val"

roi_config:
  roi_box_size: 128
  random_roi_count: 5
  background_class: "background"
```

---

## gps-update-config-example.yaml

**Purpose**: Update image GPS from CSV

**Location**: `wildata/configs/gps-update-config-example.yaml`

### Configuration

```yaml
image_folder: "D:/images/"
csv_path: "gps_coordinates.csv"
output_dir: "D:/images_with_gps/"

# CSV Parsing
skip_rows: 0
filename_col: "filename"
lat_col: "latitude"
lon_col: "longitude"
alt_col: "altitude"

# Options
overwrite_existing: false        # Overwrite existing GPS
create_backup: true              # Backup original files
validate_coordinates: true       # Validate GPS coordinates
```

### CSV Format

```csv
filename,latitude,longitude,altitude
image001.jpg,40.7128,-74.0060,10.5
image002.jpg,40.7589,-73.9851,15.2
```

---

## label_studio_config.xml

**Purpose**: Label Studio annotation interface configuration

**Location**: `wildata/configs/label_studio_config.xml`

### Example Configuration

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="elephant" background="red"/>
    <Label value="giraffe" background="blue"/>
    <Label value="zebra" background="green"/>
    <Label value="buffalo" background="yellow"/>
  </RectangleLabels>
</View>
```

---

## Configuration Examples

### Import COCO with Tiling

```yaml
source_path: "D:/coco/annotations.json"
source_format: "coco"
dataset_name: "wildlife_tiled"

root: "D:/data"
split_name: "train"

transformations:
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
    min_visibility: 0.7
```

### Import Label Studio

```yaml
source_path: "D:/label_studio/export.json"
source_format: "ls"
dataset_name: "annotated_data"

ls_xml_config: "configs/label_studio_config.xml"
ls_parse_config: true

roi_config:
  roi_box_size: 128
  random_roi_count: 5
```

### Import with Full Pipeline

```yaml
source_path: "raw_annotations.json"
source_format: "coco"
dataset_name: "processed_dataset"

transformations:
  enable_bbox_clipping: true
  enable_tiling: true
  tiling:
    tile_size: 800
    stride: 640
  enable_augmentation: true
  augmentation:
    num_transforms: 2
    probability: 0.8

roi_config:
  roi_box_size: 384
  random_roi_count: 10
```

---

## Best Practices

1. **Use absolute paths** for cross-platform compatibility
2. **Enable bbox_clipping** to fix annotation errors
3. **Tile large images** for better training
4. **Sample background ROIs** for balanced datasets
5. **Version control** configuration changes

---

## Next Steps

- [WilData Scripts](../../scripts/wildata/index.md)
- [Dataset Preparation Tutorial](../../tutorials/dataset-preparation.md)
- [WilData API Reference](../../api-reference/wildata-api.md)

