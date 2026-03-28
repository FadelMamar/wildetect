# Census Campaign Tutorial

Learn how to conduct a complete wildlife census campaign using WildDetect.

## Overview

A census campaign includes detection, population statistics, geographic analysis, and reporting — all orchestrated through a single config-driven command.

## Prerequisites

- WildDetect installed
- Aerial survey images with GPS EXIF data
- Trained detection model registered in MLflow (see [Model Training](model-training.md))
- Optional: MLflow server running

## Step 1: Organize Survey Data

Organize your aerial survey images by flight/campaign:

```
survey_2024/
├── images/              # Survey images with GPS EXIF
│   ├── flight1/
│   ├── flight2/
│   └── ...
└── config/
    └── census.yaml      # Census configuration
```

## Step 2: Configure Census

Create or edit `config/census.yaml`:

```yaml
# Campaign Configuration
campaign:
  id: "Summer_2024_Survey"
  pilot_name: "John Doe"
  target_species: ["elephant", "giraffe", "zebra"]

# Detection Configuration
detection:
  # Image source (use ONE of these)
  image_dir: D:/survey_2024/images/
  image_paths: null        # Alternative: list of specific image paths

  # Merging thresholds for overlapping detections
  merging_iou_threshold: -0.7
  merging_min_overlap_threshold: 0.078

  # Model Configuration
  model:
    mlflow_model_name: "detector"
    mlflow_model_alias: "production"
    device: "cuda"

  # Processing Configuration
  processing:
    batch_size: 8
    tile_size: 800
    overlap_ratio: 0.2
    pipeline_type: "mt"     # mt, mp, async, simple, raster
    queue_size: 64
    num_data_workers: 1
    num_inference_workers: 1
    pin_memory: true
    nms_threshold: 0.5
    max_errors: 5

  # Label Studio (optional — for loading images from LS)
  labelstudio:
    url: null
    api_key: null
    project_id: null
    download_resources: false

  # Flight Specifications
  flight_specs:
    sensor_height: 24       # mm
    focal_length: 35        # mm
    flight_height: 180.0    # meters
    gsd: null               # cm/px (mandatory for raster detection)

  # Inference Service (optional — for remote inference)
  inference_service:
    url: null
    timeout: 60

  # Profiling (optional)
  profiling:
    enable: false
    memory_profile: false
    line_profile: false
    gpu_profile: false

# Export Configuration
export:
  to_fiftyone: true
  create_map: true
  output_directory: D:/survey_2024/census_results/
  export_to_labelstudio: true

# Logging
logging:
  verbose: false
  log_file: null
```

See [Census Config Reference](../configs/wildetect/census.md) for all configuration fields.

## Step 3: Run Census

```bash
wildetect detection census -c config/census.yaml
```

Or use the provided script:

```bash
scripts\run_census.bat
```

The command will:

1. ✅ Load and validate configuration
2. ✅ Initialize detection pipeline (with profiling if enabled)
3. ✅ Run detection on all images
4. ✅ Merge overlapping detections
5. ✅ Compute species counts and statistics
6. ✅ Export to FiftyOne dataset
7. ✅ Export to Label Studio (for review)
8. ✅ Display results summary

## Step 4: Review Results

```
census_results/
├── results.json                 # All detections with coordinates
├── statistics.json              # Population statistics
└── maps/                        # Geographic visualizations
    └── detection_map.html       # Interactive Folium map
```

## Step 5: View Results in FiftyOne

After the census, results are exported to a FiftyOne dataset:

```bash
# Launch FiftyOne app
wildetect services fiftyone -a launch
```

The dataset is named `campaign_{campaign_id}` — you can filter by species, confidence, and location in the FiftyOne UI.

## Step 6: Geographic Visualization

Extract GPS coordinates and generate CSV reports:

```bash
wildetect visualization extract-gps-coordinates -c config/census.yaml
```

## Pipeline Types

Choose the best pipeline type based on your hardware and dataset:

| Pipeline | Flag | Best For |
|----------|------|----------|
| Multi-threaded | `mt` | Standard images, GPU available |
| Multi-threaded (simple) | `mt_simple` | Simpler threading model |
| Multi-process | `mp` | CPU-bound processing |
| Async | `async` | I/O-bound workloads |
| Simple | `simple` | Debugging, small datasets |
| Raster | `raster` | GeoTIFF / large raster images |
| Default | `default` | Basic single-threaded |

## Image Sources

The census can load images from three sources:

1. **Image directory** (`image_dir`): All images in a folder
2. **Image paths** (`image_paths`): Explicit list of image files
3. **Label Studio** (`labelstudio.project_id`): Load from a Label Studio project

Only one source should be configured at a time.

---

**Next Steps:**

- [End-to-End Detection](end-to-end-detection.md) — Simpler detection workflow
- [Census Config Reference](../configs/wildetect/census.md) — All Census config fields
- [WildDetect CLI Reference](../api-reference/wildetect-cli.md) — Full CLI documentation
