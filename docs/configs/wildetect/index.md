# WildDetect Configuration Reference

This page documents all configuration files used by WildDetect for detection, census, and related operations.

## Overview

Configuration files are located in the `config/` directory and use YAML format for easy editing.

## Configuration Files

| File | Purpose |
|------|---------|
| [detection.yaml](#detectionyaml) | Main detection configuration |
| [census.yaml](#censusyaml) | Census campaign configuration |
| [benchmark.yaml](#benchmarkyaml) | Performance benchmarking |
| [visualization.yaml](#visualizationyaml) | Visualization settings |
| [extract-gps.yaml](#extract-gpsyaml) | GPS extraction configuration |
| [detector_registration.yaml](#detector_registrationyaml) | Model registration |
| [config.yaml](#configyaml) | Main application config |
| [class_mapping.json](#class_mappingjson) | Class ID to name mapping |

---

## detection.yaml

**Purpose**: Configure wildlife detection pipeline parameters.

**Location**: `config/detection.yaml`

### Complete Parameter Reference

```yaml
# Model Configuration
model:
  mlflow_model_name: "detector"          # Model name in MLflow registry
  mlflow_model_alias: "production"       # Model version alias
  mlflow_model_version: null             # Specific version (overrides alias)
  model_path: null                       # Direct path to model file (alternative to MLflow)
  device: "cuda"                         # Device: "cuda", "cpu", "auto"

# Input Sources (choose one)
image_paths:                             # List of specific image paths
  - "path/to/image1.jpg"
  - "path/to/image2.tif"
  
image_dir: null                          # Directory containing images

# EXIF GPS Update (optional)
exif_gps_update:
  image_folder: null                     # Folder with images to update
  csv_path: null                         # CSV with GPS coordinates
  skip_rows: 4                           # Rows to skip in CSV
  filename_col: "filename"               # Column name for filenames
  lat_col: "latitude"                    # Column name for latitude
  lon_col: "longitude"                   # Column name for longitude
  alt_col: "altitude"                    # Column name for altitude

# Label Studio Integration (optional)
labelstudio:
  url: null                              # Label Studio URL
  api_key: null                          # API key
  project_id: null                       # Project ID
  download_resources: false              # Download images from LS

# Processing Configuration
processing:
  batch_size: 32                         # Batch size for inference
  tile_size: 800                         # Tile size for large images (pixels)
  overlap_ratio: 0.2                     # Overlap ratio between tiles (0.0-1.0)
  pipeline_type: "raster"                # Pipeline: "raster", "multithreaded", "simple", "default"
  queue_size: 64                         # Queue size for multithreaded pipeline
  num_data_workers: 1                    # Number of data loading workers
  num_inference_workers: 1               # Number of inference workers (multiprocessing)
  prefetch_factor: 2                     # Batches to prefetch per worker
  pin_memory: true                       # Pin memory for faster GPU transfer
  nms_threshold: 0.5                     # NMS threshold for detection stitching
  max_errors: 5                          # Max errors before stopping
  confidence_threshold: 0.5              # Minimum confidence for detections

# Flight Specifications
flight_specs:
  sensor_height: 24                      # Camera sensor height (mm)
  focal_length: 35.0                     # Lens focal length (mm)
  flight_height: 180.0                   # Flight altitude (meters)
  gsd: 2.38                              # Ground Sample Distance (cm/pixel) - REQUIRED for raster

# Inference Service (optional)
inference_service:
  url: null                              # URL for external inference service
  # Example: "http://localhost:4141/predict"
  timeout: 60                            # Request timeout (seconds)

# Profiling Configuration
profiling:
  enable: false                          # Enable profiling
  memory_profile: false                  # Profile memory usage
  line_profile: false                    # Line-by-line profiling
  gpu_profile: false                     # GPU memory profiling

# Output Configuration
output:
  directory: "results"                   # Output directory for results
  dataset_name: null                     # FiftyOne dataset name (null to disable)
  save_visualizations: true              # Save visualization images
  save_crops: false                      # Save detection crops
  export_formats: ["json", "csv"]        # Export formats

# Logging Configuration
logging:
  verbose: false                         # Verbose logging
  log_file: null                         # Log file path (null for default)
  log_level: "INFO"                      # Log level: DEBUG, INFO, WARNING, ERROR
```

### Example Configurations

#### Basic Detection
```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

image_dir: "D:/images/survey/"

processing:
  batch_size: 32
  pipeline_type: "simple"

output:
  directory: "results/detections"
```

#### Raster Detection (Large GeoTIFF)
```yaml
model:
  mlflow_model_name: "detector"
  device: "cuda"

image_paths:
  - "D:/orthomosaics/ortho_large.tif"

processing:
  tile_size: 800
  overlap_ratio: 0.2
  pipeline_type: "raster"
  nms_threshold: 0.5

flight_specs:
  gsd: 2.38  # Required for raster

output:
  directory: "results/raster_detections"
```

---

## census.yaml

**Purpose**: Configure wildlife census campaigns with statistics and analysis.

**Location**: `config/census.yaml`

### Complete Parameter Reference

```yaml
# Campaign Information
campaign:
  name: "Summer_2024_Survey"             # Campaign name
  target_species: ["elephant", "giraffe", "zebra"]  # Target species list
  area_name: "Serengeti_North"           # Survey area name
  start_date: "2024-06-01"               # Campaign start date
  end_date: "2024-06-15"                 # Campaign end date
  pilot_name: null                       # Pilot name
  notes: null                            # Additional notes

# Model Configuration (same as detection.yaml)
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

# Image Sources
image_dir: "D:/census_images/"           # Directory with survey images

# Processing (same as detection.yaml)
processing:
  batch_size: 32
  tile_size: 800
  overlap_ratio: 0.2
  pipeline_type: "raster"
  nms_threshold: 0.5

# Flight Specifications
flight_specs:
  sensor_height: 24
  focal_length: 35.0
  flight_height: 120.0                   # Flight altitude (meters)
  gsd: 2.38                              # Ground Sample Distance (cm/pixel)

# Analysis Configuration
analysis:
  calculate_density: true                # Calculate population density
  density_unit: "per_km2"                # Density unit: "per_km2", "per_hectare"
  detect_hotspots: true                  # Identify concentration hotspots
  hotspot_radius: 500                    # Hotspot radius (meters)
  create_maps: true                      # Generate geographic maps
  coverage_analysis: true                # Analyze survey coverage
  species_distribution: true             # Species distribution analysis
  co_occurrence_analysis: true           # Species co-occurrence

# Statistics Configuration
statistics:
  confidence_bins: [0.5, 0.7, 0.9]       # Confidence bins for analysis
  size_bins: [50, 100, 200]              # Object size bins (pixels)
  group_size_analysis: true              # Analyze group sizes

# Visualization Settings
visualization:
  create_heatmaps: true                  # Create density heatmaps
  create_distribution_maps: true         # Create distribution maps
  create_flight_path_map: true           # Show flight path
  overlay_detections: true               # Overlay detections on maps
  color_by_species: true                 # Color code by species

# Output Configuration
output:
  directory: "census_results"            # Output directory
  dataset_name: "census_2024"            # FiftyOne dataset name
  generate_pdf_report: true              # Generate PDF report
  generate_excel: true                   # Generate Excel statistics
  save_individual_reports: true          # Save per-image reports

# Reporting
report:
  include_methodology: true              # Include methodology section
  include_confidence_analysis: true      # Include confidence analysis
  include_temporal_analysis: false       # Include time-based analysis
  executive_summary: true                # Include executive summary
  detailed_statistics: true              # Include detailed stats
```

### Example Configurations

#### Basic Census
```yaml
campaign:
  name: "Quick_Survey_2024"
  target_species: ["elephant"]

model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"

image_dir: "D:/survey/"

analysis:
  calculate_density: true
  create_maps: true

output:
  directory: "census_results"
  generate_pdf_report: true
```

#### Comprehensive Census
```yaml
campaign:
  name: "Annual_Census_2024"
  target_species: ["elephant", "giraffe", "zebra", "buffalo"]
  area_name: "Protected_Area_North"
  start_date: "2024-06-01"
  pilot_name: "John Doe"

model:
  mlflow_model_name: "detector"
  device: "cuda"

image_dir: "D:/annual_census/images/"

flight_specs:
  flight_height: 150.0
  gsd: 3.0

analysis:
  calculate_density: true
  detect_hotspots: true
  species_distribution: true
  co_occurrence_analysis: true

output:
  directory: "census_2024"
  generate_pdf_report: true
  generate_excel: true
```

---

## benchmark.yaml

**Purpose**: Configure performance benchmarking tests.

**Location**: `config/benchmark.yaml`

### Parameter Reference

```yaml
benchmark:
  test_images: ["test1.jpg", "test2.tif"]  # Images to benchmark
  iterations: 10                            # Number of iterations
  warmup_iterations: 2                      # Warmup runs
  measure_memory: true                      # Measure memory usage
  measure_gpu: true                         # Measure GPU utilization
  
models:
  - name: "yolo11n"
    path: "models/yolo11n.pt"
  - name: "yolo11s"
    path: "models/yolo11s.pt"

configurations:
  - batch_size: 1
  - batch_size: 8
  - batch_size: 32
```

---

## visualization.yaml

**Purpose**: Configure visualization settings.

**Location**: `config/visualization.yaml`

### Parameter Reference

```yaml
visualization:
  bbox_color: "red"                      # Bounding box color
  bbox_thickness: 2                      # Box line thickness
  show_labels: true                      # Show class labels
  show_confidence: true                  # Show confidence scores
  font_size: 12                          # Font size for labels
  dpi: 300                               # Output DPI for images
  
maps:
  basemap: "OpenStreetMap"               # Basemap provider
  zoom_level: 12                         # Default zoom level
  marker_size: 8                         # Marker size
```

---

## extract-gps.yaml

**Purpose**: Configure GPS coordinate extraction from images.

**Location**: `config/extract-gps.yaml`

### Parameter Reference

```yaml
gps_extraction:
  image_directory: "D:/images/"          # Directory with images
  output_file: "gps_coordinates.csv"     # Output CSV file
  recursive: true                        # Search subdirectories
  include_images_without_gps: false      # Include images without GPS
  
format:
  decimal_places: 6                      # GPS coordinate precision
  date_format: "%Y-%m-%d %H:%M:%S"       # Date format
```

---

## detector_registration.yaml

**Purpose**: Configure model registration to MLflow.

**Location**: `config/detector_registration.yaml`

### Parameter Reference

```yaml
registration:
  model_path: "models/best.pt"           # Path to model weights
  model_name: "wildlife_detector"        # Model name in registry
  model_type: "detector"                 # "detector" or "classifier"
  
  description: |
    YOLO11n model trained on aerial wildlife images
    Dataset: Wildlife Aerial v2.0
    Training date: 2024-01-15
  
  tags:
    framework: "yolo"
    version: "11n"
    dataset: "wildlife_v2"
    map50: "0.89"
    map50_95: "0.76"
    
  aliases:
    - "production"
    - "latest"
    - "v2.0"
  
  artifacts:
    - "configs/training_config.yaml"
    - "logs/training.log"
```

---

## config.yaml

**Purpose**: Main application configuration.

**Location**: `config/config.yaml`

### Parameter Reference

```yaml
app:
  name: "WildDetect"
  version: "1.0.0"
  
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "wildlife_detection"
  
paths:
  data_root: "D:/data/"
  models_root: "D:/models/"
  results_root: "D:/results/"
  
defaults:
  device: "cuda"
  batch_size: 32
  confidence_threshold: 0.5
```

---

## class_mapping.json

**Purpose**: Map class IDs to class names.

**Location**: `config/class_mapping.json`

### Format

```json
{
  "0": "elephant",
  "1": "giraffe",
  "2": "zebra",
  "3": "buffalo",
  "4": "wildebeest"
}
```

---

## Configuration Best Practices

### 1. Use Environment Variables

For sensitive data:

```yaml
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}

labelstudio:
  api_key: ${LABEL_STUDIO_API_KEY}
```

### 2. Create Config Variants

For different scenarios:

```
config/
├── detection.yaml          # Default
├── detection_dev.yaml      # Development
├── detection_prod.yaml     # Production
└── detection_test.yaml     # Testing
```

### 3. Document Custom Settings

Add comments to configs:

```yaml
processing:
  batch_size: 16  # Reduced for 8GB GPU
  tile_size: 640  # Smaller tiles for memory efficiency
```

### 4. Version Control

Track configuration changes:

```bash
git add config/
git commit -m "Update detection config for new model"
```

---

## Troubleshooting

### Invalid Configuration

**Issue**: Config validation fails

**Solutions**:
1. Check YAML syntax (indentation, colons)
2. Verify required fields are present
3. Check data types match expected types
4. Use YAML validator online

### Path Not Found

**Issue**: Image paths or model paths not found

**Solutions**:
1. Use absolute paths
2. Check path separators (use forward slashes)
3. Verify files exist
4. Check permissions

### Model Loading Fails

**Issue**: Can't load model from MLflow

**Solutions**:
1. Verify MLflow server is running
2. Check `mlflow_model_name` is correct
3. Verify model exists in registry
4. Check `MLFLOW_TRACKING_URI` environment variable

---

## Next Steps

- [WildDetect Scripts](../../scripts/wildetect/index.md)
- [Detection Tutorial](../../tutorials/end-to-end-detection.md)
- [Census Tutorial](../../tutorials/census-campaign.md)
- [CLI Reference](../../api-reference/wildetect-cli.md)

