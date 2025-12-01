# Detection Configuration

> **Location**: `config/detection.yaml`

**Purpose**: Main configuration file for the wildlife detection pipeline. Defines model settings, input sources, processing parameters, flight specifications, and output options for running detection on aerial imagery.

## Configuration Structure

### Complete Parameter Reference

```yaml
# Model Configuration
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "by-paul"
  device: "cuda"  # auto, cpu, cuda

# list of image paths
image_paths: 
 - D:\PhD\Data per camp\Orthos\Dry season orthos\ortho_k_1_4_5_rep_1.tif

# or image directory path
image_dir: null

# or exif gps update configuration
exif_gps_update:
  image_folder: null  # Path to folder containing images
  csv_path: null  # Path to CSV file with GPS coordinates
  skip_rows: 4  # Number of rows to skip in CSV (e.g., header row)
  filename_col: "filename"  # CSV column name for filenames
  lat_col: "latitude"  # CSV column name for latitude
  lon_col: "longitude"  # CSV column name for longitude
  alt_col: "altitude"  # CSV column name for altitude

# Label Studio Configuration
labelstudio:
  url: null
  api_key: null 
  project_id: null
  download_resources: false

# Processing Configuration
processing:
  batch_size: 32
  tile_size: 800
  overlap_ratio: 0.2
  pipeline_type: "raster"  # mt, mp, async, mt_simple, simple, default, raster
  queue_size: 64  # for multi-threaded pipeline
  num_data_workers: 1
  num_inference_workers: 1 # for multi-processing pipeline
  prefetch_factor: 2
  pin_memory: true
  nms_threshold: 0.5 # NMS threshold for detections stitching
  max_errors: 5
  
# Flight Specifications
flight_specs:
  sensor_height: 24  # mm
  focal_length: 35.0   # mm
  flight_height: 180.0  # meters
  gsd: 2.38 # cm/px  MANDATORY FOR RASTER DETECTION

# Inference Service Configuration
inference_service:
  url: null #http://localhost:4141/predict  # URL for external inference service
  timeout: 60

# Profiling Configuration
profiling:
  enable: false
  memory_profile: false
  line_profile: false
  gpu_profile: false

# Output Configuration
output:
  directory: "results-raster"
  dataset_name: null  # if null, fiftyone dataset upload is disabled

# Logging Configuration
logging:
  verbose: false
  log_file: null  # Will use default log path if null
```

### Parameter Descriptions

#### `model`
Model configuration for loading the detection model.

- **`mlflow_model_name`** (string): Name of the model in MLflow registry
- **`mlflow_model_alias`** (string): Model version alias (e.g., "production", "latest", "by-paul")
- **`device`** (string): Device for inference. Options: `"auto"`, `"cpu"`, `"cuda"`

#### Input Sources (choose one)
Three ways to specify input images:

**Option 1: `image_paths`**
- **Type**: list of strings
- **Description**: List of specific image file paths to process
- **Example**: `["path/to/image1.tif", "path/to/image2.tif"]`

**Option 2: `image_dir`**
- **Type**: string or null
- **Description**: Directory containing images to process
- **Example**: `"D:/survey_images/"`

**Option 3: `exif_gps_update`**
- **Type**: dict or null
- **Description**: Configuration for updating EXIF GPS data from CSV before processing
- **`image_folder`** (string): Path to folder with images
- **`csv_path`** (string): Path to CSV with GPS coordinates
- **`skip_rows`** (int): Rows to skip in CSV (header rows)
- **`filename_col`** (string): CSV column name for image filenames
- **`lat_col`** (string): CSV column name for latitude
- **`lon_col`** (string): CSV column name for longitude
- **`alt_col`** (string): CSV column name for altitude

#### `labelstudio`
Label Studio integration for importing annotations.

- **`url`** (string, optional): Label Studio server URL
- **`api_key`** (string, optional): Label Studio API key
- **`project_id`** (int, optional): Label Studio project ID to import from
- **`download_resources`** (bool): Whether to download images from Label Studio

#### `processing`
Processing pipeline configuration.

- **`batch_size`** (int): Batch size for model inference
- **`tile_size`** (int): Tile size in pixels for processing large images
- **`overlap_ratio`** (float): Overlap ratio between tiles (0.0-1.0)
- **`pipeline_type`** (string): Pipeline strategy. Options: `"raster"`, `"mt"` (multi-threaded), `"mp"` (multi-process), `"async"`, `"mt_simple"`, `"simple"`, `"default"`
- **`queue_size`** (int): Queue size for multi-threaded pipeline
- **`num_data_workers`** (int): Number of data loading workers
- **`num_inference_workers`** (int): Number of inference workers (for multi-processing)
- **`prefetch_factor`** (int): Number of batches to prefetch per worker
- **`pin_memory`** (bool): Pin memory for faster GPU transfer
- **`nms_threshold`** (float): Non-Maximum Suppression threshold for stitching detections across tiles (0.0-1.0)
- **`max_errors`** (int): Maximum errors before stopping processing

#### `flight_specs`
Flight and camera specifications for geographic calculations.

- **`sensor_height`** (float): Camera sensor height in millimeters
- **`focal_length`** (float): Lens focal length in millimeters
- **`flight_height`** (float): Flight altitude in meters
- **`gsd`** (float): Ground Sample Distance in cm/pixel. **MANDATORY for raster detection**

#### `inference_service`
External inference service configuration.

- **`url`** (string, optional): URL for external inference API (e.g., `"http://localhost:4141/predict"`)
- **`timeout`** (int): Request timeout in seconds

#### `profiling`
Performance profiling options.

- **`enable`** (bool): Enable general profiling
- **`memory_profile`** (bool): Profile memory usage
- **`line_profile`** (bool): Line-by-line profiling
- **`gpu_profile`** (bool): GPU memory profiling

#### `output`
Output configuration.

- **`directory`** (string): Output directory for results
- **`dataset_name`** (string, optional): FiftyOne dataset name. If `null`, dataset upload is disabled

#### `logging`
Logging configuration.

- **`verbose`** (bool): Enable verbose logging
- **`log_file`** (string, optional): Log file path. If `null`, uses default log path

---

## Example Configurations

### Basic Detection (Single Image)

```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

image_paths:
  - "D:/images/survey_2024/image1.tif"

processing:
  batch_size: 32
  tile_size: 800
  overlap_ratio: 0.2
  pipeline_type: "simple"

output:
  directory: "results/detections"
```

### Raster Detection (Large GeoTIFF)

```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

image_paths:
  - "D:/orthomosaics/large_ortho.tif"

processing:
  batch_size: 32
  tile_size: 800
  overlap_ratio: 0.2
  pipeline_type: "raster"
  nms_threshold: 0.5

flight_specs:
  sensor_height: 24
  focal_length: 35.0
  flight_height: 180.0
  gsd: 2.38  # Required for raster

output:
  directory: "results/raster_detections"
  dataset_name: "raster_survey_2024"
```

### Directory-Based Detection

```yaml
model:
  mlflow_model_name: "detector"
  device: "auto"

image_dir: "D:/survey_images/2024/"

processing:
  batch_size: 16
  pipeline_type: "mt"  # Multi-threaded
  queue_size: 64
  num_data_workers: 2

output:
  directory: "results/batch_detections"
```

### Detection with GPS Update

```yaml
model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

exif_gps_update:
  image_folder: "D:/images/survey/"
  csv_path: "D:/gps_coordinates.csv"
  skip_rows: 1
  filename_col: "filename"
  lat_col: "latitude"
  lon_col: "longitude"
  alt_col: "altitude"

processing:
  batch_size: 32
  pipeline_type: "simple"

output:
  directory: "results/gps_updated_detections"
```

### Multi-Threaded Detection

```yaml
model:
  mlflow_model_name: "detector"
  device: "cuda"

image_dir: "D:/large_dataset/"

processing:
  batch_size: 32
  tile_size: 800
  pipeline_type: "mt"  # Multi-threaded
  queue_size: 128
  num_data_workers: 4
  prefetch_factor: 2
  pin_memory: true

output:
  directory: "results/mt_detections"
```

---

## Best Practices

1. **Pipeline Type Selection**:
   - Use `"raster"` for large GeoTIFF files (orthomosaics)
   - Use `"mt"` (multi-threaded) for many small images
   - Use `"simple"` for basic single-image processing
   - Use `"mp"` (multi-process) for CPU-bound workloads

2. **Batch Size**: 
   - Start with 32, reduce if GPU memory errors occur
   - Increase for faster processing if memory allows

3. **Tile Size**:
   - 800 pixels is a good default
   - Reduce for lower memory usage
   - Increase for better context (if memory allows)

4. **Overlap Ratio**:
   - 0.2 (20%) is recommended for most cases
   - Increase to 0.3 for better edge detection
   - Decrease to 0.1 for faster processing

5. **GSD for Raster**:
   - **MANDATORY** for raster detection pipeline
   - Calculate from flight height, focal length, and sensor size
   - Critical for accurate geographic positioning

6. **NMS Threshold**:
   - 0.5 is a good default
   - Lower (0.3-0.4) for more detections
   - Higher (0.6-0.7) for fewer, higher-confidence detections

7. **Device Selection**:
   - Use `"auto"` to let the system choose
   - Use `"cuda"` explicitly if you have GPU
   - Use `"cpu"` for CPU-only systems

---

## Troubleshooting

### Model Not Found in MLflow

**Issue**: Cannot load model from MLflow registry

**Solutions**:
1. Verify MLflow server is running: `scripts/launch_mlflow.bat`
2. Check `mlflow_model_name` matches registry name
3. Verify `mlflow_model_alias` exists (e.g., "production", "latest")
4. Check MLflow tracking URI is correct
5. List available models: `mlflow models list`

### CUDA Out of Memory

**Issue**: GPU memory errors during inference

**Solutions**:
1. Reduce `batch_size` (try 16, 8, or 4)
2. Reduce `tile_size` (try 640 or 512)
3. Set `device: "cpu"` to use CPU instead
4. Close other GPU applications
5. Use `pipeline_type: "simple"` for lower memory usage

### Images Not Found

**Issue**: Cannot find input images

**Solutions**:
1. Verify `image_paths` or `image_dir` paths are correct
2. Use absolute paths instead of relative paths
3. Check file permissions
4. Ensure image files exist at specified locations
5. Verify image formats are supported (TIFF, JPEG, PNG)

### Raster Detection Fails

**Issue**: Raster pipeline errors

**Solutions**:
1. **Ensure `gsd` is set** - this is mandatory for raster detection
2. Verify input is a valid GeoTIFF file
3. Check file is not corrupted
4. Ensure sufficient disk space for processing
5. Try smaller `tile_size` if memory issues

### Slow Processing

**Issue**: Detection is very slow

**Solutions**:
1. Increase `batch_size` if memory allows
2. Use `pipeline_type: "mt"` for multi-threading
3. Increase `num_data_workers` for parallel data loading
4. Enable `pin_memory: true` for faster GPU transfer
5. Use GPU (`device: "cuda"`) instead of CPU
6. Reduce `tile_size` for faster tile processing

### Detections Missing at Tile Edges

**Issue**: Objects at tile boundaries are not detected

**Solutions**:
1. Increase `overlap_ratio` (try 0.3 or 0.4)
2. Lower `nms_threshold` to keep more detections
3. Use smaller `tile_size` for more overlap coverage
4. Check NMS stitching is working correctly

---

## Related Documentation

- [Configuration Overview](../index.md)
- [Census Config](census.md)
- [Detection Script](../../scripts/wildetect/run_detection.md)
- [End-to-End Detection Tutorial](../../tutorials/end-to-end-detection.md)
- [CLI Reference](../../api-reference/wildetect-cli.md)

