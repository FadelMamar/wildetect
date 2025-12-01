# Census Configuration

> **Location**: `config/census.yaml`

**Purpose**: Configuration file for wildlife census campaigns. Extends detection configuration with campaign-specific settings, analysis options, and export configurations for generating census reports and statistics.

## Configuration Structure

### Complete Parameter Reference

```yaml
# Census Campaign Configuration
# This file contains all parameters needed for the census command

# Campaign Configuration
campaign:
  id: "Sabie_granite"
  pilot_name: "Unknown"
  target_species: null  # List of target species for detection

# Detection Configuration (inherits from detection.yaml)
detection:
  # list of image paths
  image_paths: null

  # or image directory path
  image_dir: null

  # or exif gps update configuration
  exif_gps_update:
    image_folder: D:\PhD\Harvard Kruger River Surveys\20231012_SabieRiver_GraniteB\cam0
    csv_path: D:\PhD\Harvard Kruger River Surveys\20231012_SabieRiver_GraniteB\SE_SabieRiver_GraniteB_Rectimage.csv
    skip_rows: 4
    filename_col: "filename"
    lat_col: "latitude"
    lon_col: "longitude"
    alt_col: "altitude"

  # Model Configuration
  model:
    mlflow_model_name: "detector"
    mlflow_model_alias: "by-paul"
    device: "cuda"  # auto, cpu, cuda
  
  # Processing Configuration
  processing:
    batch_size: 8
    tile_size: 800
    overlap_ratio: 0.2
    pipeline_type: "mt"  # default, multi, async, mt, mt_simple, simple, raster
    queue_size: 64
    num_data_workers: 1
    num_inference_workers: 1
    pin_memory: true
    prefetch_factor: 2
    nms_threshold: 0.5
    max_errors: 5

  # Label Studio Configuration
  labelstudio:
    url: null
    api_key: null
    project_id: null
    download_resources: false
  
  # Flight Specifications
  flight_specs:
    sensor_height: 15.6  # mm
    focal_length: 16.0   # mm
    flight_height: 120.0  # meters
    gsd: null # cm/px  *MANDATORY FOR RASTER DETECTION*

  # Inference Service Configuration
  inference_service:
    url: null
    timeout: 60
  
  # Profiling Configuration
  profiling:
    enable: false
    memory_profile: false
    line_profile: false
    gpu_profile: false

# Export Configuration
export:
  to_fiftyone: false
  create_map: true
  output_directory: null  # Will use campaign_id if null
  export_to_labelstudio: true

# Logging Configuration
logging:
  verbose: false
  log_file: null  # Will use default log path if null
```

### Parameter Descriptions

#### `campaign`
Campaign metadata and identification.

- **`id`** (string): Unique campaign identifier (e.g., "Sabie_granite", "Summer_2024")
- **`pilot_name`** (string, optional): Name of the pilot or survey operator
- **`target_species`** (list, optional): List of target species for detection (e.g., `["elephant", "giraffe", "zebra"]`)

#### `detection`
Detection configuration (same structure as `detection.yaml`). See [Detection Config](detection.md) for detailed parameter descriptions.

Key parameters:
- **`image_paths`**, **`image_dir`**, or **`exif_gps_update`**: Input source configuration
- **`model`**: Model configuration for detection
- **`processing`**: Processing pipeline settings
- **`flight_specs`**: Flight and camera specifications
- **`labelstudio`**: Label Studio integration (optional)
- **`inference_service`**: External inference service (optional)
- **`profiling`**: Performance profiling options

#### `export`
Export and output configuration for census results.

- **`to_fiftyone`** (bool): Whether to export results to FiftyOne dataset
- **`create_map`** (bool): Whether to create geographic visualization maps
- **`output_directory`** (string, optional): Output directory for census results. If `null`, uses `campaign.id`
- **`export_to_labelstudio`** (bool): Whether to export detections to Label Studio for review

#### `logging`
Logging configuration.

- **`verbose`** (bool): Enable verbose logging
- **`log_file`** (string, optional): Log file path. If `null`, uses default log path

---

## Example Configurations

### Basic Census Campaign

```yaml
campaign:
  id: "Summer_2024_Survey"
  pilot_name: "John Doe"
  target_species: ["elephant", "giraffe", "zebra"]

detection:
  image_dir: "D:/census_images/2024/"
  
  model:
    mlflow_model_name: "detector"
    mlflow_model_alias: "production"
    device: "cuda"
  
  processing:
    batch_size: 32
    tile_size: 800
    pipeline_type: "mt"
  
  flight_specs:
    flight_height: 120.0
    gsd: 2.38

export:
  to_fiftyone: true
  create_map: true
  export_to_labelstudio: false

logging:
  verbose: true
```

### Census with GPS Update

```yaml
campaign:
  id: "Sabie_River_2024"
  pilot_name: "Jane Smith"
  target_species: ["buffalo", "waterbuck", "impala"]

detection:
  exif_gps_update:
    image_folder: "D:/survey_images/cam0"
    csv_path: "D:/gps_data/survey_coordinates.csv"
    skip_rows: 1
    filename_col: "filename"
    lat_col: "latitude"
    lon_col: "longitude"
    alt_col: "altitude"
  
  model:
    mlflow_model_name: "detector"
    mlflow_model_alias: "production"
    device: "cuda"
  
  processing:
    batch_size: 16
    pipeline_type: "mt"
  
  flight_specs:
    sensor_height: 15.6
    focal_length: 16.0
    flight_height: 120.0

export:
  to_fiftyone: true
  create_map: true
  output_directory: "census_results/sabie_river_2024"
  export_to_labelstudio: true
```

### Raster Census (Large Orthomosaic)

```yaml
campaign:
  id: "Ortho_Census_2024"
  target_species: ["elephant", "giraffe"]

detection:
  image_paths:
    - "D:/orthomosaics/large_survey.tif"
  
  model:
    mlflow_model_name: "detector"
    device: "cuda"
  
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

export:
  to_fiftyone: false
  create_map: true
  output_directory: "census_results/ortho_2024"
```

### Census with Label Studio Integration

```yaml
campaign:
  id: "Labeled_Census_2024"
  target_species: ["zebra", "wildebeest"]

detection:
  image_dir: "D:/labeled_images/"
  
  model:
    mlflow_model_name: "detector"
    device: "cuda"
  
  labelstudio:
    url: "http://localhost:8080"
    api_key: ${LABEL_STUDIO_API_KEY}
    project_id: 1
    download_resources: false
  
  processing:
    batch_size: 32
    pipeline_type: "simple"

export:
  to_fiftyone: true
  create_map: true
  export_to_labelstudio: true
```

---

## Best Practices

1. **Campaign ID**: Use descriptive, unique campaign IDs that include location and date (e.g., "Sabie_River_2024_06")

2. **Target Species**: Specify `target_species` to focus analysis on specific animals and improve filtering

3. **Output Directory**: Let the system use `campaign.id` as output directory, or specify a custom path

4. **Maps**: Enable `create_map: true` for geographic visualization of detections

5. **FiftyOne Export**: Use `to_fiftyone: true` for interactive dataset exploration and validation

6. **Label Studio**: Enable `export_to_labelstudio: true` to review and correct detections

7. **GPS Data**: Use `exif_gps_update` when images lack GPS metadata but you have coordinate data

8. **Flight Specs**: Ensure flight specifications match actual survey parameters for accurate geographic calculations

9. **Processing**: Use `pipeline_type: "mt"` for faster processing of multiple images

10. **GSD for Raster**: **MANDATORY** when using raster detection - calculate from flight parameters

---

## Troubleshooting

### Campaign ID Conflicts

**Issue**: Output directory already exists with same campaign ID

**Solutions**:
1. Use unique campaign IDs
2. Specify custom `output_directory`
3. Remove or rename existing output directory

### No Detections Found

**Issue**: Census completes but finds no animals

**Solutions**:
1. Check `target_species` matches class names in model
2. Verify model is appropriate for the survey area
3. Lower confidence thresholds in processing
4. Check images contain wildlife
5. Verify model alias/version is correct

### Map Generation Fails

**Issue**: Geographic maps not created

**Solutions**:
1. Ensure GPS data is available (EXIF or CSV)
2. Verify `flight_specs` are set correctly
3. Check `create_map: true` is enabled
4. Ensure sufficient disk space
5. Verify map libraries are installed

### Label Studio Export Fails

**Issue**: Cannot export to Label Studio

**Solutions**:
1. Verify Label Studio server is running
2. Check `labelstudio.url` is correct
3. Verify API key is set in `.env` file
4. Check project ID exists in Label Studio
5. Ensure network connectivity

### Memory Issues with Large Campaigns

**Issue**: Out of memory errors during census

**Solutions**:
1. Reduce `batch_size` in processing
2. Use smaller `tile_size`
3. Process images in smaller batches
4. Use `pipeline_type: "simple"` for lower memory
5. Close other applications

### GPS Coordinates Missing

**Issue**: Maps show no geographic data

**Solutions**:
1. Verify images have EXIF GPS data
2. Or configure `exif_gps_update` with CSV
3. Check CSV format matches expected columns
4. Verify `flight_specs` are set
5. Ensure GPS data is valid (lat/lon ranges)

---

## Related Documentation

- [Configuration Overview](../index.md)
- [Detection Config](detection.md)
- [Census Script](../../scripts/wildetect/run_census.md)
- [Census Campaign Tutorial](../../tutorials/census-campaign.md)
- [Visualization Config](visualization.md)

