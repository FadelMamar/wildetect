# GPS Extraction Configuration

> **Location**: `config/extract-gps.yaml`

**Purpose**: Configuration file for extracting GPS coordinates from images and visualizing detection results with geographic information. This config is used for the `extract-gps-coordinates` visualization command.

## Configuration Structure

### Complete Parameter Reference

```yaml
# Visualization Configuration
# This file contains all parameters needed for visualization commands

labelstudio:
  url: http://localhost:8080
  api_key: null  # <<< TO change HERE
  download_resources: false
  project_id: null
  json_path: null  # <<< TO change HERE
  dotenv_path: .env
  parse_ls_config: true
  ls_xml_config: null

csv_output_path: D:\PhD\Harvard Kruger River Surveys\20231017_LetabaRiver_BasaltA\annotations\detections_v2.csv
detection_type: "annotations"  # annotations, predictions

# Flight Specifications
flight_specs:
  sensor_height: 15.6  # mm
  focal_length: 16.0   # mm
  flight_height: 120.0  # meters

# Logging Configuration
logging:
  verbose: false
  log_file: null  # Will use default log path if null
```

### Parameter Descriptions

#### `labelstudio`
Label Studio integration configuration for importing annotations or predictions.

- **`url`** (string): Label Studio server URL
- **`api_key`** (string): Label Studio API key for authentication (set in `.env` file)
- **`download_resources`** (bool): Whether to download images from Label Studio
- **`project_id`** (int, optional): Label Studio project ID to import from
- **`json_path`** (string, optional): Path to Label Studio JSON export file
- **`dotenv_path`** (string): Path to `.env` file containing environment variables
- **`parse_ls_config`** (bool): Whether to parse Label Studio XML configuration
- **`ls_xml_config`** (string, optional): Path to Label Studio XML configuration file

#### `csv_output_path`
- **Type**: string
- **Description**: Path where GPS coordinates and detection data will be exported as CSV

#### `detection_type`
- **Type**: string
- **Options**: `"annotations"`, `"predictions"`
- **Description**: Type of detections to process. Use `"annotations"` for ground truth data, `"predictions"` for model outputs

#### `flight_specs`
Flight and camera specifications for geographic calculations.

- **`sensor_height`** (float): Camera sensor height in millimeters
- **`focal_length`** (float): Lens focal length in millimeters
- **`flight_height`** (float): Flight altitude in meters

#### `logging`
Logging configuration.

- **`verbose`** (bool): Enable verbose logging output
- **`log_file`** (string, optional): Path to log file. If `null`, uses default log path

---

## Example Configurations

### Extract GPS from Label Studio Annotations

```yaml
labelstudio:
  url: http://localhost:8080
  api_key: ${LABEL_STUDIO_API_KEY}  # From .env file
  download_resources: false
  project_id: 1
  dotenv_path: .env
  parse_ls_config: true

csv_output_path: results/gps_coordinates.csv
detection_type: "annotations"

flight_specs:
  sensor_height: 15.6
  focal_length: 16.0
  flight_height: 120.0

logging:
  verbose: true
  log_file: null
```

### Extract GPS from Predictions File

```yaml
labelstudio:
  json_path: results/predictions.json
  download_resources: false

csv_output_path: results/predictions_gps.csv
detection_type: "predictions"

flight_specs:
  sensor_height: 24.0
  focal_length: 35.0
  flight_height: 180.0

logging:
  verbose: false
```

### Extract GPS with Custom Label Studio Config

```yaml
labelstudio:
  url: http://localhost:8080
  api_key: null
  project_id: 2
  ls_xml_config: configs/label_studio_config.xml
  parse_ls_config: true

csv_output_path: results/annotations_gps.csv
detection_type: "annotations"

flight_specs:
  sensor_height: 15.6
  focal_length: 16.0
  flight_height: 120.0
```

---

## Best Practices

1. **API Key Security**: Store Label Studio API key in `.env` file, not in the config file
2. **Output Paths**: Use absolute paths or paths relative to project root for `csv_output_path`
3. **Flight Specs**: Ensure flight specifications match the actual survey parameters for accurate GPS calculations
4. **Detection Type**: Use `"annotations"` for ground truth data analysis, `"predictions"` for model evaluation
5. **Label Studio**: If using Label Studio, ensure the server is running and accessible
6. **CSV Output**: The output CSV will contain columns for image paths, GPS coordinates, and detection information

---

## Troubleshooting

### Label Studio Connection Failed

**Issue**: Cannot connect to Label Studio server

**Solutions**:
1. Verify Label Studio is running: `scripts/launch_labelstudio.bat`
2. Check `url` is correct (default: `http://localhost:8080`)
3. Verify API key is set correctly in `.env` file
4. Check network connectivity

### API Key Not Found

**Issue**: Label Studio API key error

**Solutions**:
1. Set `LABEL_STUDIO_API_KEY` in `.env` file
2. Or set `api_key` directly in config (not recommended for production)
3. Verify API key has correct permissions

### GPS Coordinates Missing

**Issue**: GPS coordinates not extracted from images

**Solutions**:
1. Verify images have EXIF GPS data
2. Check image file formats support EXIF (JPEG, TIFF)
3. Ensure images were taken with GPS-enabled camera
4. Check `flight_specs` are correct for geographic calculations

### CSV Output Not Created

**Issue**: CSV file not generated

**Solutions**:
1. Verify `csv_output_path` directory exists or can be created
2. Check file permissions for output directory
3. Ensure input data (annotations/predictions) is valid
4. Check logs for error messages

---

## Related Documentation

- [Configuration Overview](index.md)
- [GPS Extraction Script](../../scripts/wildetect/extract_gps.md)
- [Visualization Config](visualization.md)
- [Label Studio Setup](../../scripts/wildetect/launch_labelstudio.md)

