# GPS Extraction Script

> **Location**: `scripts/extract_gps.bat`

**Purpose**: Extract GPS coordinates from images and create visualizations with geographic information. This script runs the GPS extraction and visualization command.

## Usage

```batch
scripts\extract_gps.bat
```

The script automatically:
1. Changes to the project root directory
2. Runs the GPS extraction command

## Command Executed

```batch
uv run wildetect visualization extract-gps-coordinates -c config/extract-gps.yaml
```

## Configuration

**Config File**: `config/extract-gps.yaml`

See [GPS Extraction Configuration](../../configs/wildetect/extract-gps.md) for complete parameter reference.

## Prerequisites

1. **Images with GPS Data**: Images must have EXIF GPS metadata, OR
2. **Label Studio Data**: Label Studio project with GPS data, OR
3. **CSV with GPS**: CSV file with GPS coordinates matching image filenames

4. **Configuration**: `config/extract-gps.yaml` must be properly configured

## Example Workflow

### 1. Configure GPS Extraction

Edit `config/extract-gps.yaml`:

```yaml
labelstudio:
  url: http://localhost:8080
  api_key: ${LABEL_STUDIO_API_KEY}
  project_id: 1
  download_resources: false
  dotenv_path: .env

csv_output_path: results/gps_coordinates.csv
detection_type: "annotations"

flight_specs:
  sensor_height: 15.6
  focal_length: 16.0
  flight_height: 120.0

logging:
  verbose: true
```

### 2. Run GPS Extraction

```batch
scripts\extract_gps.bat
```

### 3. View Results

- CSV file with GPS coordinates at `csv_output_path`
- Geographic visualizations (if enabled)
- Summary statistics

## Output

The GPS extraction script generates:

- **CSV File**: GPS coordinates and detection data exported to CSV
- **Geographic Visualizations**: Maps showing detection locations (if configured)
- **Statistics**: Summary of GPS coverage and detection distribution
- **Coverage Maps**: Visualization of survey coverage area

## Common Use Cases

### Extract GPS from Label Studio Annotations

```yaml
labelstudio:
  url: http://localhost:8080
  api_key: ${LABEL_STUDIO_API_KEY}
  project_id: 1

csv_output_path: results/annotations_gps.csv
detection_type: "annotations"
```

### Extract GPS from Predictions

```yaml
labelstudio:
  json_path: results/predictions.json

csv_output_path: results/predictions_gps.csv
detection_type: "predictions"
```

### Extract GPS from Images with EXIF

If images have EXIF GPS data, the script will extract it automatically when processing through Label Studio or detection results.

## Troubleshooting

### Label Studio Connection Failed

**Issue**: Cannot connect to Label Studio server

**Solutions**:
1. Start Label Studio: `scripts\launch_labelstudio.bat`
2. Verify `url` in config is correct (default: `http://localhost:8080`)
3. Check API key is set in `.env` file
4. Verify network connectivity

### GPS Coordinates Missing

**Issue**: No GPS coordinates extracted

**Solutions**:
1. Verify images have EXIF GPS data
2. Check image formats support EXIF (JPEG, TIFF)
3. Ensure images were taken with GPS-enabled camera
4. Verify Label Studio project contains GPS data
5. Check CSV format if using CSV input

### CSV Output Not Created

**Issue**: CSV file not generated

**Solutions**:
1. Verify `csv_output_path` directory exists or can be created
2. Check file permissions for output directory
3. Ensure input data (annotations/predictions) is valid
4. Check logs for error messages
5. Verify sufficient disk space

### API Key Error

**Issue**: Label Studio API key authentication fails

**Solutions**:
1. Set `LABEL_STUDIO_API_KEY` in `.env` file
2. Get API key from Label Studio: Settings → Access Token
3. Verify API key has correct permissions
4. Check API key format is correct

### Flight Specs Missing

**Issue**: Geographic calculations fail

**Solutions**:
1. Ensure `flight_specs` are set in config
2. Verify flight specifications match actual survey
3. Check sensor_height, focal_length, and flight_height are correct
4. Ensure GSD can be calculated from flight specs

## Related Documentation

- [GPS Extraction Configuration](../../configs/wildetect/extract-gps.md)
- [Label Studio Launch Script](launch_labelstudio.md)
- [Visualization Config](../../configs/wildetect/visualization.md)
- [CLI Reference](../../api-reference/wildetect-cli.md)

