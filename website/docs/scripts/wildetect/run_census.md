# Census Script

> **Location**: `scripts/run_census.bat`

**Purpose**: Run a complete wildlife census campaign with detection, statistics, geographic analysis, and report generation. This script executes the census pipeline configured in `config/census.yaml`.

## Usage

```batch
scripts\run_census.bat
```

The script automatically:
1. Changes to the project root directory
2. Loads environment variables from `.env` file
3. Runs the census command with profiling disabled (for production use)

## Command Executed

```batch
uv run --env-file .env --no-sync wildetect detection census -c config/census.yaml
```

Note: The `--no-sync` flag prevents `uv` from syncing dependencies, speeding up execution.

## Configuration

**Config File**: `config/census.yaml`

See [Census Configuration](../../configs/wildetect/census.md) for complete parameter reference.

## Prerequisites

1. **Environment Setup**:
   - `.env` file exists in project root
   - Environment variables configured

2. **Model Availability**:
   - Model registered in MLflow (or model path specified)
   - MLflow server running (if using MLflow models)

3. **Input Images**:
   - Images specified in `config/census.yaml`
   - GPS data available (for geographic analysis)

4. **Dependencies**:
   - All Python dependencies installed

## Example Workflow

### 1. Configure Census Campaign

Edit `config/census.yaml`:

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
    pipeline_type: "mt"
  
  flight_specs:
    flight_height: 120.0
    gsd: 2.38

export:
  to_fiftyone: true
  create_map: true
  export_to_labelstudio: false
```

### 2. Start MLflow (if using MLflow models)

```batch
scripts\launch_mlflow.bat
```

### 3. Run Census

```batch
scripts\run_census.bat
```

### 4. View Results

Results will be saved to the directory specified in `export.output_directory` (or `campaign.id` if not specified):

- Census statistics and counts
- Geographic visualizations and maps
- PDF reports (if enabled)
- FiftyOne dataset (if enabled)

## Output

The census script generates:

- **Detection Results**: All detections with coordinates and metadata
- **Statistics**: Species counts, densities, distributions
- **Geographic Maps**: Interactive maps showing detection locations
- **Reports**: PDF reports with analysis (if configured)
- **FiftyOne Dataset**: Dataset for interactive exploration (if enabled)
- **Label Studio Export**: Annotations for review (if enabled)

## Common Use Cases

### Basic Census

```yaml
campaign:
  id: "Quick_Survey_2024"
  target_species: ["elephant"]

detection:
  image_dir: "D:/survey/"
  
export:
  create_map: true
```

### Comprehensive Census with Analysis

```yaml
campaign:
  id: "Annual_Census_2024"
  target_species: ["elephant", "giraffe", "zebra", "buffalo"]

detection:
  image_dir: "D:/annual_census/images/"
  
export:
  to_fiftyone: true
  create_map: true
  export_to_labelstudio: true
```

### Census with GPS Update

```yaml
campaign:
  id: "GPS_Updated_Census"

detection:
  exif_gps_update:
    image_folder: "D:/images/cam0"
    csv_path: "D:/gps_data.csv"
    filename_col: "filename"
    lat_col: "latitude"
    lon_col: "longitude"
```

## Differences from Detection Script

The census script differs from `run_detection.bat` in several ways:

1. **Campaign Metadata**: Includes campaign ID, pilot name, target species
2. **Statistics**: Generates population statistics and analysis
3. **Geographic Analysis**: Creates maps and geographic visualizations
4. **Export Options**: Supports FiftyOne, Label Studio, and report generation
5. **Nested Config**: Detection config is nested under `detection:` key

## Troubleshooting

### Campaign ID Conflicts

**Issue**: Output directory already exists

**Solutions**:
1. Use unique campaign IDs
2. Specify custom `export.output_directory`
3. Remove or rename existing output directory

### No Detections Found

**Issue**: Census completes but finds no animals

**Solutions**:
1. Check `target_species` matches model class names
2. Verify model is appropriate for survey area
3. Lower confidence thresholds in processing
4. Check images actually contain wildlife

### Map Generation Fails

**Issue**: Geographic maps not created

**Solutions**:
1. Ensure GPS data is available (EXIF or CSV)
2. Verify `flight_specs` are set correctly
3. Check `export.create_map: true` is enabled
4. Ensure sufficient disk space

### Memory Issues

**Issue**: Out of memory errors

**Solutions**:
1. Reduce `batch_size` in detection config
2. Use smaller `tile_size`
3. Process images in smaller batches
4. Use `pipeline_type: "simple"` for lower memory

### Slow Processing

**Issue**: Census takes very long

**Solutions**:
1. Use `pipeline_type: "mt"` for multi-threading
2. Increase `batch_size` if memory allows
3. Use GPU (`device: "cuda"`)
4. Reduce number of images processed at once

## Related Documentation

- [Census Configuration](../../configs/wildetect/census.md)
- [Detection Script](run_detection.md)
- [Census Campaign Tutorial](../../tutorials/census-campaign.md)
- [Profiling Script](profile_census.md)

