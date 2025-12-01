# FiftyOne Launch Script

> **Location**: `scripts/launch_fiftyone.bat`

**Purpose**: Launch the FiftyOne app for interactive dataset visualization, detection review, and data exploration.

## Usage

```batch
scripts\launch_fiftyone.bat
```

The script automatically:
1. Changes to the project root directory
2. Loads environment variables from `.env` file
3. Launches FiftyOne app

## Command Executed

```batch
uv run --no-sync --env-file .env fiftyone app launch
```

Note: The `--no-sync` flag prevents `uv` from syncing dependencies.

## Access

Once launched, the FiftyOne app will be available at:
- **URL**: `http://localhost:5151`
- Opens automatically in your default web browser

## Features

FiftyOne provides:

- **Interactive Dataset Viewer**: Browse images and detections
- **Detection Visualization**: View bounding boxes, labels, and confidence scores
- **Filtering and Querying**: Filter by class, confidence, location, etc.
- **Statistics**: View dataset statistics and distributions
- **Export Capabilities**: Export datasets in various formats
- **Annotation Review**: Review and validate detections
- **Comparison**: Compare different model predictions

## Prerequisites

1. **FiftyOne Installed**: FiftyOne should be installed (`pip install fiftyone`)
2. **Dataset Available**: Detection results should be uploaded to FiftyOne (via detection/census scripts)
3. **Port Availability**: Port 5151 should be available
4. **Environment**: `.env` file should exist (for environment variables)

## Example Workflow

### 1. Run Detection with FiftyOne Export

First, ensure detection results are exported to FiftyOne:

```yaml
# In config/detection.yaml
output:
  dataset_name: "my_detections"  # Set dataset name
```

Run detection:
```batch
scripts\run_detection.bat
```

### 2. Launch FiftyOne

```batch
scripts\launch_fiftyone.bat
```

### 3. Explore Dataset

- Browse images and detections
- Filter by species, confidence, etc.
- View statistics and distributions
- Export results if needed

## Stopping FiftyOne

Press `Ctrl+C` in the terminal window to stop the FiftyOne server.

## Troubleshooting

### Port Already in Use

**Issue**: Port 5151 is already occupied

**Solutions**:
1. Close other FiftyOne instances
2. Kill process using port 5151
3. FiftyOne will try to use next available port automatically

### Dataset Not Found

**Issue**: No datasets visible in FiftyOne

**Solutions**:
1. Verify detection was run with `output.dataset_name` set
2. Check dataset name matches in config and FiftyOne
3. Ensure detection completed successfully
4. Check FiftyOne database is accessible

### FiftyOne Not Installed

**Issue**: Command not found or import error

**Solutions**:
1. Install FiftyOne: `uv run pip install fiftyone`
2. Verify installation: `uv run python -c "import fiftyone"`
3. Check Python environment is correct

### Connection Refused

**Issue**: Cannot connect to FiftyOne server

**Solutions**:
1. Verify FiftyOne server started successfully
2. Check terminal for error messages
3. Try accessing `http://127.0.0.1:5151` instead
4. Check firewall settings

### Slow Performance

**Issue**: FiftyOne is slow or unresponsive

**Solutions**:
1. Reduce dataset size (filter samples)
2. Use lower resolution images
3. Close other applications
4. Check system resources

## Related Documentation

- [Detection Script](run_detection.md)
- [Census Script](run_census.md)
- [FiftyOne Documentation](https://docs.voxel51.com/)

