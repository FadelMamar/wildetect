# UI Launch Script

> **Location**: `scripts/launch_ui.bat`

**Purpose**: Launch the WildDetect Streamlit web interface for interactive detection, configuration editing, and results visualization.

## Usage

```batch
scripts\launch_ui.bat
```

The script automatically:
1. Changes to the project root directory
2. Launches the Streamlit UI server

## Command Executed

```batch
uv run wildetect services ui
```

## Access

Once launched, the UI will be available at:
- **URL**: `http://localhost:8501`
- Opens automatically in your default web browser

## Features

The Streamlit UI provides:

- **Interactive Detection**: Upload images and run detection in real-time
- **Configuration Editor**: Edit and test configuration files through the web interface
- **Results Visualization**: View detection results with bounding boxes and statistics
- **Real-time Processing**: Monitor detection progress and see results as they're generated
- **Model Selection**: Choose models from MLflow registry
- **Batch Processing**: Process multiple images through the web interface

## Prerequisites

1. **Dependencies**: All Python dependencies installed via `uv sync`
2. **Streamlit**: Streamlit should be installed (included in dependencies)
3. **Port Availability**: Port 8501 should be available (default Streamlit port)

## Example Workflow

### 1. Launch UI

```batch
scripts\launch_ui.bat
```

### 2. Access Interface

Open browser to `http://localhost:8501` (usually opens automatically)

### 3. Use Features

- Upload images for detection
- Configure detection parameters
- View results and visualizations
- Export results

## Stopping the UI

Press `Ctrl+C` in the terminal window to stop the Streamlit server.

## Troubleshooting

### Port Already in Use

**Issue**: Port 8501 is already occupied

**Solutions**:
1. Close other Streamlit instances
2. Kill process using port 8501: `netstat -ano | findstr :8501`
3. Use different port (modify script to add `--server.port 8502`)

### UI Won't Load

**Issue**: Browser shows connection error

**Solutions**:
1. Verify Streamlit server started successfully
2. Check firewall settings
3. Try accessing `http://127.0.0.1:8501` instead
4. Check terminal for error messages

### Dependencies Missing

**Issue**: Import errors when launching UI

**Solutions**:
1. Run `uv sync` to install dependencies
2. Verify Streamlit is installed: `uv run pip list | findstr streamlit`
3. Check Python environment is activated

### Slow Performance

**Issue**: UI is slow or unresponsive

**Solutions**:
1. Close other applications
2. Reduce image sizes for upload
3. Use smaller batch sizes in detection
4. Check system resources (CPU, memory)

## Related Documentation

- [Detection Script](run_detection.md)
- [Configuration Reference](../../configs/wildetect/index.md)
- [CLI Reference](../../api-reference/wildetect-cli.md)

