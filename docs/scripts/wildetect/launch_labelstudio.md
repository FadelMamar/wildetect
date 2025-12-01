# Label Studio Launch Script

> **Location**: `scripts/launch_labelstudio.bat`

**Purpose**: Launch Label Studio for data annotation, detection review, and annotation management. This script activates a separate virtual environment and starts the Label Studio server.

## Usage

```batch
scripts\launch_labelstudio.bat
```

The script automatically:
1. Changes to the scripts directory
2. Deactivates any active virtual environments
3. Activates the Label Studio virtual environment (`.venv-ls`)
4. Installs/updates Label Studio (version 1.20.0)
5. Launches Label Studio server with environment variables loaded

## Command Executed

```batch
call ..\.venv-ls\Scripts\activate
call uv pip install label-studio==1.20.0
call uv run --active --no-sync --env-file ..\.env label-studio start -p 8080
```

## Access

Once launched, Label Studio will be available at:
- **URL**: `http://localhost:8080`
- Opens automatically in your default web browser

## Features

Label Studio provides:

- **Data Annotation**: Annotate images with bounding boxes, polygons, etc.
- **Detection Review**: Review and correct model predictions
- **Project Management**: Organize annotation projects
- **Team Collaboration**: Multiple annotators can work on projects
- **Export Formats**: Export annotations in various formats (COCO, YOLO, etc.)
- **Import Predictions**: Import model predictions for review

## Prerequisites

1. **Virtual Environment**: Label Studio virtual environment should exist at `.venv-ls/`
2. **Port Availability**: Port 8080 should be available
3. **Environment File**: `.env` file should exist in project root
4. **Label Studio Config**: XML configuration file (optional, for custom labeling interface)

## Setup

### First-Time Setup

If `.venv-ls` doesn't exist, create it:

```batch
python -m venv .venv-ls
.venv-ls\Scripts\activate
pip install label-studio==1.20.0
```

### Environment Variables

Set in `.env` file:

```bash
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key_here
```

## Example Workflow

### 1. Launch Label Studio

```batch
scripts\launch_labelstudio.bat
```

### 2. Create Project

- Open `http://localhost:8080`
- Create new project
- Configure labeling interface (or use XML config)
- Import images

### 3. Annotate or Review

- Annotate images manually, or
- Import model predictions for review
- Correct and validate annotations

### 4. Export Annotations

- Export in desired format (COCO, YOLO, etc.)
- Use exported annotations for training or evaluation

## Integration with WildDetect

Label Studio can be integrated with WildDetect in several ways:

### Import Predictions for Review

```yaml
# In config/detection.yaml or config/census.yaml
labelstudio:
  url: http://localhost:8080
  api_key: ${LABEL_STUDIO_API_KEY}
  project_id: 1
  download_resources: false
```

### Export to Label Studio

```yaml
# In config/census.yaml
export:
  export_to_labelstudio: true
```

## Stopping Label Studio

Press `Ctrl+C` in the terminal window to stop the Label Studio server.

## Troubleshooting

### Virtual Environment Not Found

**Issue**: `.venv-ls` directory doesn't exist

**Solutions**:
1. Create virtual environment (see Setup section)
2. Verify path is correct: `..\.venv-ls\Scripts\activate`
3. Check script is run from `scripts/` directory

### Port Already in Use

**Issue**: Port 8080 is already occupied

**Solutions**:
1. Close other Label Studio instances
2. Kill process using port 8080
3. Modify script to use different port: `-p 8081`

### Label Studio Won't Start

**Issue**: Server fails to start

**Solutions**:
1. Check virtual environment is activated correctly
2. Verify Label Studio is installed: `pip list | findstr label-studio`
3. Check Python version compatibility
4. Review terminal error messages

### API Key Not Working

**Issue**: Cannot authenticate with Label Studio API

**Solutions**:
1. Verify `LABEL_STUDIO_API_KEY` in `.env` file
2. Get API key from Label Studio: Settings → Account & Settings → Access Token
3. Check API key has correct permissions
4. Ensure Label Studio server is running

### Import/Export Fails

**Issue**: Cannot import predictions or export annotations

**Solutions**:
1. Verify Label Studio server is running
2. Check `url` and `api_key` in config are correct
3. Verify project ID exists
4. Check network connectivity
5. Review Label Studio logs

## Related Documentation

- [Label Studio Configuration](../../configs/wildetect/extract-gps.md) (for XML config)
- [Detection Script](run_detection.md)
- [Census Script](run_census.md)
- [Label Studio Documentation](https://labelstud.io/guide/)

