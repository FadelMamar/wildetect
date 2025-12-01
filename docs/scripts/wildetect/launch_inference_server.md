# Inference Server Launch Script

> **Location**: `scripts/launch_inference_server.bat`

**Purpose**: Launch a FastAPI inference server for remote detection. Allows other applications or services to send images for detection via HTTP API.

## Usage

```batch
scripts\launch_inference_server.bat
```

The script automatically:
1. Changes to the project root directory
2. Launches the inference server

## Command Executed

```batch
uv run wildetect services inference-server --port 4141 --workers 2
```

## Access

Once launched, the inference server will be available at:
- **API Base URL**: `http://localhost:4141`
- **API Documentation**: `http://localhost:4141/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:4141/redoc` (ReDoc)

## Features

The inference server provides:

- **REST API**: HTTP endpoints for detection
- **Single Image Detection**: Process individual images
- **Batch Detection**: Process multiple images
- **Health Check**: Server health and status endpoint
- **Interactive Docs**: Swagger UI for API exploration
- **Model Loading**: Loads models from MLflow or local paths

## Prerequisites

1. **Dependencies**: All Python dependencies installed
2. **Port Availability**: Port 4141 should be available
3. **Model Available**: Model should be accessible (MLflow or local path)

## API Endpoints

### Health Check

```http
GET /health
```

Returns server status and health information.

### Single Image Detection

```http
POST /predict
Content-Type: multipart/form-data

{
  "file": <image_file>,
  "confidence": 0.5  # optional
}
```

### Batch Detection

```http
POST /predict/batch
Content-Type: multipart/form-data

{
  "files": [<image_file1>, <image_file2>, ...],
  "confidence": 0.5  # optional
}
```

## Example Usage

### Python Client

```python
import requests

# Single image detection
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:4141/predict",
        files={"file": f},
        data={"confidence": 0.5}
    )

detections = response.json()
print(detections)
```

### cURL

```bash
curl -X POST "http://localhost:4141/predict" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

### Using in Detection Config

```yaml
# In config/detection.yaml
inference_service:
  url: "http://localhost:4141/predict"
  timeout: 60
```

## Stopping the Server

Press `Ctrl+C` in the terminal window to stop the inference server.

## Troubleshooting

### Port Already in Use

**Issue**: Port 4141 is already occupied

**Solutions**:
1. Close other inference server instances
2. Kill process using port 4141
3. Use different port: `--port 4142`

### Model Loading Fails

**Issue**: Server cannot load model

**Solutions**:
1. Verify model is available (MLflow or local path)
2. Check MLflow server is running (if using MLflow models)
3. Verify model path is correct
4. Check model file permissions

### API Requests Fail

**Issue**: Detection requests return errors

**Solutions**:
1. Verify server is running: `curl http://localhost:4141/health`
2. Check request format matches API specification
3. Verify image file is valid
4. Check server logs for error messages
5. Ensure image format is supported

### Slow Response Times

**Issue**: API responses are slow

**Solutions**:
1. Increase `--workers` count (more parallel workers)
2. Use GPU if available
3. Reduce image size before sending
4. Check server resources (CPU, memory, GPU)

### Connection Refused

**Issue**: Cannot connect to inference server

**Solutions**:
1. Verify server started successfully
2. Check server is listening on correct port
3. Try accessing `http://127.0.0.1:4141` instead
4. Check firewall settings
5. Review terminal for error messages

## Related Documentation

- [Detection Script](run_detection.md)
- [Inference Service Config](../../configs/wildetect/detection.md#inference_service)
- [CLI Reference](../../api-reference/wildetect-cli.md)

