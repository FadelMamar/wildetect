# WilData REST API Reference

FastAPI-based REST API for WilData operations.

## Getting Started

### Start API Server

```bash
cd wildata
scripts\launch_api.bat
```

Access:
- API: `http://localhost:8441`
- Docs: `http://localhost:8441/docs`
- Redoc: `http://localhost:8441/redoc`

## Endpoints

### Health Check

```bash
GET /api/v1/health
```

### Import Dataset

```bash
POST /api/v1/datasets/import
Content-Type: application/json

{
  "source_path": "/path/to/annotations.json",
  "source_format": "coco",
  "dataset_name": "my_dataset",
  "root": "data"
}
```

### List Datasets

```bash
GET /api/v1/datasets?root=data
```

### Create ROI Dataset

```bash
POST /api/v1/roi/create
Content-Type: application/json

{
  "source_path": "/path/to/data.json",
  "source_format": "coco",
  "dataset_name": "roi_dataset",
  "roi_config": {
    "roi_box_size": 128,
    "random_roi_count": 10
  }
}
```

### Job Status

```bash
GET /api/v1/jobs/{job_id}
```

## Python Client Example

```python
import requests

# Import dataset
response = requests.post(
    "http://localhost:8441/api/v1/datasets/import",
    json={
        "source_path": "annotations.json",
        "source_format": "coco",
        "dataset_name": "my_dataset"
    }
)

job_id = response.json()["job_id"]

# Check job status
status_response = requests.get(
    f"http://localhost:8441/api/v1/jobs/{job_id}"
)

print(status_response.json())
```

---

For complete API documentation, see:
- [WilData API Documentation](../../../wildata/docs/API_README.md)
- Interactive docs at `/docs` endpoint

