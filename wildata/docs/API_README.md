# WildData API

A FastAPI wrapper around the WildData CLI functionality, providing RESTful endpoints for dataset management operations.

## Features

- **Dataset Management**: Import, export, and list datasets
- **ROI Dataset Creation**: Create Region of Interest datasets for training
- **GPS Data Updates**: Update EXIF GPS data from CSV files
- **Visualization**: Generate dataset visualizations for classification and detection
- **Background Jobs**: Long-running operations run asynchronously with job tracking
- **Health Monitoring**: Health checks and system metrics
- **OpenAPI Documentation**: Auto-generated API documentation
- **CORS Support**: Cross-origin resource sharing enabled
- **Error Handling**: Comprehensive error responses with proper HTTP status codes

## Quick Start

### Installation

The API dependencies are included in the main package. Install the project:

```bash
pip install -e .
```

### Starting the API Server

#### Option 1: Direct Python module
```bash
python -m wildata.api.main
```

#### Option 2: Using the CLI
```bash
wildata api serve
```

#### Option 3: Using the batch script (Windows)
```bash
launch_api.bat
```

### Accessing the API
- **port** (default): 8441 
- **API Documentation**: http://localhost:$port/docs
- **Alternative Docs**: http://localhost:$port/redoc
- **Health Check**: http://localhost:$port/api/v1/health
- **API Info**: http://localhost:$port/api

## API Endpoints

### Health and Monitoring

#### `GET /api/v1/health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "service": "wildata-api"
}
```

#### `GET /api/v1/health/detailed`
Detailed health check with system information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "service": "wildata-api",
  "version": "0.1.0",
  "config": {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false,
    "job_queue_size": 100
  }
}
```

### Dataset Management

#### `POST /api/v1/datasets/import`
Import a single dataset.

**Request Body:**
```json
{
  "source_path": "/path/to/dataset",
  "source_format": "coco",
  "dataset_name": "my_dataset",
  "root": "data",
  "split_name": "train",
  "processing_mode": "batch",
  "track_with_dvc": false,
  "bbox_tolerance": 5,
  "enable_dvc": false,
  "roi_config": null,
  "disable_roi": false,
  "transformations": null,
  "dotenv_path": null,
  "ls_xml_config": null,
  "ls_parse_config": false
}
```

**Response:**
```json
{
  "success": true,
  "dataset_name": "my_dataset",
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "message": "Dataset import started in background"
}
```

#### `POST /api/v1/datasets/import/bulk`
Bulk import multiple datasets.

**Request Body:**
```json
{
  "source_paths": ["/path/to/dataset1", "/path/to/dataset2"],
  "source_format": "coco",
  "root": "data",
  "split_name": "train",
  "processing_mode": "batch",
  "track_with_dvc": false,
  "bbox_tolerance": 5,
  "enable_dvc": false,
  "roi_config": null,
  "disable_roi": false,
  "transformations": null,
  "ls_xml_config": null,
  "ls_parse_config": false
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "total_datasets": 2,
  "message": "Bulk import started in background"
}
```

#### `GET /api/v1/datasets`
List all datasets.

**Query Parameters:**
- `root` (string, optional): Root directory for datasets (default: "data")

**Response:**
```json
{
  "datasets": [
    {
      "dataset_name": "my_dataset",
      "total_images": 1000,
      "total_annotations": 5000,
      "splits": ["train", "val"],
      "created_at": "2024-01-01T12:00:00",
      "last_modified": "2024-01-01T12:00:00"
    }
  ],
  "total_count": 1,
  "root_directory": "data"
}
```

#### `GET /api/v1/datasets/{dataset_name}`
Get dataset information.

**Path Parameters:**
- `dataset_name` (string): Name of the dataset

**Query Parameters:**
- `root` (string, optional): Root directory for datasets (default: "data")

**Response:**
```json
{
  "dataset_name": "my_dataset",
  "total_images": 1000,
  "total_annotations": 5000,
  "splits": ["train", "val"],
  "created_at": "2024-01-01T12:00:00",
  "last_modified": "2024-01-01T12:00:00"
}
```

#### `DELETE /api/v1/datasets/{dataset_name}`
Delete a dataset.

**Path Parameters:**
- `dataset_name` (string): Name of the dataset

**Query Parameters:**
- `root` (string, optional): Root directory for datasets (default: "data")

**Response:**
```json
{
  "message": "Dataset 'my_dataset' deleted successfully"
}
```

### ROI Dataset Management

#### `POST /api/v1/roi/create`
Create a single ROI dataset.

**Request Body:**
```json
{
  "source_path": "/path/to/dataset",
  "source_format": "coco",
  "dataset_name": "my_roi_dataset",
  "root": "data",
  "split_name": "val",
  "bbox_tolerance": 5,
  "roi_config": {
    "random_roi_count": 10,
    "roi_box_size": 128,
    "min_roi_size": 32,
    "background_class": "background",
    "save_format": "jpg"
  },
  "ls_xml_config": null,
  "ls_parse_config": false,
  "draw_original_bboxes": false
}
```

**Response:**
```json
{
  "success": true,
  "dataset_name": "my_roi_dataset",
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "message": "ROI dataset creation started in background"
}
```

#### `POST /api/v1/roi/create/bulk`
Bulk create multiple ROI datasets.

**Request Body:**
```json
{
  "source_paths": ["/path/to/dataset1", "/path/to/dataset2"],
  "source_format": "coco",
  "root": "data",
  "split_name": "val",
  "bbox_tolerance": 5,
  "roi_config": {
    "random_roi_count": 10,
    "roi_box_size": 128,
    "min_roi_size": 32,
    "background_class": "background",
    "save_format": "jpg"
  },
  "ls_xml_config": null,
  "ls_parse_config": false,
  "draw_original_bboxes": false
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "total_datasets": 2,
  "message": "Bulk ROI creation started in background"
}
```

#### `GET /api/v1/roi/{dataset_name}`
Get ROI dataset information.

**Path Parameters:**
- `dataset_name` (string): Name of the ROI dataset

**Query Parameters:**
- `root` (string, optional): Root directory for datasets (default: "data")

**Response:**
```json
{
  "dataset_name": "my_roi_dataset",
  "total_images": 500,
  "total_annotations": 2500,
  "splits": ["val"],
  "created_at": "2024-01-01T12:00:00",
  "last_modified": "2024-01-01T12:00:00"
}
```

### GPS Data Updates

#### `POST /api/v1/gps/update`
Update GPS data from CSV file.

**Request Body:**
```json
{
  "image_folder": "/path/to/images",
  "csv_path": "/path/to/gps_data.csv",
  "output_dir": "/path/to/output",
  "skip_rows": 0,
  "filename_col": "filename",
  "lat_col": "latitude",
  "lon_col": "longitude",
  "alt_col": "altitude"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "message": "GPS update started in background",
  "updated_images_count": 0,
  "output_dir": "/path/to/output"
}
```

### Visualization

#### `POST /api/v1/visualize/classification`
Visualize a classification dataset.

**Request Body:**
```json
{
  "dataset_name": "my_dataset",
  "root_data_directory": "data",
  "split": "train",
  "load_as_single_class": false,
  "background_class_name": "background",
  "single_class_name": "wildlife",
  "keep_classes": null,
  "discard_classes": null
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "message": "Classification visualization started in background",
  "dataset_name": "my_dataset",
  "split": "train"
}
```

#### `POST /api/v1/visualize/detection`
Visualize a detection dataset.

**Request Body:**
```json
{
  "dataset_name": "my_dataset",
  "root_data_directory": "data",
  "split": "train",
  "load_as_single_class": false,
  "background_class_name": "background",
  "single_class_name": "wildlife",
  "keep_classes": null,
  "discard_classes": null
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "message": "Detection visualization started in background",
  "dataset_name": "my_dataset",
  "split": "train"
}
```

### Job Management

#### `GET /api/v1/jobs/{job_id}`
Get job status.

**Path Parameters:**
- `job_id` (string): Job identifier

**Response:**
```json
{
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "status": "running",
  "progress": 45.5,
  "created_at": "2024-01-01T12:00:00",
  "started_at": "2024-01-01T12:00:05",
  "completed_at": null,
  "result": null,
  "job_type": "import_dataset"
}
```

#### `GET /api/v1/jobs`
List all jobs with optional filtering.

**Query Parameters:**
- `status_filter` (string, optional): Filter by job status (pending/running/completed/failed/cancelled)
- `limit` (integer, optional): Maximum number of jobs to return (default: 50)
- `offset` (integer, optional): Number of jobs to skip (default: 0)

**Response:**
```json
[
  {
    "job_id": "12345678-1234-1234-1234-123456789abc",
    "status": "running",
    "progress": 45.5,
    "created_at": "2024-01-01T12:00:00",
    "started_at": "2024-01-01T12:00:05",
    "completed_at": null,
    "result": null,
    "job_type": "import_dataset"
  }
]
```

#### `DELETE /api/v1/jobs/{job_id}`
Cancel a job.

**Path Parameters:**
- `job_id` (string): Job identifier

**Response:**
```json
{
  "message": "Job 12345678-1234-1234-1234-123456789abc cancelled successfully"
}
```

## Configuration

The API can be configured using environment variables with the `WILDATA_API_` prefix:

### Server Settings
- `WILDATA_API_HOST` - Server host (default: 0.0.0.0)
- `WILDATA_API_PORT` - Server port (default: 8000)
- `WILDATA_API_DEBUG` - Enable debug mode (default: false)

### CORS Settings
- `WILDATA_API_CORS_ORIGINS` - Allowed CORS origins (comma-separated)
- `WILDATA_API_CORS_ALLOW_CREDENTIALS` - Allow CORS credentials (default: true)

### Background Job Settings
- `WILDATA_API_JOB_QUEUE_SIZE` - Background job queue size (default: 100)
- `WILDATA_API_JOB_TIMEOUT` - Job timeout in seconds (default: 3600)

### Security Settings
- `WILDATA_API_SECRET_KEY` - Secret key for JWT (default: "your-secret-key-change-in-production")
- `WILDATA_API_ALGORITHM` - JWT algorithm (default: "HS256")
- `WILDATA_API_ACCESS_TOKEN_EXPIRE_MINUTES` - Access token expiry in minutes (default: 30)

### Database Settings
- `WILDATA_API_DATABASE_URL` - Database URL for job persistence (optional)

## Background Jobs

Long-running operations like dataset imports are handled as background jobs:

1. **Job Creation**: When you submit a request, a job is created with status "pending"
2. **Job Execution**: The job runs in the background with status "running"
3. **Progress Tracking**: Job progress is updated during execution
4. **Job Completion**: Job status becomes "completed" or "failed" with results

### Job Statuses

- `pending` - Job is queued but not started
- `running` - Job is currently executing
- `completed` - Job finished successfully
- `failed` - Job failed with an error
- `cancelled` - Job was cancelled by user

### Job Types

- `import_dataset` - Single dataset import
- `bulk_import` - Bulk dataset import
- `create_roi` - Single ROI dataset creation
- `bulk_create_roi` - Bulk ROI dataset creation
- `update_gps` - GPS data update
- `visualize_classification` - Classification dataset visualization
- `visualize_detection` - Detection dataset visualization

## Error Handling

The API returns structured error responses:

```json
{
  "message": "Dataset not found",
  "error_code": "NOT_FOUND",
  "details": {
    "resource_type": "dataset"
  }
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `404` - Not Found
- `409` - Conflict
- `500` - Internal Server Error

## Request Models

### ImportDatasetRequest
- `source_path` (string, required): Path to source dataset
- `source_format` (string, required): Source format (coco/yolo/ls)
- `dataset_name` (string, required): Name for the dataset
- `root` (string, optional): Root directory for data storage (default: "data")
- `split_name` (string, optional): Split name (train/val/test) (default: "train")
- `processing_mode` (string, optional): Processing mode (streaming/batch) (default: "batch")
- `track_with_dvc` (boolean, optional): Track dataset with DVC (default: false)
- `bbox_tolerance` (integer, optional): Bbox validation tolerance (default: 5)
- `enable_dvc` (boolean, optional): Enable DVC integration (default: false)
- `roi_config` (object, optional): ROI configuration
- `disable_roi` (boolean, optional): Disable ROI extraction (default: false)
- `transformations` (object, optional): Transformation pipeline config
- `dotenv_path` (string, optional): Path to .env file
- `ls_xml_config` (string, optional): Label Studio XML config path
- `ls_parse_config` (boolean, optional): Parse Label Studio config (default: false)

### CreateROIRequest
- `source_path` (string, required): Path to source dataset
- `source_format` (string, required): Source format (coco/yolo)
- `dataset_name` (string, required): Name for the dataset
- `root` (string, optional): Root directory for data storage (default: "data")
- `split_name` (string, optional): Split name (train/val/test) (default: "val")
- `bbox_tolerance` (integer, optional): Bbox validation tolerance (default: 5)
- `roi_config` (object, required): ROI configuration
- `ls_xml_config` (string, optional): Label Studio XML config path
- `ls_parse_config` (boolean, optional): Parse Label Studio config (default: false)
- `draw_original_bboxes` (boolean, optional): Draw original bounding boxes (default: false)

### UpdateGPSRequest
- `image_folder` (string, required): Path to folder containing images
- `csv_path` (string, required): Path to CSV file with GPS coordinates
- `output_dir` (string, required): Output directory for updated images
- `skip_rows` (integer, optional): Number of rows to skip in CSV (default: 0)
- `filename_col` (string, optional): CSV column name for filenames (default: "filename")
- `lat_col` (string, optional): CSV column name for latitude (default: "latitude")
- `lon_col` (string, optional): CSV column name for longitude (default: "longitude")
- `alt_col` (string, optional): CSV column name for altitude (default: "altitude")

### VisualizeRequest
- `dataset_name` (string, required): Name for the FiftyOne dataset
- `root_data_directory` (string, optional): Root data directory
- `split` (string, optional): Dataset split (train/val/test) (default: "train")
- `load_as_single_class` (boolean, optional): Load as single class (default: false)
- `background_class_name` (string, optional): Background class name (default: "background")
- `single_class_name` (string, optional): Single class name (default: "wildlife")
- `keep_classes` (array, optional): Classes to keep
- `discard_classes` (array, optional): Classes to discard

## Development

### Running Tests

```bash
pytest tests/api/ -v
```

### API Configuration Check

```bash
wildata api check
```

### Development Mode

```bash
wildata api serve --reload
```

## Architecture

The API is built with a modular architecture:

- **Models**: Pydantic models for requests/responses
- **Routers**: FastAPI routers for endpoint organization
- **Services**: Background job queue and task handlers
- **Dependencies**: Shared utilities and authentication
- **Exceptions**: Custom exception handling

## Future Enhancements

- [ ] File upload endpoints
- [ ] Authentication system
- [ ] Advanced monitoring and logging
- [ ] Production deployment configuration
- [ ] Rate limiting
- [ ] Database persistence for jobs
- [ ] WebSocket support for real-time job updates

## Contributing

The API follows the same development practices as the main WildData project. See the main README for contribution guidelines. 