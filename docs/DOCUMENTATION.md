# WildDetect Reflex Frontend Documentation

## Overview

The WildDetect Reflex frontend provides a modern, responsive web interface for the WildDetect wildlife detection system. Built with [Reflex](https://reflex.dev/), it offers seamless integration with the existing FastAPI backend and provides real-time monitoring of detection jobs and census campaigns.

## Features

### 🎯 Core Functionality
- **File Upload**: Drag-and-drop interface for uploading images
- **Detection Jobs**: Start and monitor wildlife detection jobs
- **Census Campaigns**: Run comprehensive wildlife census campaigns
- **Real-time Updates**: Live job status and progress tracking
- **FiftyOne Integration**: Launch and manage FiftyOne datasets
- **System Monitoring**: View system information and dependencies

### 🎨 User Interface
- **Modern Design**: Clean, responsive interface built with Reflex components
- **Tab Navigation**: Organized sections for different functionalities
- **Progress Indicators**: Visual feedback for long-running operations
- **Error Handling**: Comprehensive error messages and status updates
- **Loading States**: Smooth loading animations and overlays

### 🔧 Technical Features
- **API Integration**: Seamless communication with FastAPI backend
- **State Management**: Reactive state management with Reflex
- **File Handling**: Secure file upload and processing
- **Background Polling**: Automatic job status updates
- **CORS Support**: Cross-origin resource sharing enabled

## Architecture

### File Structure
```
frontend/
├── app.py                 # Main Reflex application
├── state.py              # State management
├── pages/
│   └── index.py          # Main dashboard page
├── rxconfig.py           # Reflex configuration
├── requirements.txt       # Python dependencies
├── launch_frontend.py    # Launch script
├── example_usage.py      # Usage examples
└── README.md            # Setup instructions
```

### State Management
The frontend uses Reflex's built-in state management system:

```python
class WildDetectState(State):
    # API configuration
    api_base_url: str = "http://localhost:8000"
    
    # File upload state
    uploaded_files: List[str] = []
    upload_progress: int = 0
    upload_status: str = ""
    
    # Job management
    detection_jobs: Dict[str, Dict] = {}
    census_jobs: Dict[str, Dict] = {}
    
    # UI state
    loading: bool = False
    error_message: str = ""
```

### API Integration
The frontend communicates with the FastAPI backend through HTTP requests:

- **GET /info**: System information
- **POST /upload**: File upload
- **POST /detect**: Start detection job
- **POST /census**: Start census campaign
- **GET /jobs/{job_id}**: Job status
- **GET /fiftyone/launch**: Launch FiftyOne

## Setup and Installation

### Prerequisites
- Python 3.8+
- WildDetect API backend running on `http://localhost:8000`
- Reflex framework

### Installation

1. **Install Dependencies**:
```bash
cd src/wildetect/ui/frontend
pip install -r requirements.txt
```

2. **Initialize Reflex App**:
```bash
reflex init
```

3. **Launch the Frontend**:
```bash
python launch_frontend.py
```

### Alternative Launch Methods

**Using Reflex CLI**:
```bash
reflex run
```

**Using Python Script**:
```bash
python launch_frontend.py
```

## Usage Guide

### 1. Starting the Application

1. **Start the API Backend**:
```bash
# From the project root
python -m wildetect.api.main
```

2. **Start the Reflex Frontend**:
```bash
cd src/wildetect/ui/frontend
python launch_frontend.py
```

3. **Access the Interface**:
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

### 2. Uploading Images

1. Navigate to the "Upload Files" tab
2. Drag and drop images or click to select files
3. Configure detection parameters:
   - Model path (optional)
   - Confidence threshold
   - Device selection
4. Click "Start Detection"

### 3. Monitoring Jobs

1. Go to the "Detection Jobs" tab
2. View real-time job status and progress
3. Monitor job completion and results

### 4. Running Census Campaigns

1. Navigate to the "Census Campaigns" tab
2. Enter campaign details:
   - Campaign ID
   - Pilot name
   - Target species
3. Click "Start Census Campaign"

### 5. FiftyOne Integration

1. Go to the "FiftyOne" tab
2. Click "Launch FiftyOne" to open the dataset viewer
3. Use "Refresh Datasets" to update available datasets

## Configuration

### API Configuration
Edit `state.py` to change API settings:

```python
class WildDetectState(State):
    api_base_url: str = "http://localhost:8000"  # Change API URL
```

### Reflex Configuration
Modify `rxconfig.py` for app settings:

```python
class Config(rx.Config):
    app_name = "WildDetect"
    title = "WildDetect - Wildlife Detection System"
    frontend_port = 3000
    backend_port = 8000
```

### Environment Variables
Set environment variables for customization:

```bash
export WILDDETECT_API_URL="http://localhost:8000"
export WILDDETECT_FRONTEND_PORT=3000
```

## API Endpoints

### System Information
- **GET /info**: Get system information and dependencies

### File Operations
- **POST /upload**: Upload image files for processing

### Detection Jobs
- **POST /detect**: Start wildlife detection job
- **GET /jobs/{job_id}**: Get job status and progress

### Census Campaigns
- **POST /census**: Start wildlife census campaign

### Visualization
- **POST /visualize**: Create geographic visualizations
- **POST /analyze**: Analyze detection results

### FiftyOne Integration
- **GET /fiftyone/launch**: Launch FiftyOne app
- **GET /fiftyone/datasets/{dataset_name}**: Get dataset information
- **POST /fiftyone/export/{dataset_name}**: Export dataset

## Error Handling

### Common Issues

1. **API Connection Failed**:
   - Ensure the API backend is running on `http://localhost:8000`
   - Check firewall settings
   - Verify CORS configuration

2. **File Upload Errors**:
   - Check file format (images only)
   - Verify file size limits
   - Ensure proper permissions

3. **Job Failures**:
   - Check system resources (GPU/CPU)
   - Verify model file paths
   - Review job logs

### Debugging

1. **Enable Debug Mode**:
```python
# In rxconfig.py
debug = True
```

2. **Check Browser Console**:
   - Open developer tools (F12)
   - Monitor network requests
   - Check for JavaScript errors

3. **Review Logs**:
   - Frontend logs in terminal
   - API logs in backend console

## Development

### Adding New Features

1. **Create New State Methods**:
```python
class WildDetectState(State):
    async def new_feature(self):
        # Implementation
        pass
```

2. **Add UI Components**:
```python
def new_component() -> rx.Component:
    return rx.box(
        # Component content
    )
```

3. **Update Pages**:
```python
def index() -> rx.Component:
    return rx.vstack(
        # Existing components
        new_component(),
    )
```

### Testing

1. **Run Reflex Tests**:
```bash
reflex test
```

2. **Manual Testing**:
   - Test file upload functionality
   - Verify job monitoring
   - Check error handling

### Deployment

1. **Build for Production**:
```bash
reflex export
```

2. **Deploy to Cloud**:
```bash
reflex deploy
```

## Integration with WildDetect

### Backend Integration
The frontend integrates seamlessly with the existing WildDetect API:

- **Shared Models**: Uses the same Pydantic models as the API
- **Consistent Endpoints**: Matches API endpoint structure
- **Error Handling**: Consistent error responses
- **Job Management**: Real-time job status updates

### Data Flow
1. **Upload**: Images uploaded through frontend → API storage
2. **Processing**: API processes images → Frontend monitors progress
3. **Results**: API generates results → Frontend displays outcomes
4. **Visualization**: API creates visualizations → Frontend shows maps

## Performance Considerations

### Optimization Tips
1. **Image Compression**: Compress images before upload
2. **Batch Processing**: Use appropriate batch sizes
3. **Caching**: Implement result caching for repeated requests
4. **Polling**: Adjust polling frequency based on job duration

### Resource Usage
- **Memory**: Monitor memory usage during large uploads
- **CPU**: Balance between UI responsiveness and background processing
- **Network**: Optimize API request frequency

## Security

### Best Practices
1. **Input Validation**: Validate all user inputs
2. **File Security**: Scan uploaded files for malware
3. **API Security**: Use HTTPS in production
4. **Authentication**: Implement user authentication if needed

### CORS Configuration
```python
# In API backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Common Problems

1. **Frontend Won't Start**:
   - Check Reflex installation
   - Verify Python version (3.8+)
   - Clear Reflex cache: `reflex init --force`

2. **API Connection Issues**:
   - Verify API is running on correct port
   - Check network connectivity
   - Review CORS settings

3. **File Upload Problems**:
   - Check file permissions
   - Verify file formats
   - Monitor disk space

4. **Job Monitoring Issues**:
   - Check API job status endpoint
   - Verify polling frequency
   - Review job logs

### Getting Help

1. **Check Logs**: Review terminal output for errors
2. **API Documentation**: Visit http://localhost:8000/docs
3. **Reflex Documentation**: https://reflex.dev/docs
4. **GitHub Issues**: Report bugs and feature requests

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write tests for new features

## License

This frontend is part of the WildDetect project and follows the same license terms.

## References

- [Reflex Documentation](https://reflex.dev/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WildDetect API Documentation](http://localhost:8000/docs)
- [FiftyOne Documentation](https://voxel51.com/docs/fiftyone/) 