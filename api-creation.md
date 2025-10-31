# WildDetect API Alignment Plan

## Executive Summary

The current WildDetect API has several critical issues that prevent it from being production-ready and consistent with the CLI:

### Key Issues Identified
1. **Type Safety Violations**: `pipeline_type` expects `Literal["single", "multi"]` but API uses `str`
2. **Inconsistent Data Models**: API models don't extend CLI's Pydantic models
3. **Hardcoded Configuration**: No integration with existing config loading system
4. **Missing Validation**: Limited validation compared to CLI's comprehensive system
5. **Error Handling**: Inconsistent error responses and missing proper exception handling
6. **Response Models**: Inconsistent response structures across endpoints

### Solution Overview
This plan creates a unified API that:
- **Extends CLI Models**: API models inherit from existing CLI Pydantic models
- **Uses Shared Configuration**: Leverages existing config loading and validation system
- **Provides Type Safety**: Full Pydantic validation throughout
- **Standardizes Responses**: Consistent response models and error handling
- **Maintains Backward Compatibility**: Gradual migration path

## Overview

This document outlines a comprehensive plan to align the current FastAPI implementation with the existing CLI data models and configuration system. The goal is to create a consistent, type-safe, and maintainable API that leverages the existing Pydantic models and configuration infrastructure.

## Current State Analysis

### Existing CLI Structure
- **Configuration Models**: Well-defined Pydantic models in `src/wildetect/core/config_models.py`
  - `DetectConfigModel` - Complete detection configuration
  - `CensusConfigModel` - Complete census campaign configuration  
  - `VisualizeConfigModel` - Complete visualization configuration
  - Supporting models: `FlightSpecsModel`, `ModelConfigModel`, `ProcessingConfigModel`, etc.

- **CLI Commands**: Structured commands in `src/wildetect/cli/commands/`
  - `core_commands.py` - `detect`, `census`, `analyze` commands
  - `visualization_commands.py` - `visualize`, `visualize_geographic_bounds` commands
  - `service_commands.py` - Service management commands
  - `utility_commands.py` - Utility commands

- **Configuration Loading**: Robust config loading system in `src/wildetect/core/config_loader.py`
  - YAML file loading with fallbacks
  - Environment variable substitution
  - CLI override merging
  - Pydantic validation

### Current API Issues
1. **Inconsistent Data Models**: API uses custom Pydantic models that don't align with CLI models
2. **Missing Validation**: Limited validation compared to CLI's comprehensive Pydantic models
3. **Hardcoded Defaults**: Defaults scattered throughout API code instead of using centralized config
4. **Type Safety Issues**: Several linter errors indicating type mismatches
5. **Configuration Management**: No integration with existing config loading system
6. **Response Models**: Inconsistent response structures

## Alignment Plan

### Phase 1: Model Alignment and Consolidation

#### 1.1 Create Shared API Models
**File**: `src/wildetect/api/models.py`

```python
# New shared models that extend CLI models
from ..core.config_models import (
    DetectConfigModel, 
    CensusConfigModel, 
    VisualizeConfigModel,
    FlightSpecsModel,
    ModelConfigModel,
    ProcessingConfigModel
)

class DetectionRequest(DetectConfigModel):
    """API request model for detection endpoint."""
    images: Optional[List[str]] = Field(default=None, description="Image paths")
    
class CensusRequest(CensusConfigModel):
    """API request model for census endpoint."""
    pass

class VisualizationRequest(VisualizeConfigModel):
    """API request model for visualization endpoint."""
    results_path: str = Field(description="Path to results file")

class AnalysisRequest(BaseModel):
    """API request model for analysis endpoint."""
    results_path: str = Field(description="Path to results file")
    output_dir: Optional[str] = Field(default="analysis", description="Output directory")
    create_map: bool = Field(default=True, description="Create geographic map")

# Response models
class JobResponse(BaseModel):
    """Base response model for background jobs."""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status")
    message: str = Field(description="Status message")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")

class DetectionResponse(JobResponse):
    """Response model for detection jobs."""
    results_path: Optional[str] = Field(default=None, description="Path to results")
    total_images: Optional[int] = Field(default=None, description="Total images processed")
    total_detections: Optional[int] = Field(default=None, description="Total detections")
    species_counts: Optional[Dict[str, int]] = Field(default=None, description="Species counts")

class CensusResponse(JobResponse):
    """Response model for census jobs."""
    campaign_id: str = Field(description="Campaign identifier")
    statistics: Optional[Dict] = Field(default=None, description="Campaign statistics")
```

#### 1.2 Create API Configuration Loader
**File**: `src/wildetect/api/config.py`

```python
from ..core.config_loader import ConfigLoader, load_config_with_pydantic
from ..core.config_models import DetectConfigModel, CensusConfigModel, VisualizeConfigModel

class APIConfigLoader:
    """Configuration loader specifically for API endpoints."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
    
    def load_detection_config(
        self, 
        request_data: dict, 
        config_path: Optional[str] = None
    ) -> DetectConfigModel:
        """Load detection configuration with API request overrides."""
        # Load base config
        base_config = self.config_loader.load_config_with_pydantic(
            "detect", config_path
        )
        
        # Apply API request overrides
        for key, value in request_data.items():
            if hasattr(base_config, key) and value is not None:
                setattr(base_config, key, value)
        
        return base_config
    
    def load_census_config(
        self, 
        request_data: dict, 
        config_path: Optional[str] = None
    ) -> CensusConfigModel:
        """Load census configuration with API request overrides."""
        base_config = self.config_loader.load_config_with_pydantic(
            "census", config_path
        )
        
        # Apply overrides
        for key, value in request_data.items():
            if hasattr(base_config, key) and value is not None:
                setattr(base_config, key, value)
        
        return base_config
```

### Phase 2: API Endpoint Refactoring

#### 2.1 Refactor Detection Endpoint
**File**: `src/wildetect/api/main.py`

```python
from .models import DetectionRequest, DetectionResponse
from .config import APIConfigLoader

@app.post("/detect", response_model=DetectionResponse)
async def detect_wildlife(
    request: DetectionRequest, 
    background_tasks: BackgroundTasks
):
    """Run wildlife detection on images."""
    job_id = str(uuid4())
    
    # Initialize job status
    job_status[job_id] = {
        "status": "running",
        "message": "Starting detection...",
        "progress": 0,
    }
    
    def run_detection():
        try:
            # Load configuration using shared loader
            config_loader = APIConfigLoader()
            config = config_loader.load_detection_config(request.dict())
            
            # Convert to internal config objects
            pred_config = config.to_prediction_config(verbose=True)
            loader_config = config.to_loader_config()
            
            # Create detection pipeline
            if pred_config.pipeline_type == "multi":
                pipeline = MultiThreadedDetectionPipeline(
                    config=pred_config,
                    loader_config=loader_config,
                    queue_size=pred_config.queue_size,
                )
            else:
                pipeline = DetectionPipeline(
                    config=pred_config,
                    loader_config=loader_config,
                )
            
            # Run detection
            drone_images = pipeline.run_detection(
                image_paths=request.images or [],
                image_dir=None,
                save_path=config.output.directory,
            )
            
            # Calculate statistics
            stats = get_detection_statistics(drone_images)
            
            # Update job status
            job_status[job_id].update({
                "status": "completed",
                "message": "Detection completed successfully",
                "progress": 100,
                "results": {
                    "results_path": config.output.directory,
                    "total_images": len(drone_images),
                    "total_detections": stats["total_detections"],
                    "species_counts": stats["species_counts"],
                }
            })
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            job_status[job_id].update({
                "status": "failed",
                "message": f"Detection failed: {str(e)}",
                "progress": 0,
            })
    
    background_tasks.add_task(run_detection)
    
    return DetectionResponse(
        job_id=job_id,
        status="running",
        message="Detection job started",
    )
```

#### 2.2 Refactor Census Endpoint
**File**: `src/wildetect/api/main.py`

```python
@app.post("/census", response_model=CensusResponse)
async def run_census_campaign(
    request: CensusRequest, 
    background_tasks: BackgroundTasks
):
    """Run wildlife census campaign."""
    job_id = str(uuid4())
    
    job_status[job_id] = {
        "status": "running",
        "message": "Starting census campaign...",
        "progress": 0,
    }
    
    def run_census():
        try:
            # Load configuration using shared loader
            config_loader = APIConfigLoader()
            config = config_loader.load_census_config(request.dict())
            
            # Create campaign configuration
            campaign_config = CampaignConfig(
                campaign_id=config.campaign.id,
                loader_config=config.detection.to_loader_config(),
                prediction_config=config.detection.to_prediction_config(verbose=True),
                metadata={
                    "pilot_info": {"name": config.campaign.pilot_name},
                    "target_species": config.campaign.target_species,
                },
                fiftyone_dataset_name=f"campaign_{config.campaign.id}",
            )
            
            # Initialize campaign manager
            campaign_manager = CampaignManager(campaign_config)
            
            # Run campaign
            results = campaign_manager.run_complete_campaign(
                image_paths=request.images or [],
                output_dir=config.export.output_directory,
                export_to_fiftyone=config.export.to_fiftyone,
            )
            
            job_status[job_id].update({
                "status": "completed",
                "message": "Census campaign completed successfully",
                "progress": 100,
                "results": {
                    "campaign_id": config.campaign.id,
                    "statistics": results["statistics"],
                    "output_dir": config.export.output_directory,
                }
            })
            
        except Exception as e:
            logger.error(f"Census campaign failed: {e}")
            job_status[job_id].update({
                "status": "failed",
                "message": f"Census campaign failed: {str(e)}",
                "progress": 0,
            })
    
    background_tasks.add_task(run_census)
    
    return CensusResponse(
        job_id=job_id,
        status="running",
        message="Census campaign started",
        campaign_id=request.campaign.id,
    )
```

### Phase 3: Enhanced Error Handling and Validation

#### 3.1 Create API Exception Handlers
**File**: `src/wildetect/api/exceptions.py`

```python
from fastapi import HTTPException
from typing import Any, Dict

class WildDetectAPIException(HTTPException):
    """Base exception for WildDetect API."""
    
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

class ConfigurationError(WildDetectAPIException):
    """Raised when configuration is invalid."""
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail, error_code="CONFIG_ERROR")

class ValidationError(WildDetectAPIException):
    """Raised when request validation fails."""
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail, error_code="VALIDATION_ERROR")

class JobNotFoundError(WildDetectAPIException):
    """Raised when job is not found."""
    def __init__(self, job_id: str):
        super().__init__(
            status_code=404, 
            detail=f"Job {job_id} not found", 
            error_code="JOB_NOT_FOUND"
        )
```

#### 3.2 Add Request Validation
**File**: `src/wildetect/api/validators.py`

```python
from typing import List, Optional
from pydantic import field_validator
from .models import DetectionRequest, CensusRequest

class APIValidators:
    """Validators for API requests."""
    
    @staticmethod
    def validate_image_paths(paths: Optional[List[str]]) -> List[str]:
        """Validate image paths exist."""
        if not paths:
            return []
        
        valid_paths = []
        for path in paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                raise ValueError(f"Image path does not exist: {path}")
        
        return valid_paths
    
    @staticmethod
    def validate_confidence_threshold(value: float) -> float:
        """Validate confidence threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return value
```

### Phase 4: Configuration Management Integration

#### 4.1 Create API Configuration Service
**File**: `src/wildetect/api/services.py`

```python
from typing import Dict, Any, Optional
from pathlib import Path
from ..core.config_loader import ConfigLoader
from ..core.config_models import DetectConfigModel, CensusConfigModel, VisualizeConfigModel

class ConfigurationService:
    """Service for managing API configurations."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config_cache: Dict[str, Any] = {}
    
    def get_detection_config(
        self, 
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> DetectConfigModel:
        """Get detection configuration with caching."""
        cache_key = f"detect_{config_path}_{hash(str(overrides))}"
        
        if cache_key not in self.config_cache:
            config = self.config_loader.load_config_with_pydantic(
                "detect", config_path, overrides
            )
            self.config_cache[cache_key] = config
        
        return self.config_cache[cache_key]
    
    def get_census_config(
        self, 
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> CensusConfigModel:
        """Get census configuration with caching."""
        cache_key = f"census_{config_path}_{hash(str(overrides))}"
        
        if cache_key not in self.config_cache:
            config = self.config_loader.load_config_with_pydantic(
                "census", config_path, overrides
            )
            self.config_cache[cache_key] = config
        
        return self.config_cache[cache_key]
    
    def clear_cache(self):
        """Clear configuration cache."""
        self.config_cache.clear()
```

### Phase 5: Response Standardization

#### 5.1 Create Standard Response Models
**File**: `src/wildetect/api/responses.py`

```python
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class StandardResponse(BaseModel):
    """Standard API response model."""
    success: bool = Field(description="Whether the request was successful")
    message: str = Field(description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    data: Optional[Any] = Field(default=None, description="Response data")

class PaginatedResponse(StandardResponse):
    """Paginated response model."""
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Page size")
    total_pages: int = Field(description="Total number of pages")
    total_items: int = Field(description="Total number of items")

class ErrorResponse(StandardResponse):
    """Error response model."""
    error_code: str = Field(description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
```

### Phase 6: Testing and Documentation

#### 6.1 Create API Tests
**File**: `tests/test_api_alignment.py`

```python
import pytest
from fastapi.testclient import TestClient
from wildetect.api.main import app
from wildetect.core.config_models import DetectConfigModel, CensusConfigModel

client = TestClient(app)

def test_detection_request_alignment():
    """Test that detection request aligns with CLI model."""
    request_data = {
        "model_path": "models/yolo.pt",
        "confidence": 0.3,
        "batch_size": 16,
        "tile_size": 800,
    }
    
    response = client.post("/detect", json=request_data)
    assert response.status_code == 200
    
    # Verify response structure
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert data["status"] == "running"

def test_census_request_alignment():
    """Test that census request aligns with CLI model."""
    request_data = {
        "campaign_id": "test_campaign",
        "pilot_name": "Test Pilot",
        "target_species": ["deer", "elk"],
    }
    
    response = client.post("/census", json=request_data)
    assert response.status_code == 200
```

#### 6.2 Update API Documentation
**File**: `docs/api_documentation.md`

```markdown
# WildDetect API Documentation

## Overview

The WildDetect API provides REST endpoints for wildlife detection, census campaigns, and data analysis. The API is built on FastAPI and uses the same configuration models as the CLI for consistency.

## Configuration

The API uses the same configuration system as the CLI:
- YAML configuration files
- Environment variable substitution
- Pydantic validation
- Default value management

## Endpoints

### Detection

**POST** `/detect`

Start a wildlife detection job.

**Request Body:**
```json
{
  "model_path": "models/yolo.pt",
  "confidence": 0.3,
  "batch_size": 16,
  "tile_size": 800,
  "images": ["path/to/image1.jpg", "path/to/image2.jpg"]
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "running",
  "message": "Detection job started",
  "progress": 0
}
```

### Census

**POST** `/census`

Start a census campaign.

**Request Body:**
```json
{
  "campaign_id": "campaign_001",
  "pilot_name": "John Doe",
  "target_species": ["deer", "elk"],
  "model_path": "models/yolo.pt"
}
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "message": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```
```

## Implementation Steps

### Step 1: Create New API Models
1. Create `src/wildetect/api/models.py` with shared models
2. Create `src/wildetect/api/config.py` with API config loader
3. Create `src/wildetect/api/exceptions.py` with custom exceptions
4. Create `src/wildetect/api/validators.py` with validation logic

### Step 2: Refactor Existing Endpoints
1. Update `src/wildetect/api/main.py` to use new models
2. Integrate configuration loading system
3. Add proper error handling
4. Standardize response formats

### Step 3: Add Configuration Management
1. Create `src/wildetect/api/services.py` with config service
2. Integrate with existing config loader
3. Add caching for performance
4. Add configuration validation

### Step 4: Testing and Documentation
1. Create comprehensive tests
2. Update API documentation
3. Add examples and usage guides
4. Validate alignment with CLI

### Step 5: Migration and Cleanup
1. Deprecate old API models
2. Update existing clients
3. Clean up unused code
4. Update documentation

## Benefits

1. **Consistency**: API and CLI use the same data models and validation
2. **Maintainability**: Single source of truth for configuration
3. **Type Safety**: Full Pydantic validation throughout
4. **Developer Experience**: Consistent API design and documentation
5. **Testing**: Shared test utilities and validation
6. **Performance**: Configuration caching and optimization

## Timeline

- **Week 1**: Model creation and configuration integration
- **Week 2**: Endpoint refactoring and error handling
- **Week 3**: Testing and documentation
- **Week 4**: Migration and cleanup

This plan ensures a smooth transition to a more robust, maintainable, and consistent API that fully leverages the existing CLI infrastructure.
