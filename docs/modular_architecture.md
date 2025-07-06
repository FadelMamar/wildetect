# WildDetect Modular Architecture

## Overview

The WildDetect system has been reorganized into a well-defined, modular architecture that provides clear separation of concerns and enables flexible pipeline orchestration for wildlife detection campaigns.

## Architecture Components

### 1. Core Data Management

#### `CensusDataManager` (`src/wildetect/core/data/census.py`)
- **Purpose**: High-level data management for drone flight campaigns
- **Key Features**:
  - Campaign metadata management
  - Image ingestion and validation
  - DroneImage creation and tiling
  - Flight path analysis
  - Geographic merging of detections
  - Comprehensive statistics generation

#### `DetectionResults` (Data Structure)
- **Purpose**: Structured results from detection campaigns
- **Fields**:
  - `total_images`: Number of images processed
  - `total_detections`: Total detections found
  - `detection_by_class`: Detections grouped by species
  - `processing_time`: Time taken for detection
  - `detection_confidence_stats`: Confidence statistics
  - `geographic_coverage`: Geographic coverage information
  - `campaign_id`: Campaign identifier
  - `metadata`: Additional campaign metadata

### 2. Detection Pipeline

#### `DetectionPipeline` (`src/wildetect/core/detection_pipeline.py`)
- **Purpose**: End-to-end detection processing
- **Key Features**:
  - Model loading and initialization
  - Batch processing of images
  - Tile-based detection
  - Post-processing and result aggregation
  - Result saving and export

#### `ObjectDetectionSystem` (`src/wildetect/core/detectors/object_detection_system.py`)
- **Purpose**: Orchestrates detection models and processors
- **Key Features**:
  - Model management
  - ROI post-processing
  - Batch prediction
  - Result formatting

### 3. Flight Analysis

#### `FlightPathAnalyzer` (`src/wildetect/core/flight/flight_analyzer.py`)
- **Purpose**: Analyzes GPS flight paths and calculates efficiency metrics
- **Key Features**:
  - Flight path reconstruction from GPS data
  - Distance and area calculations
  - Coverage efficiency analysis
  - Overlap detection

#### `FlightEfficiency` (Data Structure)
- **Purpose**: Flight efficiency metrics
- **Fields**:
  - `total_distance_km`: Total flight distance
  - `total_area_covered_sqkm`: Area covered by images
  - `coverage_efficiency`: Area covered per distance flown
  - `overlap_percentage`: Percentage of overlapping images
  - `average_altitude_m`: Average flight altitude
  - `image_density_per_sqkm`: Images per square kilometer

### 4. Geographic Processing

#### `GeographicMerger` (`src/wildetect/core/flight/geographic_merger.py`)
- **Purpose**: Merges detections across overlapping geographic regions
- **Key Features**:
  - IoU-based duplicate detection
  - Geographic overlap analysis
  - Detection merging strategies
  - Quality metrics calculation

### 5. Visualization

#### `GeographicVisualizer` (`src/wildetect/core/visualization/geographic.py`)
- **Purpose**: Creates interactive geographic visualizations
- **Key Features**:
  - Folium-based map generation
  - Image footprint visualization
  - Detection overlay
  - Coverage statistics

#### `FiftyOneManager` (`src/wildetect/core/visualization/fiftyone_manager.py`)
- **Purpose**: FiftyOne integration for dataset management
- **Key Features**:
  - Dataset creation and management
  - Detection annotation export
  - Geographic data integration
  - Advanced visualization capabilities

### 6. Campaign Orchestration

#### `CampaignManager` (`src/wildetect/core/campaign_manager.py`)
- **Purpose**: High-level campaign orchestration
- **Key Features**:
  - Unified interface for complete campaigns
  - Component integration
  - Pipeline orchestration
  - Result aggregation and reporting

#### `CampaignConfig` (Configuration)
- **Purpose**: Campaign configuration structure
- **Fields**:
  - `campaign_id`: Campaign identifier
  - `loader_config`: Data loading configuration
  - `prediction_config`: Detection configuration
  - `metadata`: Campaign metadata
  - `visualization_config`: Visualization settings
  - `fiftyone_dataset_name`: FiftyOne dataset name

## Pipeline Flow

### 1. Data Ingestion
```python
# Add images to campaign
campaign_manager.add_images_from_paths(image_paths)
campaign_manager.add_images_from_directory(directory_path)
```

### 2. Data Preparation
```python
# Create DroneImage instances with tiling
campaign_manager.prepare_data(tile_size=640, overlap=0.2)
```

### 3. Detection Processing
```python
# Run detection on all images
detection_results = campaign_manager.run_detection(
    save_results=True,
    output_dir="./results"
)
```

### 4. Flight Analysis
```python
# Analyze flight path and efficiency
flight_path = campaign_manager.analyze_flight_path()
flight_efficiency = campaign_manager.calculate_flight_efficiency()
```

### 5. Geographic Merging
```python
# Merge detections across overlapping regions
merged_images = campaign_manager.merge_detections_geographically(
    iou_threshold=0.8
)
```

### 6. Visualization and Export
```python
# Create geographic visualization
visualization_path = campaign_manager.create_geographic_visualization(
    "./visualization.html"
)

# Export to FiftyOne
campaign_manager.export_to_fiftyone()

# Export detection report
campaign_manager.export_detection_report("./report.json")
```

## Complete Campaign Example

```python
from src.wildetect.core.campaign_manager import CampaignConfig, CampaignManager
from src.wildetect.core.config import LoaderConfig, PredictionConfig, FlightSpecs

# Create campaign configuration
config = CampaignConfig(
    campaign_id="wildlife_survey_2024",
    loader_config=LoaderConfig(
        tile_size=640,
        overlap=0.2,
        batch_size=4,
        flight_specs=FlightSpecs(
            sensor_height=24.0,
            focal_length=35.0,
            flight_height=180.0
        )
    ),
    prediction_config=PredictionConfig(
        model_path="/path/to/model.pt",
        model_type="yolo",
        confidence_threshold=0.25,
        device="cpu"
    ),
    metadata={
        "pilot_info": {"name": "John Doe"},
        "target_species": ["elephant", "giraffe", "zebra"]
    }
)

# Initialize campaign manager
campaign_manager = CampaignManager(config)

# Run complete campaign
results = campaign_manager.run_complete_campaign(
    image_paths=["/path/to/image1.jpg", "/path/to/image2.jpg"],
    output_dir="./campaign_results",
    run_flight_analysis=True,
    run_geographic_merging=True,
    create_visualization=True,
    export_to_fiftyone=True
)

# Access results
print(f"Detections: {results['detection_results'].total_detections}")
print(f"Flight distance: {results['flight_efficiency'].total_distance_km:.2f} km")
```

## CLI Integration

The command-line interface has been updated to use the new modular architecture:

```bash
# Run a complete census campaign
wildetect census wildlife_survey_2024 /path/to/images \
    --model /path/to/model.pt \
    --output ./results \
    --pilot "John Doe" \
    --species elephant giraffe zebra \
    --map
```

## Benefits of the Modular Architecture

### 1. **Separation of Concerns**
- Each component has a well-defined responsibility
- Easy to test individual components
- Clear interfaces between modules

### 2. **Flexibility**
- Components can be used independently
- Easy to swap implementations
- Configurable pipeline stages

### 3. **Extensibility**
- New components can be added easily
- Existing components can be enhanced
- Plugin architecture for custom functionality

### 4. **Maintainability**
- Clear code organization
- Comprehensive error handling
- Detailed logging and monitoring

### 5. **Reusability**
- Components can be reused across different campaigns
- Configuration-driven behavior
- Standardized interfaces

## Error Handling and Validation

The modular architecture includes comprehensive error handling:

- **Input Validation**: All inputs are validated before processing
- **Graceful Degradation**: Components handle missing data gracefully
- **Detailed Logging**: Comprehensive logging for debugging
- **Error Recovery**: Components can recover from partial failures

## Performance Considerations

- **Batch Processing**: Efficient batch processing of images
- **Memory Management**: Careful memory management for large datasets
- **Parallel Processing**: Support for parallel processing where applicable
- **Caching**: Intelligent caching of intermediate results

## Future Enhancements

The modular architecture enables future enhancements:

1. **Additional Detection Models**: Easy integration of new model types
2. **Advanced Visualization**: Enhanced geographic and statistical visualizations
3. **Real-time Processing**: Support for real-time detection pipelines
4. **Cloud Integration**: Cloud-based processing and storage
5. **API Interface**: RESTful API for remote campaign management

## Testing Strategy

Each component includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete pipeline testing
- **Performance Tests**: Performance benchmarking

This modular architecture provides a solid foundation for the WildDetect system, enabling efficient wildlife detection campaigns with comprehensive analysis and reporting capabilities. 