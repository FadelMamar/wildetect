# WildDetect Phase 2 Features

This document describes the advanced features implemented in Phase 2 of the WildDetect project, focusing on flight path analysis, geographic merging, and performance optimizations.

## Overview

Phase 2 introduces sophisticated analysis capabilities for drone image campaigns, including:

- **Flight Path Analysis**: GPS-based flight path reconstruction and efficiency metrics
- **Geographic Merging**: Intelligent merging of detections across overlapping regions
- **Performance Optimizations**: Caching, parallel processing, and memory management
- **Enhanced Metadata**: Comprehensive campaign statistics and reporting

## Flight Path Analysis

### FlightPathAnalyzer

The `FlightPathAnalyzer` class provides comprehensive flight path analysis capabilities:

```python
from src.wildetect.core.data import FlightPathAnalyzer

# Initialize analyzer
analyzer = FlightPathAnalyzer()

# Analyze flight path from drone images
flight_path = analyzer.analyze_flight_path(drone_images)

# Calculate flight efficiency metrics
efficiency = analyzer.calculate_flight_efficiency(flight_path, drone_images)

# Detect overlapping regions
overlapping_regions = analyzer.detect_overlapping_regions(drone_images, overlap_threshold=0.1)
```

### Key Features

- **GPS Coordinate Extraction**: Automatic extraction and validation of GPS coordinates
- **Flight Path Reconstruction**: Sequential ordering of waypoints based on timestamps
- **Distance Calculations**: Haversine formula for accurate geographic distance calculations
- **Efficiency Metrics**: Coverage efficiency, overlap percentage, image density
- **Altitude Analysis**: Average, minimum, and maximum altitude tracking

### Flight Efficiency Metrics

The `FlightEfficiency` dataclass provides comprehensive metrics:

- `total_distance_km`: Total flight distance in kilometers
- `total_area_covered_sqkm`: Total area covered by the campaign
- `coverage_efficiency`: Area covered per distance flown
- `overlap_percentage`: Percentage of overlapping image regions
- `flight_duration_hours`: Estimated flight duration
- `average_altitude_m`: Average flight altitude
- `image_density_per_sqkm`: Number of images per square kilometer

## Geographic Merging

### GeographicMerger

The `GeographicMerger` class intelligently merges detections across overlapping geographic regions:

```python
from src.wildetect.core.data import GeographicMerger

# Initialize merger with distance threshold
merger = GeographicMerger(merge_distance_threshold_m=50.0)

# Merge detections geographically
geographic_dataset = merger.merge_detections_geographically(
    drone_images, overlapping_regions
)
```

### Merging Algorithm

1. **Geographic Detection Collection**: Collect all detections with GPS coordinates
2. **Class-based Grouping**: Group detections by class name for accurate merging
3. **Proximity Clustering**: Use hierarchical clustering to group nearby detections
4. **Confidence Aggregation**: Combine confidence scores using maximum or average
5. **Bounding Box Calculation**: Calculate geographic bounding boxes for merged detections

### MergedDetection Features

- `class_name`: Detection class (e.g., "elephant", "giraffe")
- `center_lat/lon`: Geographic center of merged detection
- `confidence`: Maximum confidence score from source detections
- `bounding_box`: Geographic bounding box (lat_min, lat_max, lon_min, lon_max)
- `source_images`: List of source image paths
- `detection_count`: Number of source detections merged
- `average_confidence`: Average confidence of source detections
- `geographic_area_sqm`: Geographic area covered by merged detection

## Performance Optimizations

### PerformanceOptimizer

The `PerformanceOptimizer` class provides caching and parallel processing capabilities:

```python
from src.wildetect.core.data import PerformanceOptimizer

# Initialize optimizer
optimizer = PerformanceOptimizer(cache_dir="cache", max_workers=4)

# Cache expensive computations
optimizer.cache_flight_path(flight_path, campaign_id)
optimizer.cache_geographic_dataset(dataset, campaign_id)

# Load cached results
cached_flight_path = optimizer.load_cached_flight_path(campaign_id)

# Parallel processing
parallel_flight_path = optimizer.parallel_flight_path_analysis(drone_images, analyzer)
```

### Optimization Features

#### Caching System
- **Flight Path Caching**: Cache flight path analysis results
- **Geographic Dataset Caching**: Cache merged detection datasets
- **Automatic Cleanup**: Remove old cache files based on age
- **Memory Management**: Clear caches to free memory

#### Parallel Processing
- **GPS Data Extraction**: Parallel extraction of GPS coordinates
- **Overlap Detection**: Parallel processing of image pairs
- **Worker Pool Management**: Optimal worker count based on CPU cores
- **Error Handling**: Graceful handling of parallel processing errors

#### Memory Optimization
- **Image Data Clearing**: Remove unnecessary image data from memory
- **Garbage Collection**: Force garbage collection after optimizations
- **Memory Monitoring**: Track memory usage and performance metrics

## Enhanced CensusData Integration

The `CensusData` class has been enhanced with Phase 2 features:

```python
from src.wildetect.core.data import CensusData, LoaderConfig

# Initialize with Phase 2 capabilities
census_data = CensusData(
    campaign_id="wildlife_survey_2024",
    loading_config=loader_config,
    metadata=campaign_metadata
)

# Phase 2 analysis methods
flight_path = census_data.analyze_flight_path()
efficiency = census_data.calculate_flight_efficiency()
overlapping_regions = census_data.detect_overlapping_regions()
geographic_dataset = census_data.merge_detections_geographically()

# Complete Phase 2 analysis
results = census_data.run_complete_phase2_analysis()
```

### New Methods

#### Flight Analysis
- `analyze_flight_path()`: Analyze GPS-based flight path
- `calculate_flight_efficiency()`: Calculate efficiency metrics
- `detect_overlapping_regions()`: Detect overlapping image regions

#### Geographic Merging
- `merge_detections_geographically()`: Merge detections across regions
- `export_geographic_dataset()`: Export merged dataset to file

#### Enhanced Statistics
- `get_enhanced_campaign_statistics()`: Comprehensive statistics including Phase 2 data
- `run_complete_phase2_analysis()`: Complete Phase 2 analysis workflow

## Usage Examples

### Basic Flight Path Analysis

```python
from src.wildetect.core.data import CensusData, LoaderConfig
from src.wildetect.core.config import FlightSpecs

# Setup configuration
loader_config = LoaderConfig(
    image_dir="data/drone_images",
    tile_size=640,
    overlap=0.2,
    flight_specs=FlightSpecs(
        sensor_height=24.0,
        focal_length=35.0,
        flight_height=180.0
    )
)

# Initialize campaign
census_data = CensusData(
    campaign_id="elephant_survey_2024",
    loading_config=loader_config
)

# Add images and create drone images
census_data.add_images_from_directory("data/drone_images")
census_data.create_drone_images()

# Perform flight analysis
flight_path = census_data.analyze_flight_path()
efficiency = census_data.calculate_flight_efficiency()

print(f"Flight distance: {efficiency.total_distance_km:.2f} km")
print(f"Area covered: {efficiency.total_area_covered_sqkm:.2f} sq km")
print(f"Coverage efficiency: {efficiency.coverage_efficiency:.2f}")
```

### Geographic Merging with Performance Optimization

```python
from src.wildetect.core.data import PerformanceOptimizer

# Initialize performance optimizer
optimizer = PerformanceOptimizer(cache_dir="cache", max_workers=4)

# Check for cached results
cached_dataset = optimizer.load_cached_geographic_dataset("elephant_survey_2024")

if cached_dataset:
    print(f"Using cached dataset with {len(cached_dataset.merged_detections)} detections")
else:
    # Perform geographic merging
    geographic_dataset = census_data.merge_detections_geographically(
        merge_distance_threshold_m=50.0
    )
    
    # Cache results for future use
    optimizer.cache_geographic_dataset(geographic_dataset, "elephant_survey_2024")

# Export results
census_data.export_geographic_dataset("output/merged_detections.json")
```

### Complete Phase 2 Analysis

```python
# Run complete Phase 2 analysis
results = census_data.run_complete_phase2_analysis(
    overlap_threshold=0.1,
    merge_distance_threshold_m=50.0
)

# Export comprehensive report
census_data.export_detection_report("output/phase2_report.json")

# Get enhanced statistics
stats = census_data.get_enhanced_campaign_statistics()
print(f"Campaign statistics: {stats}")
```

## Performance Considerations

### Memory Management
- Use `optimizer.optimize_memory_usage()` to clear unnecessary image data
- Monitor memory usage with `optimizer.get_performance_metrics()`
- Clear caches periodically with `optimizer.clear_caches()`

### Caching Strategy
- Cache expensive computations like flight path analysis
- Use campaign-specific cache keys for easy retrieval
- Implement cache cleanup to prevent disk space issues

### Parallel Processing
- Adjust `max_workers` based on your system's CPU cores
- Monitor performance with `optimizer.get_performance_metrics()`
- Use ThreadPoolExecutor for I/O-bound tasks and ProcessPoolExecutor for CPU-bound tasks

## File Structure

```
src/wildetect/core/data/
├── flight_analyzer.py          # Flight path analysis
├── geographic_merger.py        # Geographic detection merging
├── performance_optimizer.py    # Performance optimizations
├── dataset.py                  # Enhanced CensusData with Phase 2
├── loader.py                   # Data loading utilities
├── tile.py                     # Tile management
├── drone_image.py              # Drone image representation
├── detection.py                # Detection data structures
└── utils.py                    # Utility functions
```

## Configuration

### LoaderConfig Enhancements
- `image_dir`: Directory containing drone images
- `flight_specs`: Flight specifications for GPS calculations
- `extract_gps`: Enable GPS extraction from images
- `cache_images`: Enable image caching for performance

### Performance Settings
- `max_workers`: Number of parallel workers (default: CPU count, max 8)
- `cache_dir`: Directory for caching results (default: "cache")
- `overlap_threshold`: Minimum overlap for detection (default: 0.1)
- `merge_distance_threshold_m`: Distance threshold for merging (default: 50.0)

## Future Enhancements

### Planned Features
- **Real-time Analysis**: Live flight path tracking and analysis
- **Advanced Clustering**: Machine learning-based detection clustering
- **3D Visualization**: Three-dimensional flight path and detection visualization
- **Integration APIs**: REST APIs for external system integration
- **Cloud Deployment**: Cloud-based processing and storage capabilities

### Performance Improvements
- **GPU Acceleration**: CUDA-based parallel processing
- **Distributed Processing**: Multi-node processing for large datasets
- **Streaming Analysis**: Real-time processing of drone video streams
- **Advanced Caching**: Redis-based distributed caching

## Troubleshooting

### Common Issues

1. **No GPS Data**: Ensure images contain GPS metadata in EXIF format
2. **Memory Issues**: Use `optimizer.optimize_memory_usage()` for large datasets
3. **Slow Performance**: Enable caching and adjust `max_workers` parameter
4. **Cache Corruption**: Clear cache directory and regenerate results

### Debug Mode
Enable debug logging for detailed analysis:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring
Monitor performance metrics:

```python
optimizer = PerformanceOptimizer()
metrics = optimizer.get_performance_metrics()
print(f"Memory usage: {metrics['memory_usage_mb']:.1f} MB")
print(f"CPU cores: {metrics['cpu_count']}")
```

## Conclusion

Phase 2 provides a comprehensive suite of advanced analysis capabilities for drone image campaigns. The combination of flight path analysis, geographic merging, and performance optimizations enables sophisticated wildlife monitoring and habitat analysis workflows.

For more information, see the example scripts in the `examples/` directory and the API documentation in the source code. 