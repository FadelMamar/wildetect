# FiftyOne Integration for WildDetect

This document describes the integration between WildDetect's `Detection` and `DroneImage` classes with FiftyOne for interactive dataset visualization and analysis.

## Overview

The `FiftyOneManager` class provides a bridge between WildDetect's wildlife detection pipeline and FiftyOne's powerful dataset visualization capabilities. This integration enables:

- **Interactive Visualization**: View detection results in FiftyOne's web interface
- **Metadata Preservation**: Maintain GPS coordinates, geographic footprints, and detection metadata
- **Spatial Analysis**: Filter and analyze detections by geographic regions
- **Quality Assessment**: Track detection confidence and identify challenging samples
- **Dataset Management**: Efficiently add and update detection results

## Key Features

### 1. Detection Integration
- Convert `Detection` objects to FiftyOne format
- Preserve bounding boxes, confidence scores, and class information
- Maintain GPS coordinates and geographic footprints
- Handle coordinate transformations (relative to absolute)

### 2. DroneImage Integration
- Process entire drone images with multiple tiles
- Aggregate detections from all tiles in a drone image
- Preserve image metadata (GPS, timestamp, GSD, geographic footprint)
- Track species counts and detection statistics

### 3. Geographic Features
- Filter samples by geographic bounds
- Find samples with GPS data
- Support spatial analysis and mapping
- Preserve geographic footprints for area calculations

### 4. Advanced Analytics
- Compute similarity between samples
- Identify challenging samples for annotation
- Track detection quality metrics
- Generate comprehensive statistics

## Usage Examples

### Basic Usage

```python
from wildetect.core.visualization.fiftyone_manager import FiftyOneManager
from wildetect.core.data.detection import Detection
from wildetect.core.data.drone_image import DroneImage

# Initialize FiftyOne manager
fiftyone_manager = FiftyOneManager("wildlife_detection_dataset")

# Add individual detections
detections = [detection1, detection2, detection3]
fiftyone_manager.add_detections(detections, "/path/to/image.jpg")

# Add a complete drone image
drone_image = DroneImage.from_image_path("/path/to/drone_image.jpg")
fiftyone_manager.add_drone_image(drone_image)

# Launch the FiftyOne app
fiftyone_manager.launch_app()
```

### Batch Processing

```python
# Add multiple drone images at once
drone_images = [drone_image1, drone_image2, drone_image3]
fiftyone_manager.add_drone_images(drone_images)

# Get statistics
stats = fiftyone_manager.get_annotation_stats()
print(f"Total detections: {stats['total_detections']}")
print(f"Species found: {list(stats['species_counts'].keys())}")
```

### Geographic Analysis

```python
# Get samples with GPS data
gps_samples = fiftyone_manager.get_detections_with_gps()

# Filter by geographic bounds
from wildetect.core.gps.geographic_bounds import GeographicBounds
bounds = GeographicBounds(min_lat=40.0, max_lat=41.0, min_lon=-74.0, max_lon=-73.0)
region_samples = fiftyone_manager.filter_by_geographic_bounds(bounds)
```

## Data Structure Mapping

### Detection → FiftyOne Sample
| Detection Field | FiftyOne Field | Description |
|----------------|----------------|-------------|
| `bbox` | `bounding_box` | Bounding box coordinates |
| `confidence` | `confidence` | Detection confidence score |
| `class_name` | `label` | Species/class name |
| `gps_loc` | `metadata.gps_loc` | GPS coordinates |
| `geographic_footprint` | `metadata.geographic_footprint` | Geographic bounds |

### DroneImage → FiftyOne Sample
| DroneImage Field | FiftyOne Field | Description |
|------------------|----------------|-------------|
| `image_path` | `filepath` | Image file path |
| `latitude/longitude` | `gps_latitude/gps_longitude` | Image GPS coordinates |
| `geographic_footprint` | `geographic_footprint` | Image geographic bounds |
| `gsd` | `gsd` | Ground sample distance |
| `timestamp` | `timestamp` | Image capture timestamp |
| `tiles` | `num_tiles` | Number of tiles in image |
| `predictions` | `detections` | All detections from all tiles |

## Configuration

The FiftyOne integration can be configured through the main WildDetect configuration:

```yaml
fiftyone:
  dataset_name: "wildlife_detection_dataset"
  enable_brain: true  # Enable FiftyOne Brain features
```

## Error Handling

The integration includes comprehensive error handling:

- **Dataset Initialization**: Graceful handling of dataset creation/loading failures
- **Data Validation**: Validation of detection bounding boxes and GPS coordinates
- **Type Safety**: Proper handling of optional fields and null values
- **Graceful Degradation**: Continue processing even if some detections fail

## Performance Considerations

### Batch Operations
- Use `add_drone_images()` for multiple images instead of individual calls
- Process large datasets in chunks to manage memory usage
- Leverage FiftyOne's bulk operations when possible

### Memory Management
- Close datasets properly to free resources
- Monitor memory usage with large datasets
- Use streaming for very large datasets

### Caching
- FiftyOne caches dataset operations automatically
- Avoid redundant conversions of the same detections
- Use dataset persistence for long-running operations

## Advanced Features

### Similarity Computation
```python
# Compute similarity between samples
fiftyone_manager.compute_similarity()
```

### Hard Sample Detection
```python
# Find challenging samples for annotation
hard_samples = fiftyone_manager.find_hardest_samples(num_samples=100)
```

### Dataset Statistics
```python
# Get comprehensive dataset statistics
stats = fiftyone_manager.get_annotation_stats()
print(f"Total samples: {stats['total_samples']}")
print(f"Annotated samples: {stats['annotated_samples']}")
print(f"Species counts: {stats['species_counts']}")
```

## Troubleshooting

### Common Issues

1. **Dataset Not Initialized**
   - Ensure `_init_dataset()` is called before operations
   - Check that dataset name is valid

2. **GPS Data Missing**
   - Verify GPS coordinates are in correct format
   - Check that `Detection` objects have valid GPS data

3. **Bounding Box Errors**
   - Ensure bbox coordinates are within image bounds
   - Validate coordinate format (x1, y1, x2, y2)

4. **Memory Issues**
   - Process datasets in smaller chunks
   - Close datasets when not in use
   - Monitor system memory usage

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("wildetect.core.visualization.fiftyone_manager").setLevel(logging.DEBUG)
```

## Future Enhancements

- **Active Learning**: Integration with annotation workflows
- **Model Comparison**: Compare results from different detection models
- **Temporal Analysis**: Track detections over time
- **Export Features**: Export datasets in various formats
- **Real-time Updates**: Stream detection results to FiftyOne

## Dependencies

- `fiftyone>=0.20.0`
- `numpy>=1.21.0`
- `PIL>=8.0.0`

## Example Script

See `examples/fiftyone_integration_example.py` for a complete working example. 