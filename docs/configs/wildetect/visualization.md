# Visualization Configuration

> **Location**: `config/visualization.yaml`

**Purpose**: Configuration file for geographic visualization of detection results. Creates interactive maps showing detection locations, generates statistics, and visualizes wildlife distribution patterns.

## Configuration Structure

### Complete Parameter Reference

```yaml
# Visualization Configuration
# This file contains all parameters needed for visualization commands

image_dir: null

# Geographic Visualization of predictions
geographic:
  create_map: true
  show_confidence: false
  output_directory: "visualizations"
  map_type: "OpenStreetMap"  # OpenStreetMap, folium, etc.
  zoom_level: 12
  center_on_data: true

# Flight Specifications
flight_specs:
  sensor_height: 15.6  # mm
  focal_length: 16.0   # mm
  flight_height: 120.0  # meters

# Visualization Options
visualization:
  show_detections: true
  show_footprints: true
  show_statistics: true
  color_by_confidence: false
  confidence_threshold: 0.2

# Output Configuration
output:
  format: "html"  # html, png, pdf
  include_legend: true
  include_statistics: true
  auto_open: false  # Open in browser automatically

# Logging Configuration
logging:
  verbose: false
  log_file: null  # Will use default log path if null
```

### Parameter Descriptions

#### `image_dir`
- **Type**: string or null
- **Description**: Directory containing images to visualize. If `null`, uses detection results from previous runs.

#### `geographic`
Geographic visualization settings for creating maps.

- **`create_map`** (bool): Whether to create an interactive geographic map
- **`show_confidence`** (bool): Whether to display confidence scores on the map
- **`output_directory`** (string): Directory where visualization outputs will be saved
- **`map_type`** (string): Basemap provider. Options: `"OpenStreetMap"`, `"folium"`, or other supported providers
- **`zoom_level`** (int): Default zoom level for the map (1-20, higher = more zoomed in)
- **`center_on_data`** (bool): Whether to automatically center map on detection data

#### `flight_specs`
Flight and camera specifications for accurate geographic positioning.

- **`sensor_height`** (float): Camera sensor height in millimeters
- **`focal_length`** (float): Lens focal length in millimeters
- **`flight_height`** (float): Flight altitude in meters

#### `visualization`
Visualization display options.

- **`show_detections`** (bool): Whether to show detection markers on the map
- **`show_footprints`** (bool): Whether to show image footprints/coverage areas
- **`show_statistics`** (bool): Whether to display statistics overlay
- **`color_by_confidence`** (bool): Whether to color-code detections by confidence score
- **`confidence_threshold`** (float): Minimum confidence threshold for displaying detections (0.0-1.0)

#### `output`
Output format and options.

- **`format`** (string): Output file format. Options: `"html"` (interactive), `"png"` (static image), `"pdf"` (document)
- **`include_legend`** (bool): Whether to include a legend in the visualization
- **`include_statistics`** (bool): Whether to include statistics panel
- **`auto_open`** (bool): Whether to automatically open the visualization in a browser after creation

#### `logging`
Logging configuration.

- **`verbose`** (bool): Enable verbose logging output
- **`log_file`** (string, optional): Path to log file. If `null`, uses default log path

---

## Example Configurations

### Basic Geographic Visualization

```yaml
image_dir: "results/detections/"

geographic:
  create_map: true
  show_confidence: false
  output_directory: "visualizations"
  map_type: "OpenStreetMap"
  zoom_level: 12
  center_on_data: true

flight_specs:
  sensor_height: 15.6
  focal_length: 16.0
  flight_height: 120.0

visualization:
  show_detections: true
  show_footprints: true
  show_statistics: true
  color_by_confidence: false
  confidence_threshold: 0.2

output:
  format: "html"
  include_legend: true
  include_statistics: true
  auto_open: true
```

### High-Confidence Visualization

```yaml
geographic:
  create_map: true
  show_confidence: true
  output_directory: "visualizations/high_confidence"
  zoom_level: 14

visualization:
  show_detections: true
  show_footprints: false
  show_statistics: true
  color_by_confidence: true
  confidence_threshold: 0.7  # Only high-confidence detections

output:
  format: "html"
  include_legend: true
  auto_open: false
```

### Static Map Export

```yaml
geographic:
  create_map: true
  output_directory: "visualizations/static"
  map_type: "OpenStreetMap"
  zoom_level: 11

visualization:
  show_detections: true
  show_footprints: true
  show_statistics: false
  confidence_threshold: 0.5

output:
  format: "png"  # Static image
  include_legend: true
  include_statistics: false
  auto_open: false
```

### PDF Report Generation

```yaml
geographic:
  create_map: true
  output_directory: "visualizations/report"
  zoom_level: 10

visualization:
  show_detections: true
  show_footprints: true
  show_statistics: true
  confidence_threshold: 0.3

output:
  format: "pdf"  # PDF document
  include_legend: true
  include_statistics: true
  auto_open: false
```

### Detailed Analysis Visualization

```yaml
geographic:
  create_map: true
  show_confidence: true
  output_directory: "visualizations/detailed"
  map_type: "OpenStreetMap"
  zoom_level: 15  # High zoom for detail
  center_on_data: true

flight_specs:
  sensor_height: 24.0
  focal_length: 35.0
  flight_height: 180.0

visualization:
  show_detections: true
  show_footprints: true
  show_statistics: true
  color_by_confidence: true
  confidence_threshold: 0.2

output:
  format: "html"
  include_legend: true
  include_statistics: true
  auto_open: true

logging:
  verbose: true
```

---

## Best Practices

1. **Output Format Selection**:
   - Use `"html"` for interactive exploration and sharing
   - Use `"png"` for static images in reports
   - Use `"pdf"` for formal documentation

2. **Zoom Level**:
   - Lower zoom (8-10) for overview of large areas
   - Medium zoom (12-13) for regional analysis
   - Higher zoom (14-16) for detailed local views

3. **Confidence Threshold**:
   - Lower threshold (0.2-0.3) to see all detections
   - Medium threshold (0.5) for balanced view
   - Higher threshold (0.7+) for high-confidence only

4. **Color Coding**: Enable `color_by_confidence: true` to visually distinguish detection quality

5. **Statistics**: Include statistics for quantitative analysis and reporting

6. **Flight Specs**: Ensure flight specifications match actual survey for accurate positioning

7. **Auto-Open**: Use `auto_open: true` for quick viewing, `false` for batch processing

8. **Output Directory**: Organize visualizations by campaign or date for easy management

---

## Troubleshooting

### Map Not Created

**Issue**: Visualization completes but no map file is generated

**Solutions**:
1. Verify `create_map: true` is enabled
2. Check `output_directory` exists or can be created
3. Ensure detection results contain GPS coordinates
4. Check file permissions for output directory
5. Review logs for error messages

### No Detections Shown on Map

**Issue**: Map is created but shows no detection markers

**Solutions**:
1. Check `show_detections: true` is enabled
2. Verify detection results contain valid GPS coordinates
3. Lower `confidence_threshold` to show more detections
4. Check if detections are outside map bounds (adjust zoom/center)
5. Verify detection results file format is correct

### GPS Coordinates Missing

**Issue**: Cannot create map because GPS data is missing

**Solutions**:
1. Ensure images have EXIF GPS metadata
2. Or use detection results that include GPS coordinates
3. Verify `flight_specs` are set for geographic calculations
4. Check if GPS data was included in detection pipeline
5. Use `exif_gps_update` in detection config if GPS is in CSV

### Map Too Zoomed In/Out

**Issue**: Map zoom level is inappropriate

**Solutions**:
1. Adjust `zoom_level` (lower = zoomed out, higher = zoomed in)
2. Enable `center_on_data: true` to auto-center
3. Manually set map center coordinates if needed
4. Test different zoom levels (8-16 range)

### HTML File Won't Open

**Issue**: Generated HTML file doesn't open or display correctly

**Solutions**:
1. Check browser compatibility (use modern browser)
2. Verify file path is correct
3. Check if file was fully written (file size > 0)
4. Try opening in different browser
5. Check browser console for JavaScript errors

### Static Export Fails

**Issue**: PNG or PDF export doesn't work

**Solutions**:
1. Verify required libraries are installed (PIL, reportlab)
2. Check `format` is set correctly (`"png"` or `"pdf"`)
3. Ensure sufficient disk space
4. Check file permissions
5. Try HTML format first to verify visualization works

### Performance Issues

**Issue**: Visualization is slow or hangs

**Solutions**:
1. Reduce number of detections (increase `confidence_threshold`)
2. Disable `show_footprints` if not needed
3. Use lower `zoom_level` for faster rendering
4. Process in smaller batches
5. Check available system memory

---

## Related Documentation

- [Configuration Overview](index.md)
- [Detection Config](detection.md)
- [Census Config](census.md)
- [GPS Extraction Config](extract-gps.md)
- [Visualization Tutorial](../../tutorials/end-to-end-detection.md)

