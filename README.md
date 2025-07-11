# WildDetect - Wildlife Detection System

A comprehensive wildlife detection and census system designed for aerial imagery analysis, featuring advanced geographic analysis, population estimation, and conservation planning tools.

## 🚀 Key Features

### Core Detection
- **Multi-species Detection**: YOLO-based detection for elephants, giraffes, zebras, lions, and more
- **High-accuracy Models**: Pre-trained models optimized for aerial wildlife imagery
- **Batch Processing**: Efficient processing of large image datasets
- **GPU Acceleration**: CUDA support for faster inference

### Wildlife Census & Analysis
- **Campaign Management**: Complete census campaign orchestration
- **Geographic Analysis**: GPS-based coverage mapping and overlap detection
- **Population Statistics**: Species-specific detection counts and density estimation
- **Flight Path Analysis**: Survey efficiency and coverage optimization
- **Interactive Maps**: Folium-based geographic visualizations

### Data Quality & Reporting
- **Confidence Thresholding**: Configurable detection sensitivity
- **Quality Metrics**: False positive filtering and validation
- **Comprehensive Reporting**: JSON exports with statistical summaries
- **Metadata Management**: Campaign tracking and documentation

## 📋 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/wildetect.git
cd wildetect

# Install dependencies
uv sync

# Install CUDA-compatible PyTorch (if CUDA is available)
uv run scripts/install_cuda.py

# Verify installation
uv run python -m wildetect.cli info
```

### CUDA Support

For optimal performance on GPU-enabled systems, install CUDA-compatible PyTorch:

```bash
# Auto-detect and install CUDA PyTorch
uv run python -m wildetect.cli install-cuda

# Install specific CUDA version
uv run python -m wildetect.cli install-cuda --cuda-version 121

# Force CPU-only installation
uv run python -m wildetect.cli install-cuda --cpu-only

# Or use the standalone script
uv run scripts/install_cuda.py
```

## 🎯 Quick Start

### Basic Detection
```bash
# Detect wildlife in images
uv run python -m wildetect.cli detect /path/to/images --model model.pt --output results/
```

### Wildlife Census Campaign
```bash
# Run comprehensive census
uv run python -m wildetect.cli census campaign_2024 /path/to/images \
  --model model.pt \
  --pilot "John Doe" \
  --species elephant giraffe zebra \
  --output campaign_results/
```

### Analysis & Visualization
```bash
# Analyze detection results
uv run python -m wildetect.cli analyze results.json --output analysis/ --map

# Create geographic visualizations
uv run python -m wildetect.cli visualize results.json --output maps/ --map
```

## 🗺️ CLI Commands

### `detect` - Basic Wildlife Detection
```bash
wildetect detect [OPTIONS] IMAGES...

Options:
  --model, -m PATH        Path to model weights
  --confidence, -c FLOAT  Confidence threshold [default: 0.25]
  --device, -d TEXT       Device (auto/cpu/cuda) [default: auto]
  --batch-size, -b INT    Batch size [default: 8]
  --output, -o PATH       Output directory
  --verbose, -v           Verbose logging
```

### `census` - Wildlife Census Campaign
```bash
wildetect census [OPTIONS] CAMPAIGN_ID IMAGES...

Options:
  --model, -m PATH        Path to model weights
  --pilot TEXT            Pilot name for campaign metadata
  --species TEXT...       Target species for detection
  --confidence, -c FLOAT  Confidence threshold [default: 0.25]
  --output, -o PATH       Output directory
  --map                   Create geographic visualization [default: true]
  --verbose, -v           Verbose logging
```

### `analyze` - Post-processing Analysis
```bash
wildetect analyze [OPTIONS] RESULTS_PATH

Options:
  --output, -o PATH       Output directory [default: analysis]
  --map                   Create geographic visualization [default: true]
  --verbose, -v           Verbose logging
```

### `visualize` - Geographic Visualization
```bash
wildetect visualize [OPTIONS] RESULTS_PATH

Options:
  --output, -o PATH       Output directory [default: visualizations]
  --map                   Create geographic visualization [default: true]
  --show-confidence       Show confidence scores [default: true]
```

### `info` - System Information
```bash
wildetect info
```

### `install-cuda` - Install CUDA Support
```bash
uv run python -m wildetect.cli install-cuda [OPTIONS]

Options:
  --cuda-version, -c TEXT  Specific CUDA version (118, 121)
  --cpu-only               Force CPU-only installation
  --verbose, -v            Verbose logging
```

## 🔬 Wildlife Census Features

### Campaign Management
- **Metadata Tracking**: Flight dates, pilot info, equipment details
- **Species Targeting**: Configurable target species lists
- **Mission Objectives**: Survey type and conservation goals
- **Quality Control**: Confidence thresholds and validation

### Geographic Analysis
- **GPS Processing**: Coordinate extraction and validation
- **Coverage Mapping**: Area calculation and overlap detection
- **Flight Path Analysis**: Survey efficiency metrics
- **Interactive Maps**: Folium-based visualizations

### Population Statistics
- **Species Counts**: Per-species detection tallies
- **Density Estimation**: Population density calculations
- **Distribution Mapping**: Geographic species distribution
- **Trend Analysis**: Temporal population changes

### Conservation Applications
- **Protected Area Monitoring**: Regular population surveys
- **Wildlife Corridor Assessment**: Connectivity analysis
- **Threat Assessment**: Population decline detection
- **Habitat Suitability**: Environmental factor analysis

## 📊 Output Formats

### Detection Results
```json
{
  "image_path": "sample.jpg",
  "total_detections": 5,
  "class_counts": {"elephant": 2, "giraffe": 3},
  "confidence_scores": [0.85, 0.92, 0.78, 0.88, 0.91],
  "geographic_bounds": {
    "min_lat": -1.234567,
    "max_lat": -1.234000,
    "min_lon": 36.789000,
    "max_lon": 36.789567
  }
}
```

### Campaign Reports
```json
{
  "campaign_id": "census_2024",
  "metadata": {
    "flight_date": "2024-01-15T10:30:00",
    "pilot_info": {"name": "John Doe"},
    "target_species": ["elephant", "giraffe", "zebra"]
  },
  "statistics": {
    "total_images": 150,
    "total_detections": 45,
    "coverage_area_km2": 25.5,
    "flight_efficiency": 0.87
  }
}
```

## 🗺️ Geographic Visualization

The system generates interactive HTML maps showing:
- **Image Coverage**: Geographic footprints of survey images
- **Detection Locations**: GPS coordinates of wildlife detections
- **Species Distribution**: Color-coded species mapping
- **Coverage Statistics**: Area calculations and efficiency metrics
- **Flight Paths**: Survey route visualization

## 🔧 Configuration

### Model Settings
```yaml
model:
  type: "yolo"
  weights: "models/yolo_wildlife.pt"
  confidence_threshold: 0.5
  device: "auto"
```

### Detection Settings
```yaml
detection:
  min_confidence: 0.3
  species_classes:
    - "elephant"
    - "giraffe"
    - "zebra"
    - "lion"
    - "rhino"
```

## 📈 Performance

- **Processing Speed**: 10-50 images/second (GPU dependent)
- **Memory Usage**: 2-8GB RAM (batch size dependent)
- **Accuracy**: 85-95% mAP on wildlife datasets
- **Scalability**: Supports 1000+ image campaigns

## 🌍 Conservation Impact

WildDetect enables:
- **Population Monitoring**: Regular wildlife surveys
- **Habitat Assessment**: Coverage and connectivity analysis
- **Threat Detection**: Population decline identification
- **Conservation Planning**: Data-driven decision making
- **Research Support**: Scientific analysis and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLO community for detection models
- Conservation organizations for field testing
- Open source contributors for geographic libraries 