<div align="center">
  <img src="assets/image.png" alt="WildDetect"/>
</div>

# WildDetect Monorepo

**A complete ecosystem for wildlife monitoring and conservation** using aerial imagery.

WildDetect is an integrated monorepo containing three specialized packages for the entire wildlife detection workflow—from data management to model training and deployment.

## 📦 Packages

### 🗂️ [WilData](wildata/) - Data Management
Unified data pipeline for dataset management and preparation.

- Multi-format import/export (COCO, YOLO, Label Studio)
- Data transformations (tiling, augmentation, clipping)
- ROI dataset creation for classification
- DVC integration for version control
- REST API for programmatic access

**[📖 WilData Documentation](wildata/README.md)**

### 🎓 [WildTrain](wildtrain/) - Model Training
Modular training framework for detection and classification models.

- YOLO and MMDetection support
- PyTorch Lightning for classification
- MLflow experiment tracking
- Hyperparameter optimization (Optuna)
- Model registration and versioning

**[📖 WildTrain Documentation](wildtrain/README.md)**

### 🔍 [WildDetect](src/wildetect/) - Detection & Analysis
Production-ready detection system with census capabilities.

- Multi-threaded detection pipelines
- Large raster image support (GeoTIFF)
- Census campaign orchestration
- Geographic analysis and visualization
- FiftyOne integration
- Comprehensive reporting (JSON, CSV, PDF)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fadelmamar/wildetect.git
cd wildetect

# Create virtual environment
uv venv --python 3.10
.venv\Scripts\activate  # Windows

# Install all packages
cd wildata && uv pip install -e . && cd ..
cd wildtrain && uv pip install -e . && cd ..
uv pip install -e .
```

**[📖 Full Installation Guide](docs/getting-started/installation.md)**

### Quick Detection

```bash
# Run detection
wildetect detect /path/to/images --model model.pt --output results/

# Run census campaign
wildetect census campaign_2024 /path/to/images --model model.pt --output campaign_results/
```

**[📖 Quick Start Guide](docs/getting-started/quick-start.md)**

---

## 📚 Documentation

**Complete documentation is available at: [WildDetect Documentation](docs/index.md)**

### Getting Started
- [Installation Guide](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quick-start.md)
- [Environment Setup](docs/getting-started/environment-setup.md)

### Architecture
- [Monorepo Overview](docs/architecture/overview.md)
- [WilData Architecture](docs/architecture/wildata.md)
- [WildTrain Architecture](docs/architecture/wildtrain.md)
- [WildDetect Architecture](docs/architecture/wildetect.md)
- [Data Flow](docs/architecture/data-flow.md)

### Tutorials
- [End-to-End Detection](docs/tutorials/end-to-end-detection.md)
- [Dataset Preparation](docs/tutorials/dataset-preparation.md)
- [Model Training](docs/tutorials/model-training.md)
- [Census Campaign](docs/tutorials/census-campaign.md)

### Reference
- [WildDetect Scripts](docs/scripts/wildetect/index.md)
- [WilData Scripts](docs/scripts/wildata/index.md)
- [WildTrain Scripts](docs/scripts/wildtrain/index.md)
- [Configuration Files](docs/configs/wildetect/index.md)
- [CLI Reference](docs/api-reference/wildetect-cli.md)
- [Python API](docs/api-reference/python-api.md)

### Support
- [Troubleshooting Guide](docs/troubleshooting.md)
- [GitHub Issues](https://github.com/fadelmamar/wildetect/issues)

---

## 🎯 Main Features

### Detection & Analysis
- Multi-species detection optimized for aerial imagery
- Batch processing of large datasets
- Raster detection for GeoTIFF files
- Multi-threaded pipelines for performance

### Data Management
- Import from COCO, YOLO, Label Studio
- Data transformations (tiling, augmentation)
- ROI extraction for classification
- DVC integration for versioning

### Model Training
- YOLO and MMDetection frameworks
- PyTorch Lightning for classification
- MLflow experiment tracking
- Hyperparameter optimization

### Geographic Analysis
- GPS metadata extraction and management
- Flight path analysis and coverage maps
- Population density calculations
- Interactive visualizations with FiftyOne

### Census Capabilities
- Full census campaign orchestration
- Species counting and statistics
- Distribution analysis
- Comprehensive PDF reports

---

## 🔧 Main CLI Commands

### WildDetect
```bash
wildetect detect      # Run wildlife detection
wildetect census      # Census campaign
wildetect analyze     # Analyze results
wildetect visualize   # Create visualizations
wildetect fiftyone    # Launch FiftyOne viewer
wildetect ui          # Launch web interface
```

### WilData
```bash
wildata import-dataset    # Import dataset
wildata dataset list      # List datasets
wildata dataset export    # Export dataset
wildata create-roi        # Create ROI dataset
wildata visualize-dataset # Visualize data
```

### WildTrain
```bash
wildtrain train classifier  # Train classification model
wildtrain train detector    # Train detection model
wildtrain eval classifier   # Evaluate model
wildtrain register         # Register to MLflow
```

For all options:
```bash
wildetect --help
wildata --help
wildtrain --help
```

## 🐾 WildDetect Command-Line Interface (CLI)

WildDetect provides a powerful and flexible command-line interface (CLI) built with [Typer](https://typer.tiangolo.com/), making it easy to run wildlife detection, census campaigns, analysis, visualization, and more—all from your terminal.

### How to Use

After installing WildDetect, simply run:

```bash
wildetect [COMMAND] [OPTIONS]
```

You can always see all available commands and options with:

```bash
wildetect --help
```

### Main Commands

- **detect**  
  Run wildlife detection on images or directories of images.
  ```bash
  wildetect detect /path/to/images --model model.pt --output results/
  ```
  Options include model type, confidence threshold, device (CPU/GPU), batch size, tiling, and more.

- **census**  
  Orchestrate a full wildlife census campaign, including detection, statistics, and reporting.
  ```bash
  wildetect census campaign_2024 /path/to/images --model model.pt --output campaign_results/
  ```
  Supports campaign metadata, pilot info, target species, and advanced analysis.

- **analyze**  
  Analyze detection results for statistics and insights.
  ```bash
  wildetect analyze results.json --output analysis/
  ```

- **visualize**  
  Create interactive geographic maps and visualizations from detection results.
  ```bash
  wildetect visualize results.json --output maps/
  ```

- **info**  
  Display system and environment information, including dependencies and hardware support.
  ```bash
  wildetect info
  ```

- **ui**  
  Launch the WildDetect web interface (Streamlit-based) for interactive exploration.
  ```bash
  wildetect ui
  ```

- **fiftyone**  
  Manage [FiftyOne](https://voxel51.com/docs/fiftyone/) datasets: launch the app, get info, or export data.
  ```bash
  wildetect fiftyone --action launch
  wildetect fiftyone --action info --dataset my_dataset
  wildetect fiftyone --action export --format coco --output export_dir/
  ```

- **clear-results**  
  Delete all detection results in a specified directory (with confirmation).

### General CLI Features

- **Rich Output**: Uses [rich](https://rich.readthedocs.io/) for beautiful tables, progress bars, and colored logs.
- **Flexible Input**: Accepts both individual image files and directories.
- **Advanced Options**: Fine-tune detection, tiling, device selection, and more.
- **Batch Processing**: Efficiently processes large datasets.
- **Integration**: Seamless export to FiftyOne, JSON, and CSV formats.
- **Help for Every Command**: Use `wildetect [COMMAND] --help` for detailed options.

## 📁 Repository Structure

```
wildetect/                    # Monorepo root
├── wildata/                  # Data management package
│   ├── src/wildata/
│   ├── configs/
│   ├── scripts/
│   └── tests/
├── wildtrain/                # Training framework package
│   ├── src/wildtrain/
│   ├── configs/
│   ├── scripts/
│   └── tests/
├── src/wildetect/            # Detection package
│   ├── core/
│   ├── cli/
│   ├── api/
│   └── ui/
├── config/                   # WildDetect configurations
├── scripts/                  # Batch scripts
├── docs/                     # Documentation
└── tests/                    # Tests
```

---

## 🔄 Common Workflows

### Detection Workflow
```bash
# 1. Run detection
wildetect detect images/ --model model.pt --output results/

# 2. View in FiftyOne
wildetect fiftyone --action launch

# 3. Analyze results
wildetect analyze results/detections.json
```

### Data Preparation Workflow
```bash
# 1. Import dataset
cd wildata
wildata import-dataset annotations.json --format coco --name dataset

# 2. Apply transformations (tiling)
# (configured in import config)

# 3. Export for training
wildata dataset export dataset --format yolo
```

### Training Workflow
```bash
# 1. Train model
cd wildtrain
wildtrain train detector -c configs/detection/yolo.yaml

# 2. Evaluate
wildtrain eval detector -c configs/detection/yolo_eval.yaml

# 3. Register to MLflow
wildtrain register detector config/registration.yaml
```

### Census Workflow
```bash
# 1. Configure census (edit config/census.yaml)

# 2. Run census
wildetect census campaign_2024 images/ -c config/census.yaml

# 3. View report
# Open census_results/report.pdf
```

---

## 🌟 Key Technologies

- **Python 3.9+** - Primary language
- **PyTorch** - Deep learning framework
- **YOLO / MMDetection** - Object detection
- **PyTorch Lightning** - Training framework
- **MLflow** - Experiment tracking
- **FiftyOne** - Dataset visualization
- **FastAPI** - REST API
- **Streamlit** - Web interfaces
- **DVC** - Data versioning
- **Hydra** - Configuration management
- **Typer** - CLI interfaces

---

## 🤝 Contributing

Contributions are welcome! This is an open-source project for the conservation community.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows project style (use `ruff`)
- Tests pass (`uv run pytest tests/ -v`)
- Documentation is updated
- Commit messages are descriptive

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📧 Support & Contact

- **Documentation**: [docs/index.md](docs/index.md)
- **Issues**: [GitHub Issues](https://github.com/fadelmamar/wildetect/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fadelmamar/wildetect/discussions)

---

## 🙏 Acknowledgments

- Conservation organizations using this toolkit
- Open-source community
- YOLO and MMDetection teams
- PyTorch and Lightning teams

---

**Ready to get started?** Head to the [Installation Guide](docs/getting-started/installation.md) or [Quick Start](docs/getting-started/quick-start.md)! 