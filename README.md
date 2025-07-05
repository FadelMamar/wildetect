# WildDetect - Wildlife Detection from Aerial Images

A standalone software for semi-automated detection and counting of wildlife species from aerial images, with FiftyOne integration for visualization and annotation collection.

## MVP Features

### Core Functionality
- **Aerial Image Processing**: Load and preprocess drone/satellite imagery
- **Wildlife Detection**: Detect and count wildlife species using pre-trained models
- **FiftyOne Integration**: Visualize detections and collect annotations
- **LabelStudio Integration**: Professional annotation job management
- **Annotation Management**: Store and manage manual corrections
- **Model Retraining**: Fine-tune detection models with new annotations
- **Simple UI**: Basic interface to orchestrate the workflow

### Technical Stack
- **Python 3.8+**: Core application language
- **PyTorch/YOLO**: Detection models
- **FiftyOne**: Dataset visualization and annotation
- **LabelStudio**: Professional annotation management
- **FastAPI**: REST API for the backend
- **Streamlit**: Simple web interface
- **SQLite**: Local annotation storage
- **OpenCV**: Image processing utilities

## Project Structure

```
wildetect/
├── app/                    # Main application
│   ├── api/               # FastAPI backend
│   ├── ui/                # Streamlit frontend
│   └── core/              # Core detection logic
├── models/                 # Pre-trained models
├── data/                  # Dataset storage
│   ├── images/            # Input aerial images
│   ├── annotations/       # Collected annotations
│   └── datasets/          # FiftyOne datasets
├── config/                # Configuration files
├── scripts/               # Utility scripts
└── tests/                 # Unit tests
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Model**:
   ```bash
   python scripts/download_models.py
   ```

3. **Run Detection**:
   ```bash
   python scripts/detect.py --images data/images/*.jpg --confidence 0.5 --results detections.json
   ```

4. **Add to FiftyOne**:
   ```bash
   python scripts/fiftyone.py add --images data/images/*.jpg --detections detections.json
   ```

5. **Launch FiftyOne**:
   ```bash
   python scripts/fiftyone.py launch
   ```

6. **Start LabelStudio** (optional):
   ```bash
   python scripts/start_labelstudio.py
   ```

## CLI Usage Workflow

### 1. Detection
```bash
# Run detection on images
python scripts/detect.py --images data/images/*.jpg --confidence 0.5 --results detections.json

# Save visualizations
python scripts/detect.py --images data/images/*.jpg --output data/visualizations --results detections.json
```

### 2. FiftyOne Management
```bash
# Add images with detections to FiftyOne
python scripts/fiftyone.py add --images data/images/*.jpg --detections detections.json

# Launch FiftyOne app
python scripts/fiftyone.py launch

# Export dataset annotations
python scripts/fiftyone.py export --output data/annotations --format coco

# Show dataset statistics
python scripts/fiftyone.py stats
```

### 3. Training Pipeline
```bash
# Prepare training data from FiftyOne dataset
python scripts/train.py prepare-data --dataset wildlife_detection --output data/training

# Train model
python scripts/train.py train --data data/training/dataset.yaml --epochs 100 --output models

# Evaluate model
python scripts/train.py evaluate --model models/best.pt --data data/training/dataset.yaml
```

### 4. Complete Workflow Example
```bash
# 1. Run detection
python scripts/detect.py --images data/images/*.jpg --results detections.json

# 2. Add to FiftyOne for review
python scripts/fiftyone.py add --images data/images/*.jpg --detections detections.json

# 3. Launch FiftyOne for annotation
python scripts/fiftyone.py launch

# 4. Export annotations for training
python scripts/fiftyone.py export --output data/annotations --format yolo

# 5. Prepare training data
python scripts/train.py prepare-data --dataset wildlife_detection --output data/training

# 6. Train improved model
python scripts/train.py train --data data/training/dataset.yaml --epochs 100
```

## Configuration

Edit `config/settings.yaml` to customize:
- Model parameters
- Detection thresholds
- File paths
- FiftyOne dataset settings

## Development

- **Backend**: FastAPI with async processing
- **Frontend**: Streamlit for simple web interface
- **Database**: SQLite for local annotation storage
- **Models**: YOLO-based wildlife detection models

## License

MIT License - see LICENSE file for details. 