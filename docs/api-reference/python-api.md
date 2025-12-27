# Python API Reference

Using WildDetect packages programmatically.

## WildDetect

### Detection Pipeline

```python
from wildetect.core.pipeline import DetectionPipeline

# Initialize
pipeline = DetectionPipeline(
    model_path="detector.pt",
    device="cuda"
)

# Detect single image
result = pipeline.detect("image.jpg")

# Detect batch
results = pipeline.detect_batch("images/")

# Save results
pipeline.save_results(results, "results.json")
```

### Census Engine

```python
from wildetect.core.census import CensusEngine, CensusConfig

# Configure
config = CensusConfig.from_yaml("config/census.yaml")

# Run census
engine = CensusEngine(config)
census_result = engine.run_census("survey_images/")

# Generate report
census_result.save_report("report.pdf")
```

## WilData

### Data Pipeline

```python
from wildata.pipeline import DataPipeline

# Initialize
pipeline = DataPipeline("data")

# Import dataset
result = pipeline.import_dataset(
    source_path="annotations.json",
    source_format="coco",
    dataset_name="my_dataset"
)

# List datasets
datasets = pipeline.list_datasets()

# Export
pipeline.export_dataset("my_dataset", "yolo")
```

### ROI Adapter

```python
from wildata.adapters import ROIAdapter
import json

# Load COCO data
with open("annotations.json") as f:
    coco_data = json.load(f)

# Create ROI adapter
adapter = ROIAdapter(
    coco_data,
    roi_box_size=128,
    random_roi_count=10
)

# Convert
roi_data = adapter.convert()

# Save
adapter.save(roi_data, "roi_dataset/")
```

## WildTrain

### Training (Classification)

```python
from wildtrain.models import ImageClassifier
from wildtrain.data import ClassificationDataModule
import pytorch_lightning as pl

# Create model
model = ImageClassifier(
    architecture="resnet50",
    num_classes=10,
    learning_rate=0.001
)

# Create data module
datamodule = ClassificationDataModule(
    data_root="data/roi_dataset",
    batch_size=32
)

# Train
trainer = pl.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, datamodule)
```

### Model Registry

```python
from wildtrain.registry import ModelRegistry

# Initialize
registry = ModelRegistry("http://localhost:5000")

# Register model
version = registry.register_model(
    model_path="checkpoints/best.ckpt",
    model_name="wildlife_classifier",
    description="ResNet50 classifier",
    tags={"accuracy": "0.95"}
)

# Load model
model = registry.load_model("wildlife_classifier", version="latest")

# Promote to production
registry.promote_model("wildlife_classifier", version, "Production")
```

---

For complete architecture details, see:
- [WildDetect Architecture](../architecture/wildetect.md)
- [WilData Architecture](../architecture/wildata.md)
- [WildTrain Architecture](../architecture/wildtrain.md)

