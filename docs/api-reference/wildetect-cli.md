# WildDetect CLI Reference

Complete command-line interface reference for WildDetect.

## Main Commands

```bash
wildetect [COMMAND] [OPTIONS]
```

### detect

Run wildlife detection on images.

```bash
wildetect detect <images> [OPTIONS]
```

**Arguments:**
- `images`: Path to image file or directory

**Options:**
- `-m, --model TEXT`: Model path or MLflow model name
- `-c, --config PATH`: Configuration file path
- `-o, --output PATH`: Output directory
- `--device TEXT`: Device (cuda/cpu/auto)
- `--batch-size INTEGER`: Batch size
- `--confidence FLOAT`: Confidence threshold
- `--tile-size INTEGER`: Tile size for large images
- `-v, --verbose`: Verbose output

**Examples:**
```bash
# Basic detection
wildetect detect images/ --model detector.pt

# With config file
wildetect detect images/ -c config/detection.yaml

# Custom settings
wildetect detect images/ --model detector.pt --batch-size 32 --confidence 0.7
```

### census

Run census campaign with analysis and reporting.

```bash
wildetect census <campaign_name> <images> [OPTIONS]
```

**Arguments:**
- `campaign_name`: Census campaign name
- `images`: Image directory

**Options:**
- `-c, --config PATH`: Configuration file (required)
- `-o, --output PATH`: Output directory
- `--species TEXT`: Target species (comma-separated)
- `--generate-report`: Generate PDF report

**Examples:**
```bash
wildetect census summer_2024 images/ -c config/census.yaml
```

### analyze

Analyze detection results.

```bash
wildetect analyze <results> [OPTIONS]
```

**Arguments:**
- `results`: Path to detection results JSON

**Options:**
- `-o, --output PATH`: Output directory
- `--format TEXT`: Output format (json/csv/excel)

### fiftyone

Manage FiftyOne datasets.

```bash
wildetect fiftyone [OPTIONS]
```

**Options:**
- `--action TEXT`: Action (launch/info/export)
- `--dataset TEXT`: Dataset name
- `--port INTEGER`: Port number

**Examples:**
```bash
# Launch viewer
wildetect fiftyone --action launch --dataset my_dataset

# Get dataset info
wildetect fiftyone --action info --dataset my_dataset

# Export
wildetect fiftyone --action export --format coco --output export/
```

### ui

Launch Streamlit web interface.

```bash
wildetect ui
```

### info

Show system and environment information.

```bash
wildetect info
```

---

For detailed script documentation, see [WildDetect Scripts](../scripts/wildetect/index.md).

