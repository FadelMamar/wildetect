# WilData CLI Reference

Complete command-line interface reference for the `wildata` CLI.

WilData uses [Typer](https://typer.tiangolo.com/) for its CLI, providing dataset import, transformation, ROI extraction, visualization, GPS management, and a REST API server.

```bash
wildata [COMMAND] [OPTIONS]
```

---

## `version`

Show version information.

```bash
wildata version
```

---

## `import-dataset`

Import a dataset from various formats (COCO, YOLO, Label Studio) into the WilData pipeline.

```bash
wildata import-dataset [SOURCE_PATH] [OPTIONS]
```

You can use **either** a config file (`--config`) or provide arguments directly — not both.

### Using a Config File (Recommended)

```bash
wildata import-dataset --config configs/import-config-example.yaml
```

### Using Direct Arguments

```bash
wildata import-dataset annotations.json --format coco --name my_dataset
```

### All Options

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `SOURCE_PATH` | `PATH` | `None` | Path to source dataset (argument) |
| `-c`, `--config` | `PATH` | `None` | Path to YAML config file |
| `-f`, `--format` | `str` | `None` | Source format: `coco`, `yolo`, or `ls` |
| `-n`, `--name` | `str` | `None` | Dataset name |
| `-r`, `--root` | `str` | `data` | Root directory for data storage |
| `-s`, `--split` | `str` | `train` | Split name: `train`, `val`, or `test` |
| `-m`, `--mode` | `str` | `batch` | Processing mode: `streaming` or `batch` |
| `--track-dvc` | `bool` | `false` | Track dataset with DVC |
| `--bbox-tolerance` | `int` | `5` | Bounding box validation tolerance |
| `--dotenv` | `str` | `None` | Path to `.env` file |
| `--ls-config` | `str` | `None` | Label Studio XML config path |
| `--parse-ls-config` | `bool` | `false` | Parse Label Studio config |
| `--disable-roi` | `bool` | `false` | Disable ROI extraction |
| `-v`, `--verbose` | `bool` | `false` | Verbose output |

#### Transformation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--enable-bbox-clipping` | `bool` | `None` | Enable bounding box clipping |
| `--bbox-clipping-tolerance` | `int` | `5` | Bbox clipping tolerance |
| `--skip-invalid-bbox` | `bool` | `false` | Skip invalid bounding boxes |
| `--enable-augmentation` | `bool` | `None` | Enable data augmentation |
| `--aug-prob` | `float` | `1.0` | Augmentation probability |
| `--num-augs` | `int` | `2` | Number of augmentations per image |
| `--enable-tiling` | `bool` | `None` | Enable image tiling |
| `--tile-size` | `int` | `512` | Tile size in pixels |
| `--tile-stride` | `int` | `416` | Tile stride in pixels |
| `--min-visibility` | `float` | `0.1` | Minimum visibility ratio for bboxes in tiles |

**Examples:**

```bash
# From config (recommended for production)
wildata import-dataset -c configs/import-config-example.yaml

# Direct with tiling
wildata import-dataset annotations.json \
    --format coco \
    --name wildlife_train \
    --enable-tiling \
    --tile-size 800 \
    --tile-stride 640

# With verbose output
wildata import-dataset -c my_config.yaml -v
```

See [Import Config Reference](../configs/wildata/import-config.md) for full YAML config documentation.

---

## `bulk-import-datasets`

Bulk import multiple datasets from all files in a directory using multiprocessing.

```bash
wildata bulk-import-datasets [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | *(required)* | Path to YAML config file |
| `-v`, `--verbose` | `bool` | `false` | Verbose output |
| `-n`, `--num-workers` | `int` | `2` | Number of parallel workers |

Each file in the source directory is imported as a separate dataset. Dataset names are derived from filenames.

**Example:**

```bash
wildata bulk-import-datasets -c configs/bulk-import-train.yaml -n 4
```

---

## `create-roi-dataset`

Create an ROI (Region of Interest) classification dataset from a source detection dataset.

```bash
wildata create-roi-dataset [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | *(required)* | Path to YAML config file |
| `-v`, `--verbose` | `bool` | `false` | Verbose output |

**Example:**

```bash
wildata create-roi-dataset -c configs/roi-create-config.yaml
```

See [ROI Config Reference](../configs/wildata/roi-config.md) for config documentation.

---

## `bulk-create-roi-datasets`

Bulk create ROI datasets from all files in a directory using multiprocessing.

```bash
wildata bulk-create-roi-datasets [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | *(required)* | Path to YAML config file |
| `-v`, `--verbose` | `bool` | `false` | Verbose output |
| `-n`, `--num-workers` | `int` | `2` | Number of parallel workers |

**Example:**

```bash
wildata bulk-create-roi-datasets -c configs/bulk-roi-create-config.yaml -n 4
```

---

## `list-datasets`

List all available datasets in the data root.

```bash
wildata list-datasets [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-r`, `--root` | `str` | `data` | Root directory for data storage |
| `-v`, `--verbose` | `bool` | `false` | Show detailed info per dataset |

**Example:**

```bash
# Basic listing
wildata list-datasets

# With details
wildata list-datasets -v --root D:/data
```

---

## `visualize-classification`

Visualize a classification dataset in FiftyOne.

```bash
wildata visualize-classification DATASET_NAME [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `DATASET_NAME` | `str` | *(required)* | Name for the FiftyOne dataset |
| `--root` | `str` | `None` | Root data directory |
| `--single-class` | `bool` | `false` | Load as single class |
| `--background-class` | `str` | `background` | Background class name |
| `--single-class-name` | `str` | `wildlife` | Single class name |
| `--keep-classes` | `str` | `None` | Comma-separated list of classes to keep |
| `--discard-classes` | `str` | `None` | Comma-separated list of classes to discard |
| `--split` | `str` | `train` | Dataset split |

**Example:**

```bash
wildata visualize-classification my_roi_dataset \
    --root D:/data/roi \
    --single-class \
    --split train
```

---

## `visualize-detection`

Visualize a detection dataset in FiftyOne.

```bash
wildata visualize-detection DATASET_NAME [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `DATASET_NAME` | `str` | *(required)* | Name for the FiftyOne dataset |
| `--root` | `str` | *(required)* | Root data directory |
| `--split` | `str` | `train` | Dataset split |

**Example:**

```bash
wildata visualize-detection my_detection_dataset --root D:/data/detection --split val
```

---

## `update-gps-from-csv`

Update EXIF GPS metadata for images using coordinates from a CSV file.

```bash
wildata update-gps-from-csv [OPTIONS]
```

You can use **either** a config file (`--config`) or provide arguments directly — not both.

### All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--config` | `PATH` | `None` | Path to YAML config file |
| `-i`, `--image-folder` | `PATH` | `None` | Path to folder containing images |
| `--csv` | `PATH` | `None` | Path to CSV file with GPS coordinates |
| `-o`, `--output` | `PATH` | `None` | Output directory for updated images |
| `--skip-rows` | `int` | `0` | Number of rows to skip in CSV |
| `--filename-col` | `str` | `filename` | CSV column name for filenames |
| `--lat-col` | `str` | `latitude` | CSV column name for latitude |
| `--lon-col` | `str` | `longitude` | CSV column name for longitude |
| `--alt-col` | `str` | `altitude` | CSV column name for altitude |
| `-v`, `--verbose` | `bool` | `false` | Verbose output |

**Examples:**

```bash
# Using config file
wildata update-gps-from-csv -c configs/gps-update-config-example.yaml

# Using direct arguments
wildata update-gps-from-csv \
    --image-folder images/ \
    --csv coordinates.csv \
    --output updated_images/ \
    --skip-rows 4
```

See [GPS Update Config Reference](../configs/wildata/gps-update-config.md) for YAML config details.

---

## `api` — API Server Commands

REST API server for programmatic access to WilData features.

### `api serve`

Start the WilData API server.

```bash
wildata api serve [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-h`, `--host` | `str` | `0.0.0.0` | Host to bind to |
| `-p`, `--port` | `int` | `8000` | Port to bind to |
| `-r`, `--reload` | `bool` | `false` | Enable auto-reload |
| `-w`, `--workers` | `int` | `1` | Number of worker processes |

**Example:**

```bash
wildata api serve --port 8080 --workers 4
```

API documentation is available at `http://<host>:<port>/docs`.

---

### `api check`

Check current API configuration values.

```bash
wildata api check
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `wildata import-dataset -c CONFIG` | Import dataset from config |
| `wildata bulk-import-datasets -c CONFIG` | Bulk import datasets |
| `wildata create-roi-dataset -c CONFIG` | Create ROI dataset |
| `wildata bulk-create-roi-datasets -c CONFIG` | Bulk create ROI datasets |
| `wildata list-datasets` | List all datasets |
| `wildata visualize-classification NAME` | Visualize classification data |
| `wildata visualize-detection NAME --root DIR` | Visualize detection data |
| `wildata update-gps-from-csv -c CONFIG` | Update image GPS from CSV |
| `wildata api serve` | Start REST API server |
| `wildata version` | Show version |

---

For configuration file details, see [Configuration Reference](../configs/wildata/index.md).  
For shell scripts, see [WilData Scripts](../scripts/wildata/index.md).
