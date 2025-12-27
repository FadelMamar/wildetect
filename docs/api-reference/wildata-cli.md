# WilData CLI Reference

Complete command-line interface reference for WilData.

## Main Commands

### import-dataset

Import dataset from various formats.

```bash
wildata import-dataset [SOURCE] [OPTIONS]
```

**Options:**
- `-f, --format TEXT`: Source format (coco/yolo/ls)
- `-n, --name TEXT`: Dataset name
- `-c, --config PATH`: Config file path
- `--root PATH`: Data root directory
- `--split TEXT`: Split name (train/val/test)
- `--enable-tiling`: Enable image tiling
- `--tile-size INTEGER`: Tile size
- `-v, --verbose`: Verbose output

**Examples:**
```bash
# From config
wildata import-dataset --config configs/import-config.yaml

# Direct arguments
wildata import-dataset data.json --format coco --name dataset1
```

### bulk-import-datasets

Bulk import multiple datasets.

```bash
wildata bulk-import-datasets [OPTIONS]
```

**Options:**
- `-c, --config PATH`: Bulk import config
- `-n, --num-workers INTEGER`: Number of workers

### create-roi-dataset

Create ROI classification dataset from detection annotations.

```bash
wildata create-roi-dataset [OPTIONS]
```

**Options:**
- `-c, --config PATH`: ROI config file
- `--roi-size INTEGER`: ROI box size
- `--random-count INTEGER`: Background samples per image

### dataset list

List all datasets.

```bash
wildata dataset list [--root PATH]
```

### dataset export

Export dataset to format.

```bash
wildata dataset export <name> [OPTIONS]
```

**Options:**
- `--format TEXT`: Target format (coco/yolo)
- `--output PATH`: Output directory

### visualize-dataset

Launch dataset visualization.

```bash
wildata visualize-dataset --dataset <name> --split <split>
```

### update-gps-from-csv

Update image GPS metadata from CSV.

```bash
wildata update-gps-from-csv [OPTIONS]
```

**Options:**
- `-c, --config PATH`: GPS update config
- `--image-folder PATH`: Image folder
- `--csv PATH`: CSV file path
- `--output PATH`: Output directory

---

For detailed script documentation, see [WilData Scripts](../scripts/wildata/index.md).

