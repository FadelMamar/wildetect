# WilData GPS Update Configuration

Reference for the `update-gps-from-csv` YAML configuration file.

## Overview

The GPS update config controls how EXIF GPS metadata is written into image files from a CSV source. This is used when aerial survey images lack GPS data in their EXIF headers and the coordinates are available separately (e.g., from a flight log CSV).

**Usage:**

```bash
wildata update-gps-from-csv -c configs/gps-update-config-example.yaml
```

---

## Configuration Fields

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_folder` | `str` | Path to folder containing images to update |
| `csv_path` | `str` | Path to CSV file with GPS coordinates |
| `output_dir` | `str` | Output directory for updated images (copies with GPS EXIF) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `skip_rows` | `int` | `0` | Number of rows to skip in CSV (e.g., metadata header rows) |
| `filename_col` | `str` | `filename` | CSV column name containing image filenames |
| `lat_col` | `str` | `latitude` | CSV column name for latitude values |
| `lon_col` | `str` | `longitude` | CSV column name for longitude values |
| `alt_col` | `str` | `altitude` | CSV column name for altitude values |

---

## Expected CSV Format

The CSV file should contain at minimum the filename, latitude, and longitude columns:

```csv
filename,latitude,longitude,altitude
image_001.jpg,40.7128,-74.0060,10.5
image_002.jpg,40.7589,-73.9851,15.2
image_003.jpg,40.7505,-73.9934,8.1
```

!!! note
    The `skip_rows` parameter is useful when the CSV has metadata header rows before the actual column headers (common with flight controller exports).

---

## Complete Example

```yaml
# Required parameters
image_folder: D:/surveys/flight_2024/images
csv_path: D:/surveys/flight_2024/flight_log.csv
output_dir: D:/surveys/flight_2024/images_with_gps

# Optional parameters
skip_rows: 4              # Skip flight controller metadata rows
filename_col: "filename"
lat_col: "latitude"
lon_col: "longitude"
alt_col: "altitude"
```

---

**See also:**

- [WilData CLI Reference](../../api-reference/wildata-cli.md) — `update-gps-from-csv` command
- [GPS Extraction Config (WildDetect)](../wildetect/extract-gps.md)
