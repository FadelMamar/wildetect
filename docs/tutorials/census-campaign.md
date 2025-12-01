# Census Campaign Tutorial

Learn how to conduct a complete wildlife census campaign using WildDetect.

## Overview

A census campaign includes detection, population statistics, geographic analysis, and comprehensive reporting.

## Prerequisites

- WildDetect installed
- Aerial survey images with GPS data
- Trained detection model
- MLflow server (optional)

## Step 1: Organize Survey Data

```
census_2024/
├── images/              # Survey images with GPS EXIF
│   ├── flight1/
│   ├── flight2/
│   └── ...
└── config/
    └── census.yaml
```

## Step 2: Configure Census

Create `config/census.yaml`:

```yaml
campaign:
  name: "Summer_2024_Survey"
  target_species: ["elephant", "giraffe", "zebra"]
  area_name: "Serengeti_North"
  start_date: "2024-06-01"
  end_date: "2024-06-15"

model:
  mlflow_model_name: "detector"
  mlflow_model_alias: "production"
  device: "cuda"

image_dir: "D:/census_2024/images/"

flight_specs:
  flight_height: 120.0
  gsd: 2.38

analysis:
  calculate_density: true
  detect_hotspots: true
  create_maps: true

output:
  directory: "census_results"
  generate_pdf_report: true
```

## Step 3: Run Census

```bash
cd wildetect

# Edit config
notepad config\census.yaml

# Run
scripts\run_census.bat
```

## Step 4: Review Results

```
census_results/
├── detections.json              # All detections
├── statistics.json              # Population stats
├── census_report.pdf            # PDF report
├── maps/                        # Geographic maps
│   ├── distribution_map.html
│   ├── density_heatmap.html
│   └── flight_path.html
└── visualizations/              # Annotated images
```

## Step 5: Analyze Statistics

The census generates:

- **Total counts** per species
- **Population density** (animals/km²)
- **Species distribution** analysis
- **Hotspot locations**
- **Coverage area** statistics

## Geographic Analysis

View interactive maps:

```bash
# Open in browser
explorer census_results\maps\distribution_map.html
```

Features:
- Animal locations plotted on map
- Density heatmaps
- Flight path overlay
- Filterable by species

## Generate Custom Reports

```python
from wildetect.analysis import ReportGenerator

generator = ReportGenerator("census_results/detections.json")

# Custom report
report = generator.generate_report(
    output_path="custom_report.pdf",
    include_maps=True,
    include_statistics=True,
    target_species=["elephant"]
)
```

## Example Census Output

```json
{
  "campaign": "Summer_2024_Survey",
  "survey_area": 25.5,  # km²
  "total_images": 450,
  "total_detections": 1234,
  
  "species_counts": {
    "elephant": 423,
    "giraffe": 612,
    "zebra": 199
  },
  
  "density": {
    "elephant": 16.6,  # per km²
    "giraffe": 24.0,
    "zebra": 7.8
  }
}
```

---

**Next Steps:**
- [End-to-End Detection](end-to-end-detection.md)
- [Census Configuration](../configs/wildetect/index.md#censusyaml)

