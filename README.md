# QGIS Geospatial Analysis Toolkit

A collection of QGIS tools, plugins, and processing scripts for geospatial analysis with a focus on generating high-quality training data for AI/ML models.

## Project Overview

This toolkit demonstrates end-to-end geospatial workflows:

1. **Land Cover Analyzer Plugin** - A QGIS plugin that classifies raster imagery into land cover categories and exports labeled datasets suitable for machine learning
2. **Spatial Data Processing Scripts** - Automated pipelines for cleaning, transforming, and analyzing vector/raster data using QGIS Processing and GDAL
3. **AI Training Data Generator** - Tools to convert QGIS-annotated geospatial data into formats consumed by computer vision and geospatial AI models (COCO, YOLO, GeoJSON tiles)

## Repository Structure

```
qgis-/
├── plugin/
│   └── land_cover_analyzer/   # QGIS plugin for land cover classification
│       ├── __init__.py
│       ├── land_cover_analyzer.py
│       ├── land_cover_dialog.py
│       ├── classifier.py
│       └── metadata.txt
├── scripts/
│   ├── batch_raster_analysis.py    # Batch processing of raster datasets
│   ├── vector_enrichment.py        # Enrich vector layers with spatial statistics
│   ├── change_detection.py         # Multi-temporal change detection
│   └── training_data_export.py     # Export annotated data for AI/ML
├── styles/
│   ├── land_cover.qml              # QGIS style for land cover visualization
│   └── ndvi_gradient.qml           # NDVI color ramp style
├── tests/
│   ├── test_classifier.py
│   └── test_training_export.py
├── data/
│   └── sample/                     # Small sample datasets for testing
├── requirements.txt
└── README.md
```

## Key Features

### Land Cover Analyzer Plugin
- Unsupervised classification (K-Means, ISO Cluster) of multispectral imagery
- Spectral index computation (NDVI, NDWI, NDBI)
- Accuracy assessment with confusion matrix generation
- Direct export of classified maps as GeoTIFF with metadata

### Spatial Processing Scripts
- **Batch Raster Analysis**: Process hundreds of raster tiles with consistent parameters
- **Vector Enrichment**: Compute zonal statistics, spatial joins, and proximity metrics
- **Change Detection**: Compare multi-date imagery to identify land use transitions
- **Training Data Export**: Generate image chips and labels in COCO/YOLO format

### AI/ML Integration
- Tile large rasters into fixed-size chips aligned to a grid
- Generate pixel-level masks or bounding box annotations from vector labels
- Stratified sampling to ensure balanced class representation
- Metadata tracking with spatial reference preservation

## Requirements

- QGIS 3.28+ (LTR) or QGIS 3.34+
- Python 3.9+
- See `requirements.txt` for Python dependencies

## Installation

### Plugin
```bash
# Copy plugin to QGIS plugin directory
cp -r plugin/land_cover_analyzer ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

# Or on macOS:
cp -r plugin/land_cover_analyzer ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/
```

### Scripts
```bash
pip install -r requirements.txt
# Scripts can be run standalone or within the QGIS Python console
```

## Usage Examples

### Classify a Satellite Image
```python
from scripts.batch_raster_analysis import classify_raster

classify_raster(
    input_path="data/sample/sentinel2_rgb.tif",
    output_path="output/classified.tif",
    n_classes=5,
    method="kmeans"
)
```

### Generate AI Training Data
```python
from scripts.training_data_export import TrainingDataExporter

exporter = TrainingDataExporter(
    raster_path="data/sample/sentinel2_rgb.tif",
    labels_path="data/sample/labels.geojson",
    output_dir="output/training_data",
    tile_size=256,
    output_format="coco"
)
exporter.run()
```

### Compute Change Detection
```python
from scripts.change_detection import detect_changes

changes = detect_changes(
    before="data/2020_composite.tif",
    after="data/2023_composite.tif",
    method="difference",
    threshold=0.3
)
```

## License

MIT License
