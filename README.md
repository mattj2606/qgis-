<h1 align="center">QGIS Geospatial Analysis Toolkit</h1>

<p align="center">
  <strong>Land cover classification, spatial analytics, and AI/ML training data pipelines — built with QGIS + Python</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/QGIS-3.28%2B-green?logo=qgis&logoColor=white" alt="QGIS 3.28+">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Tests-20%20passing-brightgreen" alt="Tests passing">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</p>

---

## What This Project Does

This toolkit bridges **geospatial analysis** and **machine learning** — the exact workflow needed when building AI models that understand satellite imagery, land use, and environmental change.

It includes three production-ready components:

| Component | What it does | Key tech |
|-----------|-------------|----------|
| **Land Cover Analyzer** (QGIS Plugin) | Classifies satellite imagery into land cover types using unsupervised ML | K-Means, GMM, NDVI/NDWI/NDBI |
| **Spatial Processing Scripts** | Batch analysis, vector enrichment, multi-temporal change detection | Rasterio, GeoPandas, GDAL |
| **AI Training Data Pipeline** | Converts annotated geodata into COCO, YOLO, or segmentation mask datasets | Tiling, label export, stratified splits |

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────────┐
                    │              Satellite Imagery               │
                    │         (Sentinel-2, Landsat, etc.)          │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │         Land Cover Analyzer Plugin           │
                    │                                              │
                    │  ┌─────────────┐  ┌───────────────────────┐  │
                    │  │ Unsupervised│  │  Spectral Indices     │  │
                    │  │ Classifier  │  │  (NDVI, NDWI, NDBI)   │  │
                    │  │ K-Means /   │  │                       │  │
                    │  │ ISO Cluster │  │  Accuracy Assessment  │  │
                    │  └──────┬──────┘  └───────────┬───────────┘  │
                    └─────────┼─────────────────────┼──────────────┘
                              │                     │
               ┌──────────────▼─────────────────────▼──────────────┐
               │            Processing Scripts                     │
               │                                                   │
               │  ┌──────────────┐ ┌─────────────┐ ┌────────────┐ │
               │  │    Batch     │ │   Vector    │ │   Change   │ │
               │  │   Raster    │ │ Enrichment  │ │ Detection  │ │
               │  │  Analysis   │ │  Pipeline   │ │  Engine    │ │
               │  └──────┬───────┘ └──────┬──────┘ └─────┬──────┘ │
               └─────────┼────────────────┼──────────────┼─────────┘
                         │                │              │
               ┌─────────▼────────────────▼──────────────▼─────────┐
               │          Training Data Export Pipeline             │
               │                                                   │
               │  Raster Tiling → Label Extraction → Format Export  │
               │                                                   │
               │  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
               │  │   COCO   │  │   YOLO   │  │  Segmentation  │  │
               │  │  Format  │  │  Format  │  │     Masks      │  │
               │  └──────────┘  └──────────┘  └────────────────┘  │
               └───────────────────────────────────────────────────┘
```

---

## Repository Structure

```
qgis-/
│
├── plugin/                              # QGIS Plugin
│   └── land_cover_analyzer/
│       ├── __init__.py                  # Plugin entry point (classFactory)
│       ├── land_cover_analyzer.py       # Main plugin class — toolbar, menus, I/O
│       ├── land_cover_dialog.py         # PyQt5 dialog for user configuration
│       ├── classifier.py               # Classification engine + accuracy metrics
│       └── metadata.txt                # QGIS plugin registry metadata
│
├── scripts/                             # Standalone Processing Scripts
│   ├── batch_raster_analysis.py         # Parallel batch classification
│   ├── vector_enrichment.py             # Zonal stats, spatial joins, proximity
│   ├── change_detection.py              # Multi-temporal change analysis
│   └── training_data_export.py          # ML dataset generation (COCO/YOLO/masks)
│
├── styles/                              # QGIS Symbology
│   ├── land_cover.qml                   # Land cover palette (8-class)
│   └── ndvi_gradient.qml                # NDVI diverging color ramp
│
├── tests/                               # Test Suite
│   ├── test_classifier.py               # 15 tests: indices, classifiers, accuracy
│   └── test_training_export.py          # 5 tests: COCO, YOLO, masks, splits
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Component Deep-Dive

### 1. Land Cover Analyzer Plugin

A full QGIS plugin with GUI that classifies multispectral rasters into land cover categories.

**Capabilities:**
- **Unsupervised classification** using K-Means (MiniBatch for large rasters) or ISO Cluster (Gaussian Mixture Models)
- **Spectral index computation** — NDVI (vegetation), NDWI (water), NDBI (built-up areas)
- **Accuracy assessment** — confusion matrix, overall accuracy, Cohen's Kappa, per-class producer's/user's accuracy
- **Dual I/O backends** — uses `rasterio` when available, falls back to `GDAL` bindings for compatibility

**How it works:**
```python
# The classifier reshapes a (bands, height, width) raster into a pixel table,
# clusters it, then maps labels back to a spatial grid:

band_stack = read_all_bands()        # shape: (4, 1024, 1024)
pixels = reshape_to_table()          # shape: (1048576, 4) — each row is one pixel
labels = kmeans.fit_predict(pixels)  # cluster assignment per pixel
classified = reshape_to_grid()       # shape: (1024, 1024) — spatial class map
```

### 2. Spatial Processing Scripts

Four standalone scripts for common geospatial workflows:

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `batch_raster_analysis.py` | Classify hundreds of rasters with one command | Parallel processing via `ProcessPoolExecutor`, JSON batch reports |
| `vector_enrichment.py` | Add raster-derived statistics to vector features | Zonal stats (mean/std/min/max/median), nearest spatial joins, distance metrics, auto UTM reprojection |
| `change_detection.py` | Find what changed between two dates | Image differencing, log-ratio, NDVI differencing, transition matrices |
| `training_data_export.py` | Generate ML-ready datasets from annotated geodata | Tiling with overlap, COCO/YOLO/mask output, stratified train/val splits |

### 3. AI/ML Training Data Pipeline

The most unique component — bridges the gap between GIS annotation and ML model training:

```
Satellite Image (GeoTIFF)  +  Vector Labels (GeoJSON)
         │                            │
         ▼                            ▼
    ┌─────────┐                 ┌──────────┐
    │  Tile   │                 │  Clip    │
    │ 256x256 │                 │ labels   │
    │  chips  │                 │ to tile  │
    └────┬────┘                 └────┬─────┘
         │                           │
         ▼                           ▼
    image_0001.png            ┌──────────────┐
    image_0002.png            │ COCO .json   │
    image_0003.png            │ YOLO .txt    │
    ...                       │ Mask .png    │
                              └──────────────┘
```

**Features:**
- Configurable tile size and overlap for dense object scenes
- 2-98th percentile stretch for consistent visual normalization
- Minimum label coverage filter to skip empty tiles
- Class distribution tracking for dataset balance analysis

---

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Classify a Satellite Image
```python
from scripts.batch_raster_analysis import classify_raster

# Runs K-Means clustering on all bands, outputs a classified GeoTIFF
classify_raster(
    input_path="sentinel2_scene.tif",
    output_path="classified.tif",
    n_classes=5,
    method="kmeans"
)
```

### Generate AI Training Data
```python
from scripts.training_data_export import TrainingDataExporter

# Tiles a raster + vector labels into COCO-format training data
exporter = TrainingDataExporter(
    raster_path="sentinel2_scene.tif",
    labels_path="land_use_polygons.geojson",
    output_dir="training_data/",
    tile_size=256,
    overlap=32,            # 32px overlap for edge objects
    output_format="coco",  # also: "yolo", "masks"
    class_column="land_use"
)
stats = exporter.run()
# → training_data/images/tile_000001.png ... tile_000847.png
# → training_data/annotations.json (COCO format)
# → training_data/metadata.json (class distribution, stats)
```

### Detect Land Cover Changes
```python
from scripts.change_detection import detect_changes

changes = detect_changes(
    before="2020_composite.tif",
    after="2023_composite.tif",
    output="changes_2020_2023.tif",
    method="ndvi_change",  # compares vegetation index between dates
    threshold=0.15
)
print(f"{changes['change_percentage']}% of the area changed")
print(f"  Gain: {changes['gain_pixels']} pixels")
print(f"  Loss: {changes['loss_pixels']} pixels")
```

### Enrich Vectors with Raster Statistics
```python
from scripts.vector_enrichment import zonal_statistics, compute_area_perimeter

# Add elevation stats to each parcel polygon
parcels = zonal_statistics(
    vectors_path="parcels.geojson",
    raster_path="dem.tif",
    stats=["mean", "std", "min", "max"],
    prefix="elevation"
)
# Result: each parcel now has elevation_mean, elevation_std, etc.

# Add geometric metrics (auto-reprojects to UTM for accuracy)
parcels = compute_area_perimeter(parcels)
# Result: area_m2, area_ha, perimeter_m, compactness columns
```

### Install the QGIS Plugin
```bash
# Linux
cp -r plugin/land_cover_analyzer \
  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

# macOS
cp -r plugin/land_cover_analyzer \
  ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/

# Windows
xcopy plugin\land_cover_analyzer \
  %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\land_cover_analyzer /E /I
```

Then enable "Land Cover Analyzer" in QGIS > Plugins > Manage and Install Plugins.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just the classifier tests (no geo dependencies needed beyond numpy + sklearn)
pytest tests/test_classifier.py -v

# Training data pipeline tests (requires rasterio + geopandas)
pytest tests/test_training_export.py -v
```

```
tests/test_classifier.py ........... 15 passed
tests/test_training_export.py ..... 5 passed
================================================
20 passed
```

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **GIS Platform** | QGIS 3.28+ (LTR), QGIS Processing Framework |
| **Raster I/O** | Rasterio, GDAL |
| **Vector Analysis** | GeoPandas, Shapely, Fiona |
| **Machine Learning** | scikit-learn (K-Means, GMM), scikit-image |
| **Coordinate Systems** | pyproj, automatic UTM zone detection |
| **Plugin Framework** | PyQt5, QGIS Plugin API |
| **ML Formats** | COCO JSON, YOLO txt, semantic segmentation masks |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
