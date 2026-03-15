"""Tests for the training data export pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# These tests require rasterio and geopandas
rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from training_data_export import TrainingDataExporter, stratified_split


@pytest.fixture
def sample_data(tmp_path):
    """Create sample raster and vector label files for testing."""
    # Create a 128x128 4-band raster
    raster_path = str(tmp_path / "test_raster.tif")
    height, width = 128, 128
    transform = from_bounds(0, 0, 1, 1, width, height)

    rng = np.random.RandomState(42)
    data = (rng.rand(4, height, width) * 255).astype(np.uint8)

    with rasterio.open(
        raster_path, "w", driver="GTiff",
        height=height, width=width, count=4,
        dtype="uint8", crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    # Create vector labels
    labels_path = str(tmp_path / "labels.geojson")
    polygons = [
        {"geometry": box(0.1, 0.1, 0.4, 0.4), "class": "vegetation"},
        {"geometry": box(0.5, 0.5, 0.9, 0.9), "class": "water"},
        {"geometry": box(0.1, 0.6, 0.3, 0.8), "class": "built_up"},
    ]
    gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
    gdf.to_file(labels_path, driver="GeoJSON")

    return {
        "raster": raster_path,
        "labels": labels_path,
        "tmp_dir": str(tmp_path),
    }


class TestTrainingDataExporter:
    def test_coco_export(self, sample_data):
        output_dir = os.path.join(sample_data["tmp_dir"], "coco_output")
        exporter = TrainingDataExporter(
            raster_path=sample_data["raster"],
            labels_path=sample_data["labels"],
            output_dir=output_dir,
            tile_size=64,
            output_format="coco",
            class_column="class",
        )
        stats = exporter.run()

        assert stats["total_tiles"] > 0
        assert stats["tiles_with_labels"] > 0
        assert os.path.exists(os.path.join(output_dir, "annotations.json"))
        assert os.path.exists(os.path.join(output_dir, "metadata.json"))

        # Verify COCO format
        with open(os.path.join(output_dir, "annotations.json")) as f:
            coco = json.load(f)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["categories"]) == 3

    def test_yolo_export(self, sample_data):
        output_dir = os.path.join(sample_data["tmp_dir"], "yolo_output")
        exporter = TrainingDataExporter(
            raster_path=sample_data["raster"],
            labels_path=sample_data["labels"],
            output_dir=output_dir,
            tile_size=64,
            output_format="yolo",
            class_column="class",
        )
        stats = exporter.run()

        assert stats["tiles_with_labels"] > 0
        # Check YOLO label files exist
        label_files = os.listdir(os.path.join(output_dir, "labels"))
        assert len(label_files) > 0

        # Verify YOLO format (class x_center y_center width height)
        label_path = os.path.join(output_dir, "labels", label_files[0])
        with open(label_path) as f:
            line = f.readline().strip()
        parts = line.split()
        assert len(parts) == 5
        assert all(0 <= float(p) <= 1 for p in parts[1:])

    def test_mask_export(self, sample_data):
        output_dir = os.path.join(sample_data["tmp_dir"], "mask_output")
        exporter = TrainingDataExporter(
            raster_path=sample_data["raster"],
            labels_path=sample_data["labels"],
            output_dir=output_dir,
            tile_size=64,
            output_format="masks",
            class_column="class",
        )
        stats = exporter.run()

        assert stats["tiles_with_labels"] > 0
        mask_files = os.listdir(os.path.join(output_dir, "masks"))
        assert len(mask_files) > 0

    def test_class_map_generated(self, sample_data):
        output_dir = os.path.join(sample_data["tmp_dir"], "classmap_test")
        exporter = TrainingDataExporter(
            raster_path=sample_data["raster"],
            labels_path=sample_data["labels"],
            output_dir=output_dir,
            tile_size=64,
            output_format="coco",
            class_column="class",
        )
        stats = exporter.run()

        assert "class_map" in stats
        assert len(stats["class_map"]) == 3
        assert "vegetation" in stats["class_map"]
        assert "water" in stats["class_map"]
        assert "built_up" in stats["class_map"]

    def test_overlap_generates_more_tiles(self, sample_data):
        output_no_overlap = os.path.join(sample_data["tmp_dir"], "no_overlap")
        output_overlap = os.path.join(sample_data["tmp_dir"], "with_overlap")

        exporter1 = TrainingDataExporter(
            raster_path=sample_data["raster"],
            labels_path=sample_data["labels"],
            output_dir=output_no_overlap,
            tile_size=64, overlap=0, output_format="coco",
        )
        stats1 = exporter1.run()

        exporter2 = TrainingDataExporter(
            raster_path=sample_data["raster"],
            labels_path=sample_data["labels"],
            output_dir=output_overlap,
            tile_size=64, overlap=32, output_format="coco",
        )
        stats2 = exporter2.run()

        assert stats2["total_tiles"] > stats1["total_tiles"]
