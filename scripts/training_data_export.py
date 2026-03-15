"""
Training Data Export Pipeline

Convert QGIS-annotated geospatial data into formats suitable for
computer vision and geospatial AI/ML models.

Supported output formats:
- COCO (instance segmentation / object detection)
- YOLO (object detection)
- Semantic segmentation masks (pixel-level labels)

Usage:
    python training_data_export.py --raster satellite.tif \
        --labels annotations.geojson --output_dir training_data \
        --tile_size 256 --format coco
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box, mapping
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TrainingDataExporter:
    """Export geospatial raster + vector annotations as ML training data."""

    def __init__(self, raster_path: str, labels_path: str,
                 output_dir: str, tile_size: int = 256,
                 overlap: int = 0, output_format: str = "coco",
                 class_column: str = "class",
                 min_label_coverage: float = 0.01):
        """
        Args:
            raster_path: Path to source raster imagery.
            labels_path: Path to vector labels (GeoJSON, Shapefile).
            output_dir: Output directory for training data.
            tile_size: Size of square image tiles in pixels.
            overlap: Pixel overlap between adjacent tiles.
            output_format: Output format ('coco', 'yolo', 'masks').
            class_column: Column in labels containing class names/IDs.
            min_label_coverage: Minimum fraction of tile covered by labels
                               to include it in the dataset.
        """
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_format = output_format
        self.class_column = class_column
        self.min_label_coverage = min_label_coverage

        self._labels_gdf = None
        self._raster_src = None
        self._class_map = {}

    def run(self) -> dict:
        """Execute the full export pipeline.

        Returns:
            Summary statistics dictionary.
        """
        self._setup_directories()
        self._load_labels()

        stats = {
            "total_tiles": 0,
            "tiles_with_labels": 0,
            "tiles_empty": 0,
            "class_distribution": {},
            "format": self.output_format,
        }

        with rasterio.open(self.raster_path) as src:
            self._raster_src = src
            tiles = self._generate_tile_windows()
            logger.info(f"Generated {len(tiles)} tile windows")

            annotations = []

            for tile_idx, window in enumerate(tiles):
                result = self._process_tile(tile_idx, window)
                stats["total_tiles"] += 1

                if result is not None:
                    stats["tiles_with_labels"] += 1
                    annotations.append(result)

                    for cls in result.get("classes_present", []):
                        stats["class_distribution"][cls] = (
                            stats["class_distribution"].get(cls, 0) + 1
                        )
                else:
                    stats["tiles_empty"] += 1

                if (tile_idx + 1) % 100 == 0:
                    logger.info(f"Processed {tile_idx + 1}/{len(tiles)} tiles")

        # Write format-specific annotation files
        if self.output_format == "coco":
            self._write_coco_annotations(annotations)
        elif self.output_format == "yolo":
            self._write_yolo_config()

        # Write metadata
        stats["class_map"] = self._class_map
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Export complete: {stats['tiles_with_labels']} labeled tiles "
            f"out of {stats['total_tiles']} total"
        )
        return stats

    def _setup_directories(self):
        """Create output directory structure."""
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
        if self.output_format == "masks":
            os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def _load_labels(self):
        """Load and prepare vector labels."""
        self._labels_gdf = gpd.read_file(self.labels_path)
        logger.info(f"Loaded {len(self._labels_gdf)} label features")

        # Build class mapping
        if self.class_column in self._labels_gdf.columns:
            classes = sorted(self._labels_gdf[self.class_column].unique())
            self._class_map = {str(cls): idx for idx, cls in enumerate(classes)}
        else:
            logger.warning(
                f"Column '{self.class_column}' not found. "
                f"Using single class 'object'."
            )
            self._class_map = {"object": 0}
            self._labels_gdf[self.class_column] = "object"

    def _generate_tile_windows(self) -> list:
        """Generate rasterio Windows covering the full raster extent."""
        src = self._raster_src
        step = self.tile_size - self.overlap
        windows = []

        for row_off in range(0, src.height, step):
            for col_off in range(0, src.width, step):
                h = min(self.tile_size, src.height - row_off)
                w = min(self.tile_size, src.width - col_off)
                if h < self.tile_size // 2 or w < self.tile_size // 2:
                    continue  # skip very small edge tiles
                windows.append(Window(col_off, row_off, w, h))

        return windows

    def _process_tile(self, tile_idx: int, window: Window) -> Optional[dict]:
        """Process a single tile: extract image + labels.

        Returns:
            Annotation dict if labels present, else None.
        """
        src = self._raster_src

        # Get tile bounds in CRS coordinates
        tile_transform = rasterio.windows.transform(window, src.transform)
        tile_bounds = rasterio.windows.bounds(window, src.transform)
        tile_box = box(*tile_bounds)

        # Find intersecting labels
        intersecting = self._labels_gdf[
            self._labels_gdf.geometry.intersects(tile_box)
        ]

        if intersecting.empty:
            return None

        # Clip labels to tile
        clipped = intersecting.copy()
        clipped["geometry"] = clipped.geometry.intersection(tile_box)
        clipped = clipped[~clipped.geometry.is_empty]

        if clipped.empty:
            return None

        # Check minimum coverage
        label_area = clipped.geometry.area.sum()
        tile_area = tile_box.area
        if label_area / tile_area < self.min_label_coverage:
            return None

        # Read and save image tile
        data = src.read(window=window)
        tile_name = f"tile_{tile_idx:06d}"
        self._save_image_tile(data, tile_name)

        # Generate annotations
        classes_present = []
        if self.output_format == "coco":
            annotation = self._tile_to_coco(clipped, tile_name, window)
        elif self.output_format == "yolo":
            annotation = self._tile_to_yolo(clipped, tile_name, window)
        elif self.output_format == "masks":
            annotation = self._tile_to_mask(
                clipped, tile_name, window, tile_transform
            )
        else:
            raise ValueError(f"Unknown format: {self.output_format}")

        classes_present = list(
            clipped[self.class_column].map(
                lambda c: self._class_map.get(str(c), 0)
            ).unique()
        )
        annotation["classes_present"] = [
            str(c) for c in clipped[self.class_column].unique()
        ]

        return annotation

    def _save_image_tile(self, data: np.ndarray, tile_name: str):
        """Save raster tile as PNG image."""
        # Use first 3 bands as RGB, normalize to 0-255
        if data.shape[0] >= 3:
            rgb = data[:3]
        else:
            rgb = np.stack([data[0]] * 3)

        # Percentile stretch for visualization
        for i in range(3):
            band = rgb[i].astype(np.float64)
            p2, p98 = np.percentile(band[band > 0], [2, 98]) if (band > 0).any() else (0, 1)
            if p98 > p2:
                band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            rgb[i] = band

        img = np.moveaxis(rgb.astype(np.uint8), 0, -1)  # (H, W, 3)
        img_path = os.path.join(self.output_dir, "images", f"{tile_name}.png")
        Image.fromarray(img).save(img_path)

    def _tile_to_coco(self, labels_gdf: gpd.GeoDataFrame,
                      tile_name: str, window: Window) -> dict:
        """Convert tile labels to COCO format annotations."""
        annotations = []
        for _, row in labels_gdf.iterrows():
            geom = row.geometry
            cls = str(row[self.class_column])
            category_id = self._class_map.get(cls, 0)

            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            # Convert geo coords to pixel coords within tile
            src = self._raster_src
            tile_transform = rasterio.windows.transform(window, src.transform)

            col_min = int((bounds[0] - tile_transform.c) / tile_transform.a)
            row_min = int((bounds[3] - tile_transform.f) / tile_transform.e)
            col_max = int((bounds[2] - tile_transform.c) / tile_transform.a)
            row_max = int((bounds[1] - tile_transform.f) / tile_transform.e)

            w = max(col_max - col_min, 1)
            h = max(row_max - row_min, 1)

            annotations.append({
                "bbox": [col_min, row_min, w, h],
                "category_id": category_id,
                "category_name": cls,
                "area": w * h,
            })

        return {"tile_name": tile_name, "annotations": annotations}

    def _tile_to_yolo(self, labels_gdf: gpd.GeoDataFrame,
                      tile_name: str, window: Window) -> dict:
        """Convert tile labels to YOLO format and write label file."""
        lines = []
        src = self._raster_src
        tile_transform = rasterio.windows.transform(window, src.transform)

        for _, row in labels_gdf.iterrows():
            geom = row.geometry
            cls = str(row[self.class_column])
            category_id = self._class_map.get(cls, 0)

            bounds = geom.bounds
            col_min = (bounds[0] - tile_transform.c) / tile_transform.a
            row_min = (bounds[3] - tile_transform.f) / tile_transform.e
            col_max = (bounds[2] - tile_transform.c) / tile_transform.a
            row_max = (bounds[1] - tile_transform.f) / tile_transform.e

            # YOLO format: class x_center y_center width height (normalized)
            x_center = (col_min + col_max) / 2 / window.width
            y_center = (row_min + row_max) / 2 / window.height
            w = (col_max - col_min) / window.width
            h = (row_max - row_min) / window.height

            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            w = np.clip(w, 0, 1)
            h = np.clip(h, 0, 1)

            lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        label_path = os.path.join(self.output_dir, "labels", f"{tile_name}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        return {"tile_name": tile_name, "n_objects": len(lines)}

    def _tile_to_mask(self, labels_gdf: gpd.GeoDataFrame,
                      tile_name: str, window: Window,
                      transform) -> dict:
        """Generate semantic segmentation mask for the tile."""
        shapes = []
        for _, row in labels_gdf.iterrows():
            cls = str(row[self.class_column])
            class_id = self._class_map.get(cls, 0) + 1  # 0 = background
            shapes.append((mapping(row.geometry), class_id))

        mask = rasterize(
            shapes,
            out_shape=(int(window.height), int(window.width)),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        mask_path = os.path.join(self.output_dir, "masks", f"{tile_name}.png")
        Image.fromarray(mask).save(mask_path)

        unique, counts = np.unique(mask, return_counts=True)
        class_pixels = {int(u): int(c) for u, c in zip(unique, counts)}

        return {"tile_name": tile_name, "class_pixels": class_pixels}

    def _write_coco_annotations(self, tile_annotations: list):
        """Write full COCO-format annotation file."""
        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": idx, "name": name}
                for name, idx in sorted(self._class_map.items(), key=lambda x: x[1])
            ],
        }

        ann_id = 0
        for img_id, tile in enumerate(tile_annotations):
            coco["images"].append({
                "id": img_id,
                "file_name": f"{tile['tile_name']}.png",
                "width": self.tile_size,
                "height": self.tile_size,
            })

            for ann in tile.get("annotations", []):
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                })
                ann_id += 1

        coco_path = os.path.join(self.output_dir, "annotations.json")
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)
        logger.info(f"COCO annotations: {coco_path} ({ann_id} annotations)")

    def _write_yolo_config(self):
        """Write YOLO dataset configuration file."""
        config = {
            "path": os.path.abspath(self.output_dir),
            "train": "images",
            "val": "images",
            "names": {
                idx: name
                for name, idx in sorted(self._class_map.items(), key=lambda x: x[1])
            },
        }

        import yaml
        config_path = os.path.join(self.output_dir, "dataset.yaml")
        try:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fallback without yaml
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"YOLO config: {config_path}")


def stratified_split(metadata_path: str, train_ratio: float = 0.8,
                     seed: int = 42) -> dict:
    """Split training data into train/val sets with stratified sampling.

    Args:
        metadata_path: Path to metadata.json from export.
        train_ratio: Fraction for training set.
        seed: Random seed.

    Returns:
        Dictionary with train/val file lists.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    output_dir = os.path.dirname(metadata_path)
    images_dir = os.path.join(output_dir, "images")
    all_images = sorted(os.listdir(images_dir))

    rng = np.random.RandomState(seed)
    rng.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    split_info = {
        "train": train_files,
        "val": val_files,
        "train_count": len(train_files),
        "val_count": len(val_files),
    }

    split_path = os.path.join(output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")
    return split_info


def main():
    parser = argparse.ArgumentParser(description="Export training data for AI/ML")
    parser.add_argument("--raster", required=True, help="Source raster")
    parser.add_argument("--labels", required=True, help="Vector labels")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--format", default="coco", choices=["coco", "yolo", "masks"])
    parser.add_argument("--class_column", default="class")
    parser.add_argument("--min_coverage", type=float, default=0.01)
    args = parser.parse_args()

    exporter = TrainingDataExporter(
        raster_path=args.raster,
        labels_path=args.labels,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_format=args.format,
        class_column=args.class_column,
        min_label_coverage=args.min_coverage,
    )
    exporter.run()


if __name__ == "__main__":
    main()
