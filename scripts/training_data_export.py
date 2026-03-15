"""
AI/ML Training Data Export Pipeline
=====================================

The bridge between GIS annotation and machine learning model training.

This pipeline takes a satellite image (GeoTIFF) and vector annotations
(GeoJSON/Shapefile) created in QGIS, and converts them into training data
formats that ML frameworks expect:

- **COCO**: JSON annotations with bounding boxes for object detection
            (used by Detectron2, MMDetection, etc.)
- **YOLO**: Per-image text files with normalized bounding boxes
            (used by Ultralytics YOLOv5/v8, etc.)
- **Masks**: Pixel-level segmentation masks as PNG images
            (used by U-Net, DeepLab, SegFormer, etc.)

The pipeline handles the hard parts of geo-to-ML conversion:
    1. Tiling large rasters into fixed-size chips (e.g., 256x256)
    2. Clipping vector labels to each tile
    3. Converting geographic coordinates to pixel coordinates
    4. Normalizing imagery with percentile stretch
    5. Filtering empty tiles and tracking class balance

Usage:
    python training_data_export.py --raster satellite.tif \\
        --labels annotations.geojson --output_dir training_data \\
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
    """Export geospatial raster + vector annotations as ML training data.

    Workflow overview:
        1. Load raster and vector labels
        2. Generate a grid of tile windows covering the full raster
        3. For each tile:
           a. Check if any labels intersect this tile
           b. Skip tiles with insufficient label coverage (reduce noise)
           c. Extract and normalize the image chip as PNG
           d. Convert labels to the target format (COCO/YOLO/mask)
        4. Write format-specific annotation files
        5. Generate metadata with class distribution stats
    """

    def __init__(self, raster_path: str, labels_path: str,
                 output_dir: str, tile_size: int = 256,
                 overlap: int = 0, output_format: str = "coco",
                 class_column: str = "class",
                 min_label_coverage: float = 0.01):
        """
        Args:
            raster_path: Path to source raster imagery (GeoTIFF, etc.).
            labels_path: Path to vector labels (GeoJSON, Shapefile, GeoPackage).
            output_dir: Output directory. Will create images/, labels/, masks/ subdirs.
            tile_size: Width and height of each image chip in pixels (default 256).
                       Common values: 256, 512, 1024.
            overlap: Pixel overlap between adjacent tiles (default 0).
                     Use overlap > 0 when objects near tile edges need to be
                     fully captured. 32-64px is typical.
            output_format: 'coco', 'yolo', or 'masks'.
            class_column: Column in the labels file containing class names/IDs.
                          e.g., "land_use", "class", "type".
            min_label_coverage: Minimum fraction of tile area covered by labels
                                to include it in the dataset (default 0.01 = 1%).
                                Helps filter tiles that barely touch a label edge.
        """
        self.raster_path = raster_path
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_format = output_format
        self.class_column = class_column
        self.min_label_coverage = min_label_coverage

        # Internal state — populated during run()
        self._labels_gdf = None    # Loaded vector labels as GeoDataFrame
        self._raster_src = None    # Open rasterio dataset (during processing)
        self._class_map = {}       # Mapping: class_name → integer_id

    # =========================================================================
    # Main Pipeline
    # =========================================================================

    def run(self) -> dict:
        """Execute the full export pipeline.

        Returns:
            Summary statistics dictionary with tile counts, class distribution,
            and the class name → ID mapping.
        """
        self._setup_directories()
        self._load_labels()

        # Initialize statistics tracker
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

                    # Track how many tiles contain each class
                    for cls in result.get("classes_present", []):
                        stats["class_distribution"][cls] = (
                            stats["class_distribution"].get(cls, 0) + 1
                        )
                else:
                    stats["tiles_empty"] += 1

                # Progress logging every 100 tiles
                if (tile_idx + 1) % 100 == 0:
                    logger.info(f"Processed {tile_idx + 1}/{len(tiles)} tiles")

        # Write format-specific annotation files
        if self.output_format == "coco":
            self._write_coco_annotations(annotations)
        elif self.output_format == "yolo":
            self._write_yolo_config()

        # Save metadata — useful for inspecting dataset balance and debugging
        stats["class_map"] = self._class_map
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Export complete: {stats['tiles_with_labels']} labeled tiles "
            f"out of {stats['total_tiles']} total"
        )
        return stats

    # =========================================================================
    # Setup & Loading
    # =========================================================================

    def _setup_directories(self):
        """Create the output directory structure."""
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
        if self.output_format == "masks":
            os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def _load_labels(self):
        """Load vector labels and build the class name → integer ID mapping.

        The class mapping is deterministic (sorted alphabetically) so that
        the same labels always get the same IDs across runs.
        """
        self._labels_gdf = gpd.read_file(self.labels_path)
        logger.info(f"Loaded {len(self._labels_gdf)} label features")

        # Build class mapping from unique values in the class column
        if self.class_column in self._labels_gdf.columns:
            classes = sorted(self._labels_gdf[self.class_column].unique())
            self._class_map = {str(cls): idx for idx, cls in enumerate(classes)}
        else:
            # Fallback: if the specified column doesn't exist, treat all labels
            # as a single "object" class. This is useful for binary detection.
            logger.warning(
                f"Column '{self.class_column}' not found. "
                f"Using single class 'object'."
            )
            self._class_map = {"object": 0}
            self._labels_gdf[self.class_column] = "object"

    # =========================================================================
    # Tiling
    # =========================================================================

    def _generate_tile_windows(self) -> list:
        """Generate a grid of rasterio Window objects covering the full raster.

        The step size is (tile_size - overlap), so with overlap=32 and
        tile_size=256, tiles advance by 224 pixels. Edge tiles that are
        smaller than half the tile size are discarded.

        Returns:
            List of rasterio.windows.Window objects.
        """
        src = self._raster_src
        step = self.tile_size - self.overlap
        windows = []

        for row_off in range(0, src.height, step):
            for col_off in range(0, src.width, step):
                # Clamp to raster bounds for edge tiles
                h = min(self.tile_size, src.height - row_off)
                w = min(self.tile_size, src.width - col_off)
                # Skip very small edge tiles — they produce low-quality training data
                if h < self.tile_size // 2 or w < self.tile_size // 2:
                    continue
                windows.append(Window(col_off, row_off, w, h))

        return windows

    # =========================================================================
    # Per-Tile Processing
    # =========================================================================

    def _process_tile(self, tile_idx: int, window: Window) -> Optional[dict]:
        """Process a single tile: check for labels, extract image, generate annotations.

        Args:
            tile_idx: Index of this tile (used for naming).
            window: Rasterio window defining the tile's pixel extent.

        Returns:
            Annotation dict if labels are present and meet coverage threshold,
            else None (tile is skipped).
        """
        src = self._raster_src

        # Convert pixel window to geographic coordinates for spatial querying
        tile_transform = rasterio.windows.transform(window, src.transform)
        tile_bounds = rasterio.windows.bounds(window, src.transform)
        tile_box = box(*tile_bounds)  # Shapely polygon for the tile extent

        # --- Spatial query: find labels that intersect this tile ---
        intersecting = self._labels_gdf[
            self._labels_gdf.geometry.intersects(tile_box)
        ]

        if intersecting.empty:
            return None  # No labels in this tile

        # Clip label geometries to the tile boundary (labels may extend beyond)
        clipped = intersecting.copy()
        clipped["geometry"] = clipped.geometry.intersection(tile_box)
        clipped = clipped[~clipped.geometry.is_empty]

        if clipped.empty:
            return None

        # --- Coverage filter ---
        # Skip tiles where labels cover only a tiny fraction of the tile area.
        # This prevents training on tiles with just a sliver of a polygon edge.
        label_area = clipped.geometry.area.sum()
        tile_area = tile_box.area
        if label_area / tile_area < self.min_label_coverage:
            return None

        # --- Extract and save image tile ---
        data = src.read(window=window)
        tile_name = f"tile_{tile_idx:06d}"
        self._save_image_tile(data, tile_name)

        # --- Generate format-specific annotations ---
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

        # Track which classes are present in this tile
        annotation["classes_present"] = [
            str(c) for c in clipped[self.class_column].unique()
        ]

        return annotation

    # =========================================================================
    # Image Export
    # =========================================================================

    def _save_image_tile(self, data: np.ndarray, tile_name: str):
        """Save a raster tile as a normalized PNG image.

        Satellite imagery typically has 12-16 bit values that need to be
        stretched to 0-255 for visualization and model input. We use a
        2nd-98th percentile stretch, which:
        - Removes extreme outliers (sensor artifacts, clouds)
        - Maximizes contrast for the actual ground features
        - Produces visually consistent tiles across the dataset

        Args:
            data: Raster data array of shape (bands, height, width).
            tile_name: Base name for the output file.
        """
        # Use first 3 bands as RGB (or triplicate if single-band)
        if data.shape[0] >= 3:
            rgb = data[:3]
        else:
            rgb = np.stack([data[0]] * 3)

        # 2-98th percentile stretch per band
        for i in range(3):
            band = rgb[i].astype(np.float64)
            # Only compute percentiles from non-zero pixels (zero = nodata)
            p2, p98 = np.percentile(band[band > 0], [2, 98]) if (band > 0).any() else (0, 1)
            if p98 > p2:
                band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            rgb[i] = band

        # Convert from (bands, H, W) to (H, W, bands) for PIL
        img = np.moveaxis(rgb.astype(np.uint8), 0, -1)
        img_path = os.path.join(self.output_dir, "images", f"{tile_name}.png")
        Image.fromarray(img).save(img_path)

    # =========================================================================
    # COCO Format
    # =========================================================================

    def _tile_to_coco(self, labels_gdf: gpd.GeoDataFrame,
                      tile_name: str, window: Window) -> dict:
        """Convert tile labels to COCO format bounding box annotations.

        COCO format uses [x_min, y_min, width, height] in pixel coordinates.
        This is the standard for object detection benchmarks.
        """
        annotations = []
        for _, row in labels_gdf.iterrows():
            geom = row.geometry
            cls = str(row[self.class_column])
            category_id = self._class_map.get(cls, 0)

            # Convert geographic bounds to pixel coordinates within this tile
            bounds = geom.bounds  # (minx, miny, maxx, maxy) in CRS coords
            src = self._raster_src
            tile_transform = rasterio.windows.transform(window, src.transform)

            # Affine transform: geo_x = a * col + c  →  col = (geo_x - c) / a
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

    def _write_coco_annotations(self, tile_annotations: list):
        """Write the consolidated COCO annotations.json file.

        COCO format structure:
        {
            "images": [{"id": 0, "file_name": "tile_000001.png", ...}],
            "annotations": [{"id": 0, "image_id": 0, "bbox": [...], ...}],
            "categories": [{"id": 0, "name": "vegetation"}, ...]
        }
        """
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
                    "iscrowd": 0,  # Required by COCO spec: 0 = individual object
                })
                ann_id += 1

        coco_path = os.path.join(self.output_dir, "annotations.json")
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)
        logger.info(f"COCO annotations: {coco_path} ({ann_id} annotations)")

    # =========================================================================
    # YOLO Format
    # =========================================================================

    def _tile_to_yolo(self, labels_gdf: gpd.GeoDataFrame,
                      tile_name: str, window: Window) -> dict:
        """Convert tile labels to YOLO format and write per-tile label file.

        YOLO format: one .txt file per image, each line is:
            class_id  x_center  y_center  width  height

        All coordinates are normalized to [0, 1] relative to image dimensions.
        """
        lines = []
        src = self._raster_src
        tile_transform = rasterio.windows.transform(window, src.transform)

        for _, row in labels_gdf.iterrows():
            geom = row.geometry
            cls = str(row[self.class_column])
            category_id = self._class_map.get(cls, 0)

            # Convert geo bounds → pixel coords → normalized coords
            bounds = geom.bounds
            col_min = (bounds[0] - tile_transform.c) / tile_transform.a
            row_min = (bounds[3] - tile_transform.f) / tile_transform.e
            col_max = (bounds[2] - tile_transform.c) / tile_transform.a
            row_max = (bounds[1] - tile_transform.f) / tile_transform.e

            # YOLO uses center + dimensions, normalized by image size
            x_center = (col_min + col_max) / 2 / window.width
            y_center = (row_min + row_max) / 2 / window.height
            w = (col_max - col_min) / window.width
            h = (row_max - row_min) / window.height

            # Clip to [0, 1] — labels at tile edges may slightly overflow
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            w = np.clip(w, 0, 1)
            h = np.clip(h, 0, 1)

            lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Write one label file per image tile
        label_path = os.path.join(self.output_dir, "labels", f"{tile_name}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        return {"tile_name": tile_name, "n_objects": len(lines)}

    def _write_yolo_config(self):
        """Write dataset.yaml config file for YOLO training.

        This file tells YOLO where the data is and what the class names are.
        Compatible with Ultralytics YOLOv5/v8 training scripts.
        """
        config = {
            "path": os.path.abspath(self.output_dir),
            "train": "images",
            "val": "images",
            "names": {
                idx: name
                for name, idx in sorted(self._class_map.items(), key=lambda x: x[1])
            },
        }

        config_path = os.path.join(self.output_dir, "dataset.yaml")
        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if PyYAML not installed
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"YOLO config: {config_path}")

    # =========================================================================
    # Segmentation Masks
    # =========================================================================

    def _tile_to_mask(self, labels_gdf: gpd.GeoDataFrame,
                      tile_name: str, window: Window,
                      transform) -> dict:
        """Generate a pixel-level semantic segmentation mask for a tile.

        The mask is a single-channel PNG where:
        - 0 = background (no label)
        - 1, 2, 3, ... = class IDs (shifted by 1 from the class map)

        This format is directly compatible with PyTorch/TensorFlow segmentation
        loss functions (CrossEntropyLoss, etc.).
        """
        # Build (geometry, class_id) pairs for rasterization
        shapes = []
        for _, row in labels_gdf.iterrows():
            cls = str(row[self.class_column])
            class_id = self._class_map.get(cls, 0) + 1  # +1 so background stays 0
            shapes.append((mapping(row.geometry), class_id))

        # Rasterize vector geometries into a pixel grid
        mask = rasterize(
            shapes,
            out_shape=(int(window.height), int(window.width)),
            transform=transform,
            fill=0,        # Background value
            dtype=np.uint8,
        )

        # Save mask as PNG
        mask_path = os.path.join(self.output_dir, "masks", f"{tile_name}.png")
        Image.fromarray(mask).save(mask_path)

        # Track pixel counts per class (useful for class weighting in training)
        unique, counts = np.unique(mask, return_counts=True)
        class_pixels = {int(u): int(c) for u, c in zip(unique, counts)}

        return {"tile_name": tile_name, "class_pixels": class_pixels}


# =============================================================================
# Dataset Splitting
# =============================================================================

def stratified_split(metadata_path: str, train_ratio: float = 0.8,
                     seed: int = 42) -> dict:
    """Split exported training data into train/validation sets.

    Uses random shuffling with a fixed seed for reproducibility.
    For more sophisticated stratification (by class distribution),
    extend this function to read the annotations and balance by class.

    Args:
        metadata_path: Path to metadata.json from a previous export run.
        train_ratio: Fraction of tiles for training (default 0.8 = 80%).
        seed: Random seed for reproducible splits.

    Returns:
        Dictionary with train/val file lists, saved to split.json.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    output_dir = os.path.dirname(metadata_path)
    images_dir = os.path.join(output_dir, "images")
    all_images = sorted(os.listdir(images_dir))

    # Reproducible shuffle
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


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for training data export."""
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
