"""
Batch Raster Analysis
======================

Process multiple raster files in a directory with consistent classification
parameters. Supports parallel execution for high throughput.

This script is designed for large-scale workflows where you need to classify
dozens or hundreds of satellite image tiles with the same settings — e.g.,
processing a full Sentinel-2 granule that's been split into tiles.

Features:
    - Parallel classification using ProcessPoolExecutor
    - Spectral index computation (NDVI, NDWI) per tile
    - Class distribution statistics for each output
    - Consolidated JSON batch report

Can run standalone (CLI) or be imported as a library.

Usage:
    python batch_raster_analysis.py --input_dir ./data --output_dir ./output \\
        --n_classes 5 --method kmeans --workers 4
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Import the classification engine from the plugin package.
# We add it to sys.path so this script works both standalone and when
# called from within the QGIS Python console.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin" / "land_cover_analyzer"))
from classifier import UnsupervisedClassifier, SpectralIndices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Single Raster Classification
# =============================================================================

def classify_raster(input_path: str, output_path: str,
                    n_classes: int = 5, method: str = "kmeans") -> dict:
    """Classify a single multispectral raster into land cover classes.

    Reads all bands, runs unsupervised classification, writes the result
    as a compressed GeoTIFF, and returns metadata including class distribution.

    Args:
        input_path: Path to input raster (GeoTIFF, JPEG2000, etc.).
        output_path: Path for the classified output GeoTIFF.
        n_classes: Number of land cover classes to identify (default 5).
        method: Clustering method — 'kmeans' (fast) or 'iso_cluster' (flexible).

    Returns:
        Dictionary with metadata:
        {
            "input": "/path/to/input.tif",
            "output": "/path/to/output.tif",
            "crs": "EPSG:32633",
            "bounds": [xmin, ymin, xmax, ymax],
            "shape": [bands, height, width],
            "n_classes": 5,
            "method": "kmeans",
            "class_distribution": {1: {"count": 50000, "percentage": 23.84}, ...}
        }
    """
    logger.info(f"Processing: {input_path}")

    # Read all bands into a (bands, height, width) numpy array
    with rasterio.open(input_path) as src:
        band_stack = src.read().astype(np.float32)
        profile = src.profile.copy()  # Save raster metadata for output
        meta = {
            "input": input_path,
            "crs": str(src.crs),
            "bounds": list(src.bounds),
            "shape": list(band_stack.shape),
        }

    # Run unsupervised classification
    classifier = UnsupervisedClassifier(n_classes=n_classes, method=method)
    classified = classifier.fit_predict(band_stack)

    # Update the raster profile for single-band integer output
    profile.update(
        dtype=rasterio.int32,
        count=1,           # Single band (class labels)
        compress="lzw",    # Lossless compression
        nodata=0,          # 0 = nodata / unclassified
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write classified raster
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classified, 1)

    # Compute class distribution statistics.
    # This helps assess whether classes are balanced or dominated by one type.
    unique, counts = np.unique(classified, return_counts=True)
    total_pixels = classified.size
    class_dist = {
        int(cls): {"count": int(cnt), "percentage": round(cnt / total_pixels * 100, 2)}
        for cls, cnt in zip(unique, counts)
        if cls > 0  # Skip nodata (class 0)
    }

    meta.update({
        "output": output_path,
        "n_classes": n_classes,
        "method": method,
        "class_distribution": class_dist,
    })

    logger.info(f"Classified: {output_path} ({n_classes} classes)")
    return meta


# =============================================================================
# Spectral Index Computation
# =============================================================================

def compute_indices(input_path: str, output_dir: str,
                    red_band: int = 3, nir_band: int = 4,
                    green_band: int = 2) -> list:
    """Compute spectral indices (NDVI, NDWI) for a multispectral raster.

    Outputs are saved as single-band float GeoTIFFs in the specified directory.

    Args:
        input_path: Path to multispectral raster.
        output_dir: Directory for output index rasters.
        red_band: 1-based band number for the red channel (default 3).
        nir_band: 1-based band number for near-infrared (default 4).
        green_band: 1-based band number for the green channel (default 2).

    Returns:
        List of output file paths (e.g., ["scene_ndvi.tif", "scene_ndwi.tif"]).
    """
    indices = SpectralIndices()
    outputs = []

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        n_bands = src.count

        # Skip if the raster doesn't have enough bands for the requested indices
        if n_bands < max(red_band, nir_band, green_band):
            logger.warning(f"Skipping indices for {input_path}: only {n_bands} bands")
            return outputs

        # Read the required bands
        red = src.read(red_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)
        green = src.read(green_band).astype(np.float32)

    # Update profile for single-band float output
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    os.makedirs(output_dir, exist_ok=True)
    base = Path(input_path).stem

    # Compute each index and write to disk
    index_data = {
        "ndvi": indices.ndvi(nir, red),    # Vegetation density
        "ndwi": indices.ndwi(green, nir),  # Water presence
    }

    for name, data in index_data.items():
        out_path = os.path.join(output_dir, f"{base}_{name}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        outputs.append(out_path)
        logger.info(f"Computed {name.upper()}: {out_path}")

    return outputs


# =============================================================================
# Batch Processing
# =============================================================================

def batch_process(input_dir: str, output_dir: str,
                  n_classes: int = 5, method: str = "kmeans",
                  compute_spectral_indices: bool = True,
                  max_workers: int = 4) -> list:
    """Process all raster files in a directory using parallel workers.

    Scans the input directory for supported raster formats (.tif, .tiff,
    .img, .jp2), classifies each one, and saves results to the output
    directory. A consolidated JSON report is generated at the end.

    Args:
        input_dir: Directory containing input raster files.
        output_dir: Directory for classified outputs.
        n_classes: Number of classes for classification.
        method: Clustering method ('kmeans' or 'iso_cluster').
        compute_spectral_indices: Whether to also compute NDVI/NDWI per tile.
        max_workers: Number of parallel worker processes (default 4).

    Returns:
        List of metadata dictionaries, one per processed file.
    """
    # Supported raster formats
    raster_extensions = {".tif", ".tiff", ".img", ".jp2"}
    input_files = [
        os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
        if Path(f).suffix.lower() in raster_extensions
    ]

    if not input_files:
        logger.warning(f"No raster files found in {input_dir}")
        return []

    logger.info(f"Found {len(input_files)} raster files to process")
    results = []

    # Classify rasters in parallel using a process pool.
    # Each worker gets its own copy of the classifier, so this is fully
    # parallelizable with no shared state.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for input_path in input_files:
            stem = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{stem}_classified.tif")
            future = executor.submit(
                classify_raster, input_path, output_path, n_classes, method
            )
            futures[future] = input_path

        # Collect results as they complete (not necessarily in submission order)
        for future in as_completed(futures):
            try:
                meta = future.result()
                results.append(meta)
            except Exception as e:
                logger.error(f"Failed: {futures[future]}: {e}")

    # Optionally compute spectral indices (done sequentially since it's I/O-bound)
    if compute_spectral_indices:
        indices_dir = os.path.join(output_dir, "indices")
        for input_path in input_files:
            try:
                compute_indices(input_path, indices_dir)
            except Exception as e:
                logger.error(f"Indices failed for {input_path}: {e}")

    # Write a consolidated batch report
    report_path = os.path.join(output_dir, "batch_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Batch report: {report_path}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for batch raster classification."""
    parser = argparse.ArgumentParser(description="Batch raster classification")
    parser.add_argument("--input_dir", required=True, help="Input raster directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--method", default="kmeans", choices=["kmeans", "iso_cluster"])
    parser.add_argument("--no_indices", action="store_true", help="Skip spectral indices")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    batch_process(
        args.input_dir, args.output_dir,
        n_classes=args.n_classes,
        method=args.method,
        compute_spectral_indices=not args.no_indices,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
