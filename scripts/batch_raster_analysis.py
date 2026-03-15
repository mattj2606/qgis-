"""
Batch Raster Analysis

Process multiple raster files with consistent classification parameters.
Can run standalone or within the QGIS Python console.

Usage:
    python batch_raster_analysis.py --input_dir ./data --output_dir ./output \
        --n_classes 5 --method kmeans
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

# Add parent directory so we can import the classifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin" / "land_cover_analyzer"))
from classifier import UnsupervisedClassifier, SpectralIndices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def classify_raster(input_path: str, output_path: str,
                    n_classes: int = 5, method: str = "kmeans") -> dict:
    """Classify a single raster file.

    Args:
        input_path: Path to input multispectral raster.
        output_path: Path for classified output GeoTIFF.
        n_classes: Number of land cover classes.
        method: Classification method ('kmeans' or 'iso_cluster').

    Returns:
        Dictionary with classification metadata.
    """
    logger.info(f"Processing: {input_path}")

    with rasterio.open(input_path) as src:
        band_stack = src.read().astype(np.float32)
        profile = src.profile.copy()
        meta = {
            "input": input_path,
            "crs": str(src.crs),
            "bounds": list(src.bounds),
            "shape": list(band_stack.shape),
        }

    classifier = UnsupervisedClassifier(n_classes=n_classes, method=method)
    classified = classifier.fit_predict(band_stack)

    profile.update(
        dtype=rasterio.int32,
        count=1,
        compress="lzw",
        nodata=0,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classified, 1)

    # Compute class distribution
    unique, counts = np.unique(classified, return_counts=True)
    total_pixels = classified.size
    class_dist = {
        int(cls): {"count": int(cnt), "percentage": round(cnt / total_pixels * 100, 2)}
        for cls, cnt in zip(unique, counts)
        if cls > 0  # skip nodata
    }

    meta.update({
        "output": output_path,
        "n_classes": n_classes,
        "method": method,
        "class_distribution": class_dist,
    })

    logger.info(f"Classified: {output_path} ({n_classes} classes)")
    return meta


def compute_indices(input_path: str, output_dir: str,
                    red_band: int = 3, nir_band: int = 4,
                    green_band: int = 2) -> list:
    """Compute spectral indices for a raster.

    Args:
        input_path: Path to multispectral raster.
        output_dir: Directory for output index rasters.
        red_band: 1-based band number for red.
        nir_band: 1-based band number for NIR.
        green_band: 1-based band number for green.

    Returns:
        List of output file paths.
    """
    indices = SpectralIndices()
    outputs = []

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        n_bands = src.count

        if n_bands < max(red_band, nir_band, green_band):
            logger.warning(f"Skipping indices for {input_path}: only {n_bands} bands")
            return outputs

        red = src.read(red_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)
        green = src.read(green_band).astype(np.float32)

    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    os.makedirs(output_dir, exist_ok=True)
    base = Path(input_path).stem

    index_data = {
        "ndvi": indices.ndvi(nir, red),
        "ndwi": indices.ndwi(green, nir),
    }

    for name, data in index_data.items():
        out_path = os.path.join(output_dir, f"{base}_{name}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        outputs.append(out_path)
        logger.info(f"Computed {name.upper()}: {out_path}")

    return outputs


def batch_process(input_dir: str, output_dir: str,
                  n_classes: int = 5, method: str = "kmeans",
                  compute_spectral_indices: bool = True,
                  max_workers: int = 4) -> list:
    """Process all raster files in a directory.

    Args:
        input_dir: Directory containing input rasters.
        output_dir: Directory for outputs.
        n_classes: Number of classes.
        method: Classification method.
        compute_spectral_indices: Whether to compute NDVI/NDWI.
        max_workers: Number of parallel workers.

    Returns:
        List of metadata dictionaries for each processed file.
    """
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for input_path in input_files:
            stem = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{stem}_classified.tif")
            future = executor.submit(
                classify_raster, input_path, output_path, n_classes, method
            )
            futures[future] = input_path

        for future in as_completed(futures):
            try:
                meta = future.result()
                results.append(meta)
            except Exception as e:
                logger.error(f"Failed: {futures[future]}: {e}")

    if compute_spectral_indices:
        indices_dir = os.path.join(output_dir, "indices")
        for input_path in input_files:
            try:
                compute_indices(input_path, indices_dir)
            except Exception as e:
                logger.error(f"Indices failed for {input_path}: {e}")

    # Save batch report
    report_path = os.path.join(output_dir, "batch_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Batch report: {report_path}")

    return results


def main():
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
